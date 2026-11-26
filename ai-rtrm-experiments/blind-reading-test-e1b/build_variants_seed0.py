"""E1b redesign signal check -- variants N and T, SEED 0 ONLY (R-012 s.3).

Gate (fixed in F-013 before running): a variant clears to a reader
test iff some fixed rule scored here reaches balanced accuracy >= 0.65
on seed 0 on at least one task. Seeds 1-2 are NOT built for these
variants. Nothing banked is modified; outputs go to a new folder.

Construction: identical to build_e1b_fresh.py (Study 1's 054c-C2
subtracted list at hs[11] on the fine-tuned model; pool = the cached
pool_e1b_s0.pt records; mu / top-64 V via torch.pca_lowrank with
manual_seed(0); query = unit hs[11] at the probe's final position
minus the pool mean of "is"-labelled states, then residual, then
normalise). Only the pool and the labels change, per variant:

  Variant N: each pool state is labelled by the token that FOLLOWED
  it in its sentence (the last token of a sentence gets "<END>").
  Pool, basis, queries unchanged -- the ranking is the built E1b
  ranking; only the labels move. Rule family: first rank whose
  next-token label is the correct godparent's first name <= k.

  Variant T: the pool is extended with the seed's distinct
  fine-tuning sentences (dataset_fresh.json arms.expD, deduplicated:
  60 forward facts, 30 reversed bridges, 60 fillers), run through
  the same model, one unit hs[11] state per token, each state tagged
  with its source sentence and pair. Basis and "is" mean recomputed
  on the extended pool. Neighbours are shown as source sentences
  with the matched token marked. Rule families: (i) first rank whose
  source sentence contains the godparent's first name <= k;
  (ii) any top-k neighbour from this pair's own sentence (reversed
  sentence for Task 2; the pair's fact sentence in the probe's
  direction for Task 1; any of the pair's FT sentences as a
  secondary family).

Built-in gate: before any variant is scored, the current-token
top-20s recomputed here must reproduce lists_seed0.json exactly
(60/60); any mismatch aborts.

Extra statistics for T (R-012 s.3): share of the top 20 from
fine-tuning sentences; rank of the first neighbour whose source
mentions the godchild; rank of the first neighbour from the pair's
own reversed (stored) or forward (forward-only) sentence, by subset.
Forward-only figure data: for each forward-only pair, the rank of
its forward sentence's best state while the model fails the probe.
"""
import json, re, sys, time
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

EXP = "/home/ecpi-student-05/robot-mind/exp-201-exposure-fresh"
P2 = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
E1B = f"{EXP}/blind-test-e1b"
OUT = f"{EXP}/variants-seed0"
K = 64
GATE = 0.65
SEED = 0


def word_of(tok, i):
    return tok.decode([i]).strip().lower()


def balanced_accuracy(preds, truths):
    tp = sum(1 for p, t in zip(preds, truths) if p and t)
    fn = sum(1 for p, t in zip(preds, truths) if not p and t)
    tn = sum(1 for p, t in zip(preds, truths) if not p and not t)
    fp = sum(1 for p, t in zip(preds, truths) if p and not t)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return (sens + spec) / 2.0


def residual_setup(M):
    mu = M.mean(0)
    torch.manual_seed(0)
    U, S, V = torch.pca_lowrank(M - mu, q=K)
    return mu, V


def coderes(H, mu, V):
    Hc = H - mu
    return Hc - (Hc @ V) @ V.T


def ba_table(rank_list, truth, kmax=20):
    """rank_list: per-item rank (int) or None. Rule: rank <= k."""
    table = {}
    for k in range(1, kmax + 1):
        preds = [(r is not None and r <= k) for r in rank_list]
        table[k] = round(balanced_accuracy(preds, truth), 4)
    best_k = min(table, key=lambda k: (-table[k], k))
    return {"table": {str(k): v for k, v in table.items()},
            "best_k": best_k, "best_ba": table[best_k]}


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    behavior = json.load(open(f"{EXP}/behavior_fresh.json"))
    dataset = json.load(open(f"{EXP}/dataset_fresh.json"))
    pairs = dataset["pairs"]              # [godparent, godchild]
    rev_idx = set(dataset["rev_idx"])
    brecs = {r["index"]: r for r in behavior["models"][f"expD_s{SEED}"]["reversed"]}
    for i in range(60):
        assert brecs[i]["expected_name"] == pairs[i][0], (i, "godparent mismatch")
        assert brecs[i]["godchild"] == pairs[i][1], (i, "godchild mismatch")
    truth1 = [bool(brecs[i]["pass_8tok"]) for i in range(60)]
    truth2 = [(i in rev_idx) for i in range(60)]

    # ---------------- wiki pool (cached records from the E1b build) ----
    recs = torch.load(f"{E1B}/pool_e1b_s{SEED}.pt", map_location="cpu",
                      weights_only=False)
    rows = [json.loads(l) for l in open(P2 + "/pool_build_selection.jsonl")]
    sent_text = {r["corpus_index"]: r["sentence"] for r in rows}
    vecs, words, next_words, srcs = [], [], [], []
    for r in recs:
        ids = r["ids"]
        text = sent_text[int(r["id"].split("-")[1])]
        for p, v in enumerate(r["hs11"]):
            vecs.append(v.float())
            words.append(word_of(tok, ids[p]))
            next_words.append(word_of(tok, ids[p + 1]) if p + 1 < len(ids) else "<END>")
            srcs.append({"kind": "wiki", "pair": None, "text": text,
                         "pos": p, "ids": ids})
    M = torch.stack(vecs)
    assert float((M.norm(dim=1) - 1.0).abs().max()) < 1e-4, "pool norms not unit"
    mu, V = residual_setup(M)
    Rn = F.normalize(coderes(M, mu, V), dim=1)
    is_mu = M[torch.tensor([w == "is" for w in words])].mean(0)
    print(f"wiki pool: {len(words)} states", flush=True)

    model = GPT2LMHeadModel.from_pretrained(f"{EXP}/models/expD_s{SEED}").eval()

    def query_sims(i, mu_, V_, Rn_, is_mu_):
        ids = torch.tensor([tok(brecs[i]["probe"])["input_ids"]])
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        h = F.normalize(hs[11][0].float(), dim=1)
        q1 = h[-1].unsqueeze(0) - is_mu_
        q2 = coderes(q1, mu_, V_)
        return (F.normalize(q2, dim=1) @ Rn_.T)[0]

    # ---------------- built-in gate: reproduce lists_seed0 ------------
    built = {x["index"]: x for x in json.load(open(f"{E1B}/lists_seed0.json"))["items"]}
    orders = {}
    bad = 0
    for i in range(60):
        sims = query_sims(i, mu, V, Rn, is_mu)
        order = torch.argsort(sims, descending=True)
        orders[i] = order
        top20 = [words[j] for j in order[:20]]
        if top20 != built[i]["top20"]:
            bad += 1
            print(f"  GATE MISMATCH item {i}", flush=True)
    if bad:
        print(f"BUILT-IN GATE FAILED: {bad}/60 top-20s differ -- ABORT",
              file=sys.stderr)
        sys.exit(1)
    print("built-in gate PASSED: recomputed current-token top-20s == "
          "lists_seed0.json (60/60)", flush=True)

    # ---------------- variant N ---------------------------------------
    itemsN = []
    for i in range(60):
        order = orders[i]
        order_next = [next_words[j] for j in order]
        first = pairs[i][0].split()[0].lower()
        rank = next((r + 1 for r, w in enumerate(order_next) if w == first), None)
        itemsN.append({"index": i, "probe": brecs[i]["probe"],
                       "godparent": pairs[i][0], "godchild": pairs[i][1],
                       "subset": "stored-trained" if i in rev_idx else "forward-only",
                       "pass_8tok": truth1[i],
                       "top20_next": order_next[:20],
                       "name_next_rank": rank})
    tabN = {"task1": ba_table([x["name_next_rank"] for x in itemsN], truth1),
            "task2": ba_table([x["name_next_rank"] for x in itemsN], truth2)}
    json.dump({"variant": "N", "seed": SEED, "items": itemsN, "bar_tables": tabN},
              open(f"{OUT}/variantN_seed0.json", "w"), indent=1)
    print(f"variant N: task1 best BA {tabN['task1']['best_ba']} (k={tabN['task1']['best_k']}), "
          f"task2 best BA {tabN['task2']['best_ba']} (k={tabN['task2']['best_k']})", flush=True)
    del Rn, M

    # ---------------- variant T ---------------------------------------
    # distinct FT sentences with pair/kind attribution
    ft = []  # (text, kind, pair)
    for i, (gp, gc) in enumerate(pairs):
        ft.append((f"{gp} is the godparent of {gc}.", "forward", i))
    rev_by_pair = {}
    for s in dataset["rev_bridges"]:
        for i in rev_idx:
            gp, gc = pairs[i]
            if gc in s and gp in s:
                rev_by_pair[i] = s
    assert len(rev_by_pair) == 30, len(rev_by_pair)
    for i, s in rev_by_pair.items():
        ft.append((s, "reversed", i))
    filler_pair = {}
    for s in dataset["fillers"]:
        hits = [i for i, (gp, gc) in enumerate(pairs) if gp in s]
        assert len(hits) == 1, (s, hits)
        filler_pair[s] = hits[0]
        ft.append((s, "filler", hits[0]))
    distinct_arm = set(dataset["arms"]["expD"])
    assert set(t for t, _, _ in ft) == distinct_arm, (
        f"attribution covers {len(set(t for t,_,_ in ft))} of {len(distinct_arm)} distinct arm texts")
    print(f"variant T: {len(ft)} distinct FT sentences "
          f"(60 forward, 30 reversed, 60 filler) == distinct arm texts", flush=True)

    vecsT, wordsT, nextT, srcsT = list(vecs), list(words), list(next_words), list(srcs)
    with torch.no_grad():
        for text, kind, i in ft:
            ids = tok(text)["input_ids"]
            hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
            H = F.normalize(hs[11][0].float(), dim=1)
            for p in range(len(ids)):
                vecsT.append(H[p])
                wordsT.append(word_of(tok, ids[p]))
                nextT.append(word_of(tok, ids[p + 1]) if p + 1 < len(ids) else "<END>")
                srcsT.append({"kind": kind, "pair": i, "text": text,
                              "pos": p, "ids": ids})
    MT = torch.stack(vecsT)
    assert float((MT.norm(dim=1) - 1.0).abs().max()) < 1e-4, "ext pool norms"
    muT, VT = residual_setup(MT)
    RnT = F.normalize(coderes(MT, muT, VT), dim=1)
    is_muT = MT[torch.tensor([w == "is" for w in wordsT])].mean(0)
    print(f"variant T pool: {len(wordsT)} states ({len(wordsT)-len(words)} FT)", flush=True)

    def marked(src):
        ids, p = src["ids"], src["pos"]
        toks = [tok.decode([t]) for t in ids]
        toks[p] = "⟦" + toks[p] + "⟧"
        return "".join(toks).strip()

    def first_rank(order, pred):
        for r, j in enumerate(order):
            if pred(srcsT[int(j)]):
                return r + 1
        return None

    itemsT = []
    for i in range(60):
        sims = query_sims(i, muT, VT, RnT, is_muT)
        order = torch.argsort(sims, descending=True).tolist()
        gp_first = pairs[i][0].split()[0]
        gc_first = pairs[i][1].split()[0]
        name_src_rank = first_rank(order, lambda s: re.search(
            rf"\b{re.escape(gp_first)}\b", s["text"], re.I))
        godchild_rank = first_rank(order, lambda s: re.search(
            rf"\b{re.escape(gc_first)}\b", s["text"], re.I))
        pair_rev_rank = first_rank(order, lambda s: s["pair"] == i and s["kind"] == "reversed")
        pair_fwd_rank = first_rank(order, lambda s: s["pair"] == i and s["kind"] == "forward")
        pair_any_rank = first_rank(order, lambda s: s["pair"] == i and s["kind"] != "wiki")
        order_nextT = [nextT[j] for j in order]
        name_next_rankT = next((r + 1 for r, w in enumerate(order_nextT)
                                if w == gp_first.lower()), None)
        top20 = []
        for r, j in enumerate(order[:20]):
            s = srcsT[j]
            top20.append({"rank": r + 1, "kind": s["kind"], "pair": s["pair"],
                          "marked": marked(s)})
        ft_share = sum(1 for e in top20 if e["kind"] != "wiki") / 20.0
        itemsT.append({
            "index": i, "probe": brecs[i]["probe"], "godparent": pairs[i][0],
            "godchild": pairs[i][1],
            "subset": "stored-trained" if i in rev_idx else "forward-only",
            "pass_8tok": truth1[i], "top20": top20, "top20_ft_share": ft_share,
            "name_in_source_rank": name_src_rank,
            "godchild_mention_rank": godchild_rank,
            "pair_reversed_rank": pair_rev_rank,
            "pair_forward_rank": pair_fwd_rank,
            "pair_any_ft_rank": pair_any_rank,
            "name_next_rank_extpool": name_next_rankT,
        })
    fam = {
        "T-src-name": {"task1": ba_table([x["name_in_source_rank"] for x in itemsT], truth1),
                       "task2": ba_table([x["name_in_source_rank"] for x in itemsT], truth2)},
        "T-pair-rev": {"task1": ba_table([x["pair_reversed_rank"] for x in itemsT], truth1),
                       "task2": ba_table([x["pair_reversed_rank"] for x in itemsT], truth2)},
        "T-pair-fact": {"task1": ba_table(
                            [x["pair_reversed_rank"] if x["subset"] == "stored-trained"
                             else x["pair_forward_rank"] for x in itemsT], truth1),
                        "task2": ba_table(
                            [x["pair_reversed_rank"] if x["subset"] == "stored-trained"
                             else x["pair_forward_rank"] for x in itemsT], truth2)},
        "T-pair-any": {"task1": ba_table([x["pair_any_ft_rank"] for x in itemsT], truth1),
                       "task2": ba_table([x["pair_any_ft_rank"] for x in itemsT], truth2)},
        "T-next-label": {"task1": ba_table([x["name_next_rank_extpool"] for x in itemsT], truth1),
                         "task2": ba_table([x["name_next_rank_extpool"] for x in itemsT], truth2)},
    }
    json.dump({"variant": "T", "seed": SEED, "items": itemsT, "bar_tables": fam},
              open(f"{OUT}/variantT_seed0.json", "w"), indent=1)
    for name, tt in fam.items():
        print(f"variant T {name}: task1 best BA {tt['task1']['best_ba']} (k={tt['task1']['best_k']}), "
              f"task2 best BA {tt['task2']['best_ba']} (k={tt['task2']['best_k']})", flush=True)

    # ---------------- gate evaluation ---------------------------------
    bestN = max(tabN[t]["best_ba"] for t in ("task1", "task2"))
    bestT = max(fam[f][t]["best_ba"] for f in fam for t in ("task1", "task2"))
    fwd_only = [x for x in itemsT if x["subset"] == "forward-only"]
    fig = {"forward_only_pairs": len(fwd_only),
           "forward_sentence_in_top20": sum(
               1 for x in fwd_only if x["pair_forward_rank"] is not None
               and x["pair_forward_rank"] <= 20),
           "all_forward_only_fail_probe": all(not x["pass_8tok"] for x in fwd_only),
           "per_pair_forward_rank": {str(x["index"]): x["pair_forward_rank"]
                                     for x in fwd_only}}
    summary = {
        "gate_fixed_in_F013": GATE,
        "variantN_best_fixed_rule_BA": bestN,
        "variantT_best_fixed_rule_BA": bestT,
        "variantN_gate": "MET" if bestN >= GATE else "MISSED",
        "variantT_gate": "MET" if bestT >= GATE else "MISSED",
        "variantT_extras": {
            "mean_top20_ft_share_stored": round(sum(
                x["top20_ft_share"] for x in itemsT if x["subset"] == "stored-trained") / 30, 4),
            "mean_top20_ft_share_forward_only": round(sum(
                x["top20_ft_share"] for x in itemsT if x["subset"] == "forward-only") / 30, 4),
            "godchild_mention_rank_median_stored": sorted(
                x["godchild_mention_rank"] for x in itemsT
                if x["subset"] == "stored-trained" and x["godchild_mention_rank"])[15],
            "godchild_mention_rank_median_forward_only": sorted(
                x["godchild_mention_rank"] for x in itemsT
                if x["subset"] == "forward-only" and x["godchild_mention_rank"])[15],
        },
        "variantT_forward_only_figure": fig,
    }
    json.dump(summary, open(f"{OUT}/signal_check_seed0.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print("VARIANTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
