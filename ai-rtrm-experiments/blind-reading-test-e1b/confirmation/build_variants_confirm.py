"""Confirmation on seeds 1-2 (R-014 s.4): variant T builds for the
two reserved seeds, scored against bars C1-C4 LOCKED in
CONFIRMATION_BARS_LOCKED.md (PR #18, commit 05f59f3) BEFORE this
build ran. Paul approved the proposal 2026-11-22.

Construction is build_variants_seed0.py's variant T, unchanged:
cached E1b pool per seed, extended with the 150 distinct
fine-tuning sentences through the seed's own fine-tuned model;
basis and "is" mean recomputed on the extended pool; same rule
families as ec36f83. Built-in gate per seed: recomputed
current-token top-20s must reproduce lists_seed{seed}.json (60/60)
before anything is scored.

C2 and the BM25 legs of C3/C4 are seed-independent (one dataset;
the untrained model and BM25 have no seed) and are read here from
the seed-0 control records, with their counts asserted, per the
lock file's clarification 1.

Outputs (new folder variants-confirm/): variantT_seed1.json,
variantT_seed2.json, confirmation_summary.json.
"""
import json, re, sys
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

EXP = "/home/ecpi-student-05/robot-mind/exp-201-exposure-fresh"
P2 = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
E1B = f"{EXP}/blind-test-e1b"
OUT = f"{EXP}/variants-confirm"
K = 64
SEEDS = [1, 2]


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
    table = {}
    for k in range(1, kmax + 1):
        preds = [(r is not None and r <= k) for r in rank_list]
        table[k] = round(balanced_accuracy(preds, truth), 4)
    best_k = min(table, key=lambda k: (-table[k], k))
    return {"table": {str(k): v for k, v in table.items()},
            "best_k": best_k, "best_ba": table[best_k]}


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    behavior = json.load(open(f"{EXP}/behavior_fresh.json"))
    dataset = json.load(open(f"{EXP}/dataset_fresh.json"))
    pairs = dataset["pairs"]
    rev_idx = set(dataset["rev_idx"])
    truth2 = [(i in rev_idx) for i in range(60)]

    # seed-independent control legs (lock clarification 1) ----------
    ctrlU = json.load(open(f"{EXP}/variants-seed0/controls/controlU_seed0.json"))
    ctrlB = json.load(open(f"{EXP}/variants-seed0/controls/controlB_seed0.json"))
    ctrlS = json.load(open(f"{EXP}/variants-seed0/controls/controlB_scores.json"))
    u_top1 = sum(1 for x in ctrlU["items"]
                 if x["subset"] == "stored-trained" and x["pair_reversed_rank"] == 1)
    ties = ctrlS["summary"]["pairs_with_equal_scores"]
    bm25_first = sum(1 for x in ctrlB["items"]
                     if x["subset"] == "forward-only" and x["forward_sentence_rank"] == 1)
    print(f"control legs (seed-independent): C2 U top1 = {u_top1}/30; "
          f"C3 ties = {ties}/30; C4 BM25 forward-first = {bm25_first}/30", flush=True)
    assert u_top1 == 0 and ties == 30 and bm25_first == 30

    # FT sentence attribution (shared across seeds) ------------------
    ft = []
    for i, (gp, gc) in enumerate(pairs):
        ft.append((f"{gp} is the godparent of {gc}.", "forward", i))
    rev_by_pair = {}
    for s in dataset["rev_bridges"]:
        for i in rev_idx:
            gp, gc = pairs[i]
            if gc in s and gp in s:
                rev_by_pair[i] = s
    assert len(rev_by_pair) == 30
    for i, s in rev_by_pair.items():
        ft.append((s, "reversed", i))
    for s in dataset["fillers"]:
        hits = [i for i, (gp, gc) in enumerate(pairs) if gp in s]
        assert len(hits) == 1
        ft.append((s, "filler", hits[0]))
    assert set(t for t, _, _ in ft) == set(dataset["arms"]["expD"])

    rows = [json.loads(l) for l in open(P2 + "/pool_build_selection.jsonl")]
    sent_text = {r["corpus_index"]: r["sentence"] for r in rows}

    per_seed = {}
    for SEED in SEEDS:
        brecs = {r["index"]: r for r in behavior["models"][f"expD_s{SEED}"]["reversed"]}
        for i in range(60):
            assert brecs[i]["expected_name"] == pairs[i][0]
            assert brecs[i]["godchild"] == pairs[i][1]
        truth1 = [bool(brecs[i]["pass_8tok"]) for i in range(60)]

        recs = torch.load(f"{E1B}/pool_e1b_s{SEED}.pt", map_location="cpu",
                          weights_only=False)
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
        assert float((M.norm(dim=1) - 1.0).abs().max()) < 1e-4
        mu, V = residual_setup(M)
        Rn = F.normalize(coderes(M, mu, V), dim=1)
        is_mu = M[torch.tensor([w == "is" for w in words])].mean(0)
        model = GPT2LMHeadModel.from_pretrained(f"{EXP}/models/expD_s{SEED}").eval()

        def query_sims(i, mu_, V_, Rn_, is_mu_):
            ids = torch.tensor([tok(brecs[i]["probe"])["input_ids"]])
            with torch.no_grad():
                hs = model(ids, output_hidden_states=True).hidden_states
            h = F.normalize(hs[11][0].float(), dim=1)
            q1 = h[-1].unsqueeze(0) - is_mu_
            q2 = coderes(q1, mu_, V_)
            return (F.normalize(q2, dim=1) @ Rn_.T)[0]

        built = {x["index"]: x for x in
                 json.load(open(f"{E1B}/lists_seed{SEED}.json"))["items"]}
        bad = 0
        for i in range(60):
            order = torch.argsort(query_sims(i, mu, V, Rn, is_mu), descending=True)
            if [words[j] for j in order[:20]] != built[i]["top20"]:
                bad += 1
        if bad:
            print(f"SEED {SEED}: BUILT-IN GATE FAILED {bad}/60 -- ABORT",
                  file=sys.stderr)
            sys.exit(1)
        print(f"seed {SEED}: built-in gate PASSED (60/60)", flush=True)
        del Rn, M

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
        assert float((MT.norm(dim=1) - 1.0).abs().max()) < 1e-4
        muT, VT = residual_setup(MT)
        RnT = F.normalize(coderes(MT, muT, VT), dim=1)
        is_muT = MT[torch.tensor([w == "is" for w in wordsT])].mean(0)
        print(f"seed {SEED}: variant T pool {len(wordsT)} states", flush=True)

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
            top20 = []
            for r, j in enumerate(order[:20]):
                s = srcsT[j]
                top20.append({"rank": r + 1, "kind": s["kind"], "pair": s["pair"],
                              "marked": marked(s)})
            itemsT.append({
                "index": i, "probe": brecs[i]["probe"], "godparent": pairs[i][0],
                "godchild": pairs[i][1],
                "subset": "stored-trained" if i in rev_idx else "forward-only",
                "pass_8tok": truth1[i], "top20": top20,
                "top20_ft_share": sum(1 for e in top20 if e["kind"] != "wiki") / 20.0,
                "name_in_source_rank": first_rank(order, lambda s: re.search(
                    rf"\b{re.escape(gp_first)}\b", s["text"], re.I)),
                "godchild_mention_rank": first_rank(order, lambda s: re.search(
                    rf"\b{re.escape(gc_first)}\b", s["text"], re.I)),
                "pair_reversed_rank": first_rank(
                    order, lambda s: s["pair"] == i and s["kind"] == "reversed"),
                "pair_forward_rank": first_rank(
                    order, lambda s: s["pair"] == i and s["kind"] == "forward"),
                "pair_any_ft_rank": first_rank(
                    order, lambda s: s["pair"] == i and s["kind"] != "wiki"),
                "name_next_rank_extpool": next(
                    (r + 1 for r, j in enumerate(order) if nextT[j] == gp_first.lower()),
                    None),
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
                  open(f"{OUT}/variantT_seed{SEED}.json", "w"), indent=1)

        stored = [x for x in itemsT if x["subset"] == "stored-trained"]
        fwd = [x for x in itemsT if x["subset"] == "forward-only"]
        passes = [x for x in stored if x["pass_8tok"]]
        fails = [x for x in stored if not x["pass_8tok"]]
        per_seed[SEED] = {
            "C1_top1_count": sum(1 for x in stored if x["pair_reversed_rank"] == 1),
            "C4_list_top20_count": sum(1 for x in fwd
                                       if x["pair_forward_rank"] is not None
                                       and x["pair_forward_rank"] <= 20),
            "C4_answers": sum(1 for x in fwd if x["pass_8tok"]),
            "pass_fail_split_stored": {
                "n_pass": len(passes), "n_fail": len(fails),
                "top1_among_pass": sum(1 for x in passes if x["pair_reversed_rank"] == 1),
                "top1_among_fail": sum(1 for x in fails if x["pair_reversed_rank"] == 1),
                "median_rev_rank_pass": median(
                    [x["pair_reversed_rank"] for x in passes
                     if x["pair_reversed_rank"] is not None]),
                "median_rev_rank_fail": median(
                    [x["pair_reversed_rank"] for x in fails
                     if x["pair_reversed_rank"] is not None]),
            },
        }
        print(f"seed {SEED}: C1 top1 = {per_seed[SEED]['C1_top1_count']}/30; "
              f"C4 list top20 = {per_seed[SEED]['C4_list_top20_count']}/30, "
              f"answers = {per_seed[SEED]['C4_answers']}/30", flush=True)
        del MT, RnT, vecs, vecsT, model

    bars = {}
    for SEED in SEEDS:
        bars[SEED] = {
            "C1": per_seed[SEED]["C1_top1_count"] >= 20,
            "C2": u_top1 <= 13,
            "C3": ties >= 28,
            "C4": (bm25_first >= 28 and per_seed[SEED]["C4_list_top20_count"] <= 10
                   and per_seed[SEED]["C4_answers"] == 0),
        }
    all_hold = all(all(b.values()) for b in bars.values())
    c1c2 = all(b["C1"] and b["C2"] for b in bars.values())
    summary = {
        "bars_locked": "CONFIRMATION_BARS_LOCKED.md, PR #18 commit 05f59f3 (before this build)",
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "control_legs_seed_independent": {"C2_U_top1": u_top1, "C3_ties": ties,
                                          "C4_BM25_forward_first": bm25_first},
        "bars": {str(s): bars[s] for s in SEEDS},
        "outcome": ("ALL BARS HOLD on both seeds" if all_hold else
                    "C1 and C2 hold on both seeds; C3 or C4 failed" if c1c2 else
                    "C1 or C2 FAILED — seed 0 did not replicate"),
    }
    json.dump(summary, open(f"{OUT}/confirmation_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print("CONFIRMATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
