"""Older-models run (R-017 s.3): variant T + controls on the exp-201
token-exposure arm-D models -- a DIFFERENT dataset (60 other names,
another split) from the E1b confirmation. Bars O1-O4 locked in
blind-reading-test-e1b/older-models/BARS_LOCKED.md (PR #18 commit
91dda54) before this build; the lock hash was posted in F-019 before
the run, per R-017 s.5.

Construction identical to build_variants_seed0.py's variant T (Study
1's 054c-C2 subtracted list at hs[11]; extended pool = WikiText
selection sentences + the dataset's distinct fine-tuning sentences,
each state tagged with its source; mu / top-64 V via torch.pca_lowrank
manual_seed(0); query = unit hs[11] at the probe's final position
minus the pool mean of "is"-labelled states, then residual, then
normalise). Core functions copied verbatim from the gated E1b scripts.
No banked neighbour lists exist for these models (first RTRM run on
them), so there is no list gate; the footing asserts are the behaviour
records' names against the canonical pairs, the FT-attribution
coverage assert against the distinct arm texts, and unit pool norms.

Per seed s in {0,1,2} (models/expD_s{s}):
  O1: rev-trained pairs with own reversed sentence at rank 1 (bar >=20)
  O4 list leg: forward-only pairs with own forward sentence in top 20
       (bar <=10); O4 answers leg from the banked behaviour (bar 0)
Controls, fresh on THIS dataset:
  O2: Control U (untrained gpt2, same construction), rank-1 count
      (bar <=13)
  O3: BM25 (Okapi k1=1.5 b=0.75, corpus = wiki selection sentences +
      FT sentences, query = probe) -- forward vs reversed SCORE
      equality on rev-trained pairs (bar >=28 ties)
  O4 BM25 leg: forward-only pairs with forward sentence at rank 1
      (bar >=28)
Outputs to a new folder; nothing banked is modified.
"""
import json, math, os, re, sys, time
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

E201 = "/home/ecpi-student-05/robot-mind/exp-201-exposure"
D201 = "/home/ecpi-student-05/robot-mind/exp-201/dataset_201.json"
P2 = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
OUT = f"{E201}/older-models-variantT"
K = 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    pairs = json.load(open(D201))["pairs"]            # [godparent, godchild]
    dataset = json.load(open(f"{E201}/dataset_exposure.json"))
    rev_idx = set(dataset["rev_idx"])
    behavior = json.load(open(f"{E201}/behavior_exposure.json"))

    rows = [json.loads(l) for l in open(P2 + "/pool_build_selection.jsonl")]
    wiki_texts = [r["sentence"] for r in rows]

    # ---- FT sentences with pair/kind attribution (as the E1b build) --
    ft = []
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
    for s in dataset["fillers"]:
        hits = [i for i, (gp, gc) in enumerate(pairs) if gp in s]
        assert len(hits) == 1, (s, hits)
        ft.append((s, "filler", hits[0]))
    distinct_arm = set(dataset["arms"]["expD"])
    assert set(t for t, _, _ in ft) == distinct_arm, "FT attribution mismatch"
    print(f"FT: {len(ft)} distinct sentences == distinct arm-D texts", flush=True)

    def extract(model, texts):
        vecs, words, next_words = [], [], []
        with torch.no_grad():
            for text in texts:
                ids = tok(text)["input_ids"]
                hs = model(torch.tensor([ids], device=DEV),
                           output_hidden_states=True).hidden_states
                H = F.normalize(hs[11][0].float(), dim=1).cpu()
                for p in range(len(ids)):
                    vecs.append(H[p])
                    words.append(word_of(tok, ids[p]))
                    next_words.append(word_of(tok, ids[p + 1])
                                      if p + 1 < len(ids) else "<END>")
        return vecs, words, next_words

    def build_items(model, brecs):
        vecs, words, next_words = extract(model, wiki_texts)
        srcs = []
        for t in wiki_texts:
            ids = tok(t)["input_ids"]
            for p in range(len(ids)):
                srcs.append({"kind": "wiki", "pair": None, "text": t,
                             "pos": p, "ids": ids})
        fv, fw, fnw = extract(model, [t for t, _, _ in ft])
        vecs += fv; words += fw; next_words += fnw
        for (text, kind, i) in ft:
            ids = tok(text)["input_ids"]
            for p in range(len(ids)):
                srcs.append({"kind": kind, "pair": i, "text": text,
                             "pos": p, "ids": ids})
        M = torch.stack(vecs)
        assert float((M.norm(dim=1) - 1.0).abs().max()) < 1e-4, "pool norms"
        mu, V = residual_setup(M)
        Rn = F.normalize(coderes(M, mu, V), dim=1)
        is_mu = M[torch.tensor([w == "is" for w in words])].mean(0)
        print(f"  extended pool: {len(words)} states", flush=True)

        def marked(src):
            ids, p = src["ids"], src["pos"]
            toks = [tok.decode([t]) for t in ids]
            toks[p] = "⟦" + toks[p] + "⟧"
            return "".join(toks).strip()

        def first_rank(order, pred):
            for r, j in enumerate(order):
                if pred(srcs[int(j)]):
                    return r + 1
            return None

        items = []
        for i in range(60):
            ids = torch.tensor([tok(brecs[i]["probe"])["input_ids"]], device=DEV)
            with torch.no_grad():
                hs = model(ids, output_hidden_states=True).hidden_states
            h = F.normalize(hs[11][0].float(), dim=1).cpu()
            q1 = h[-1].unsqueeze(0) - is_mu
            q2 = coderes(q1, mu, V)
            sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
            order = torch.argsort(sims, descending=True).tolist()
            gp_first = pairs[i][0].split()[0]
            gc_first = pairs[i][1].split()[0]
            top20 = []
            for r, j in enumerate(order[:20]):
                s = srcs[j]
                top20.append({"rank": r + 1, "kind": s["kind"],
                              "pair": s["pair"], "marked": marked(s)})
            items.append({
                "index": i, "probe": brecs[i]["probe"],
                "godparent": pairs[i][0], "godchild": pairs[i][1],
                "subset": "stored-trained" if i in rev_idx else "forward-only",
                "pass_8tok": bool(brecs[i]["pass_8tok"]),
                "top20": top20,
                "top20_ft_share": sum(1 for e in top20
                                      if e["kind"] != "wiki") / 20.0,
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
            })
        return items

    def families(items, truth1, truth2):
        return {
            "T-src-name": {"task1": ba_table([x["name_in_source_rank"] for x in items], truth1),
                           "task2": ba_table([x["name_in_source_rank"] for x in items], truth2)},
            "T-pair-rev": {"task1": ba_table([x["pair_reversed_rank"] for x in items], truth1),
                           "task2": ba_table([x["pair_reversed_rank"] for x in items], truth2)},
            "T-pair-fact": {"task1": ba_table(
                                [x["pair_reversed_rank"] if x["subset"] == "stored-trained"
                                 else x["pair_forward_rank"] for x in items], truth1),
                            "task2": ba_table(
                                [x["pair_reversed_rank"] if x["subset"] == "stored-trained"
                                 else x["pair_forward_rank"] for x in items], truth2)},
            "T-pair-any": {"task1": ba_table([x["pair_any_ft_rank"] for x in items], truth1),
                           "task2": ba_table([x["pair_any_ft_rank"] for x in items], truth2)},
        }

    def split(items):
        st = [x for x in items if x["subset"] == "stored-trained"]
        ps = [x for x in st if x["pass_8tok"]]
        fs = [x for x in st if not x["pass_8tok"]]
        def med(xs):
            xs = sorted(x for x in xs if x is not None)
            return xs[len(xs) // 2] if xs else None
        return {"n_pass": len(ps), "n_fail": len(fs),
                "top1_among_pass": sum(1 for x in ps if x["pair_reversed_rank"] == 1),
                "top1_among_fail": sum(1 for x in fs if x["pair_reversed_rank"] == 1),
                "median_rev_rank_pass": med([x["pair_reversed_rank"] for x in ps]),
                "median_rev_rank_fail": med([x["pair_reversed_rank"] for x in fs])}

    per_seed, items_by_seed = {}, {}
    for s in (0, 1, 2):
        brecs = {r["index"]: r for r in behavior["models"][f"expD_s{s}"]["reversed"]}
        for i in range(60):
            assert brecs[i]["expected_name"] == pairs[i][0]
            assert brecs[i]["godchild"] == pairs[i][1]
        truth1 = [bool(brecs[i]["pass_8tok"]) for i in range(60)]
        truth2 = [(i in rev_idx) for i in range(60)]
        print(f"seed {s}: building variant T for expD_s{s} ...", flush=True)
        model = GPT2LMHeadModel.from_pretrained(
            f"{E201}/models/expD_s{s}").to(DEV).eval()
        items = build_items(model, brecs)
        items_by_seed[s] = items
        json.dump({"variant": "T", "model": f"expD_s{s}", "items": items,
                   "bar_tables": families(items, truth1, truth2)},
                  open(f"{OUT}/variantT_expD_s{s}.json", "w"), indent=1)
        stored = [x for x in items if x["subset"] == "stored-trained"]
        fwd = [x for x in items if x["subset"] == "forward-only"]
        per_seed[s] = {
            "O1_top1_count": sum(1 for x in stored if x["pair_reversed_rank"] == 1),
            "O4_list_top20_count": sum(1 for x in fwd
                                       if x["pair_forward_rank"] is not None
                                       and x["pair_forward_rank"] <= 20),
            "O4_answers": sum(1 for x in fwd if x["pass_8tok"]),
            "pass_fail_split_stored": split(items),
        }
        print(f"seed {s}: O1 top1 = {per_seed[s]['O1_top1_count']}/30; "
              f"O4 list top20 = {per_seed[s]['O4_list_top20_count']}/30, "
              f"answers = {per_seed[s]['O4_answers']}/30", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---------------- Control U (untrained) ---------------------------
    print("control U: building with untrained gpt2 ...", flush=True)
    brecs0 = {r["index"]: r for r in behavior["models"]["expD_s0"]["reversed"]}
    base = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    itemsU = build_items(base, brecs0)
    json.dump({"control": "U (untrained model, variant-T construction)",
               "items": itemsU}, open(f"{OUT}/controlU_exp201.json", "w"), indent=1)
    o2 = sum(1 for x in itemsU if x["subset"] == "stored-trained"
             and x["pair_reversed_rank"] == 1)
    print(f"control U: O2 top1 = {o2}/30", flush=True)
    del base
    torch.cuda.empty_cache()

    # ---------------- BM25 (fresh on this dataset) --------------------
    docs = [(t, "wiki", None) for t in wiki_texts] + ft
    def tokens(text):
        return re.findall(r"[a-z0-9]+", text.lower())
    doc_tokens = [tokens(t) for t, _, _ in docs]
    N = len(docs)
    df = Counter()
    for dt in doc_tokens:
        for w in set(dt):
            df[w] += 1
    idf = {w: math.log(1 + (N - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    avgdl = sum(len(dt) for dt in doc_tokens) / N
    k1, b = 1.5, 0.75
    doc_tf = [Counter(dt) for dt in doc_tokens]

    def bm25_scores(query):
        q = tokens(query)
        out = []
        for d in range(N):
            dl = len(doc_tokens[d])
            sc = 0.0
            for w in q:
                f = doc_tf[d].get(w, 0)
                if f:
                    sc += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
            out.append(sc)
        return out

    ft_doc_index = {}
    for d, (t, kind, i) in enumerate(docs):
        if kind in ("forward", "reversed"):
            ft_doc_index[(kind, i)] = d
    itemsB, ties, bm_o4_first = [], 0, 0
    for i in range(60):
        scores = bm25_scores(brecs0[i]["probe"])
        order = sorted(range(N), key=lambda d: -scores[d])
        pos = {d: r + 1 for r, d in enumerate(order)}
        rec = {"index": i, "probe": brecs0[i]["probe"],
               "subset": "stored-trained" if i in rev_idx else "forward-only",
               "forward_sentence_rank": pos[ft_doc_index[("forward", i)]],
               "forward_score": scores[ft_doc_index[("forward", i)]],
               "reversed_sentence_rank": None, "reversed_score": None}
        if i in rev_idx:
            rec["reversed_sentence_rank"] = pos[ft_doc_index[("reversed", i)]]
            rec["reversed_score"] = scores[ft_doc_index[("reversed", i)]]
            if rec["forward_score"] == rec["reversed_score"]:
                ties += 1
        else:
            if rec["forward_sentence_rank"] == 1:
                bm_o4_first += 1
        itemsB.append(rec)
    json.dump({"control": "B (BM25 over wiki + FT sentences, exp-201 texts)",
               "items": itemsB}, open(f"{OUT}/controlB_exp201.json", "w"), indent=1)
    print(f"BM25: O3 ties = {ties}/30; O4 forward-first = {bm_o4_first}/30",
          flush=True)

    bars = {}
    for s in (0, 1, 2):
        bars[s] = {"O1": per_seed[s]["O1_top1_count"] >= 20,
                   "O2": o2 <= 13, "O3": ties >= 28,
                   "O4": (bm_o4_first >= 28
                          and per_seed[s]["O4_list_top20_count"] <= 10
                          and per_seed[s]["O4_answers"] == 0)}
    all_hold = all(all(v.values()) for v in bars.values())
    summary = {
        "bars_locked": "blind-reading-test-e1b/older-models/BARS_LOCKED.md, "
                       "PR #18 commit 91dda54 (hash posted in F-019 before the run)",
        "note_no_list_gate": "no banked neighbour lists exist for the exp-201 "
                             "models; construction is the gated E1b code path",
        "per_seed": per_seed,
        "control_legs_this_dataset": {"O2_U_top1": o2, "O3_ties": ties,
                                      "O4_BM25_forward_first": bm_o4_first},
        "bars": bars,
        "outcome": ("ALL BARS HOLD on all three seeds — the confirmation's "
                    "sentence loses its 'one dataset' caveat"
                    if all_hold else
                    "NOT ALL BARS HOLD — see per-seed table; report per the "
                    "locked outcome sentences"),
    }
    json.dump(summary, open(f"{OUT}/older_models_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print("OLDER-MODELS RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
