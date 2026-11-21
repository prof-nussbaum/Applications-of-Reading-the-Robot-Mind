"""Step 1 controls for the variant-T retrieval result (R-013 s.3), seed 0.

Criterion (fixed in F-014 BEFORE running): the retrieval is specific
to the fine-tuned model iff, under Control U, the pair's own reversed
sentence is the top neighbour in at most 13 of the 30 stored-trained
pairs (fine-tuned: 26 of 30).

Control U: variant T construction exactly as build_variants_seed0.py,
but base GPT-2 small ("gpt2") in place of the fine-tuned expD_s0 --
for the pool (Wikipedia selection + the 150 distinct fine-tuning
sentences) and for the probe queries. Basis, "is" mean, residual
recipe unchanged.

Control B: Okapi BM25 (k1 = 1.5, b = 0.75, idf = ln(1 + (N - df +
0.5) / (df + 0.5))) over the same sentences as variant T's pool --
the 1,756 pool_build_selection sentences plus the 150 distinct
fine-tuning sentences -- word tokens lowercased, split on
non-alphanumerics. Query: the probe text. Reported per item: rank of
the pair's reversed sentence (stored pairs) and of its forward
sentence, by subset.

Outputs (new folder variants-seed0/controls/): controlU_seed0.json,
controlB_seed0.json, controls_summary.json. Nothing banked modified.
"""
import json, math, os, re, sys
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

EXP = "/home/ecpi-student-05/robot-mind/exp-201-exposure-fresh"
P2 = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
OUT = f"{EXP}/variants-seed0/controls"
K = 64
CRITERION_TOP1_MAX = 13


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


def ba_table(rank_list, truth, kmax=20):
    table = {}
    for k in range(1, kmax + 1):
        preds = [(r is not None and r <= k) for r in rank_list]
        table[k] = round(balanced_accuracy(preds, truth), 4)
    best_k = min(table, key=lambda k: (-table[k], k))
    return {"table": {str(k): v for k, v in table.items()},
            "best_k": best_k, "best_ba": table[best_k]}


def residual_setup(M):
    mu = M.mean(0)
    torch.manual_seed(0)
    U, S, V = torch.pca_lowrank(M - mu, q=K)
    return mu, V


def coderes(H, mu, V):
    Hc = H - mu
    return Hc - (Hc @ V) @ V.T


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    behavior = json.load(open(f"{EXP}/behavior_fresh.json"))
    dataset = json.load(open(f"{EXP}/dataset_fresh.json"))
    pairs = dataset["pairs"]
    rev_idx = set(dataset["rev_idx"])
    brecs = {r["index"]: r for r in behavior["models"]["expD_s0"]["reversed"]}
    truth1 = [bool(brecs[i]["pass_8tok"]) for i in range(60)]
    truth2 = [(i in rev_idx) for i in range(60)]

    # FT sentences with pair/kind attribution (as in build_variants_seed0)
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
    wiki_texts = [r["sentence"] for r in rows]

    # ---------------- Control U --------------------------------------
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    vecs, words, srcs = [], [], []

    def add_sentence(text, kind, pair):
        ids = tok(text)["input_ids"]
        with torch.no_grad():
            hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
        H = F.normalize(hs[11][0].float(), dim=1)
        for p in range(len(ids)):
            vecs.append(H[p])
            words.append(word_of(tok, ids[p]))
            srcs.append({"kind": kind, "pair": pair, "text": text, "pos": p, "ids": ids})

    for n, text in enumerate(wiki_texts):
        add_sentence(text, "wiki", None)
        if (n + 1) % 400 == 0:
            print(f"  control U pool: {n+1}/{len(wiki_texts)} wiki sentences", flush=True)
    for text, kind, i in ft:
        add_sentence(text, kind, i)
    M = torch.stack(vecs)
    assert float((M.norm(dim=1) - 1.0).abs().max()) < 1e-4, "pool norms not unit"
    mu, V = residual_setup(M)
    Rn = F.normalize(coderes(M, mu, V), dim=1)
    is_mu = M[torch.tensor([w == "is" for w in words])].mean(0)
    print(f"control U pool: {len(words)} states (base GPT-2)", flush=True)

    def first_rank(order, pred):
        for r, j in enumerate(order):
            if pred(srcs[int(j)]):
                return r + 1
        return None

    itemsU = []
    with torch.no_grad():
        for i in range(60):
            ids = torch.tensor([tok(brecs[i]["probe"])["input_ids"]])
            hs = model(ids, output_hidden_states=True).hidden_states
            h = F.normalize(hs[11][0].float(), dim=1)
            q1 = h[-1].unsqueeze(0) - is_mu
            q2 = coderes(q1, mu, V)
            sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
            order = torch.argsort(sims, descending=True).tolist()
            gp_first = pairs[i][0].split()[0]
            gc_first = pairs[i][1].split()[0]
            itemsU.append({
                "index": i, "probe": brecs[i]["probe"], "godparent": pairs[i][0],
                "godchild": pairs[i][1],
                "subset": "stored-trained" if i in rev_idx else "forward-only",
                "pass_8tok": truth1[i],
                "top1_kind": srcs[order[0]]["kind"], "top1_pair": srcs[order[0]]["pair"],
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
                "top20_ft_share": sum(1 for j in order[:20]
                                      if srcs[j]["kind"] != "wiki") / 20.0,
            })
    del model, M, Rn
    famU = {
        "U-src-name": {"task1": ba_table([x["name_in_source_rank"] for x in itemsU], truth1),
                       "task2": ba_table([x["name_in_source_rank"] for x in itemsU], truth2)},
        "U-pair-rev": {"task1": ba_table([x["pair_reversed_rank"] for x in itemsU], truth1),
                       "task2": ba_table([x["pair_reversed_rank"] for x in itemsU], truth2)},
        "U-pair-any": {"task1": ba_table([x["pair_any_ft_rank"] for x in itemsU], truth1),
                       "task2": ba_table([x["pair_any_ft_rank"] for x in itemsU], truth2)},
    }
    json.dump({"control": "U (untrained base GPT-2, variant T construction)",
               "seed": 0, "items": itemsU, "bar_tables": famU},
              open(f"{OUT}/controlU_seed0.json", "w"), indent=1)
    storedU = [x for x in itemsU if x["subset"] == "stored-trained"]
    u_top1 = sum(1 for x in storedU if x["pair_reversed_rank"] == 1)
    u_top9 = sum(1 for x in storedU if x["pair_reversed_rank"] is not None
                 and x["pair_reversed_rank"] <= 9)
    print(f"control U: stored pairs with own reversed sentence at rank 1: "
          f"{u_top1}/30 (criterion <= {CRITERION_TOP1_MAX}); within top 9: {u_top9}/30",
          flush=True)

    # ---------------- Control B (BM25) --------------------------------
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

    def bm25_rank(query):
        q = tokens(query)
        scores = []
        for d in range(N):
            dl = len(doc_tokens[d])
            s = 0.0
            for w in q:
                f = doc_tf[d].get(w, 0)
                if f:
                    s += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
            scores.append(s)
        return sorted(range(N), key=lambda d: -scores[d])

    ft_doc_index = {}
    for d, (t, kind, i) in enumerate(docs):
        if kind in ("forward", "reversed"):
            ft_doc_index[(kind, i)] = d
    itemsB = []
    for i in range(60):
        order = bm25_rank(brecs[i]["probe"])
        pos = {d: r + 1 for r, d in enumerate(order)}
        itemsB.append({
            "index": i, "probe": brecs[i]["probe"], "godparent": pairs[i][0],
            "godchild": pairs[i][1],
            "subset": "stored-trained" if i in rev_idx else "forward-only",
            "pass_8tok": truth1[i],
            "top1_text": docs[order[0]][0], "top1_kind": docs[order[0]][1],
            "reversed_sentence_rank": (pos[ft_doc_index[("reversed", i)]]
                                       if i in rev_idx else None),
            "forward_sentence_rank": pos[ft_doc_index[("forward", i)]],
        })
    json.dump({"control": "B (BM25 over the pool sentences)", "seed": 0,
               "items": itemsB}, open(f"{OUT}/controlB_seed0.json", "w"), indent=1)
    storedB = [x for x in itemsB if x["subset"] == "stored-trained"]
    fwdB = [x for x in itemsB if x["subset"] == "forward-only"]
    b_rev_top1 = sum(1 for x in storedB if x["reversed_sentence_rank"] == 1)
    b_rev_top9 = sum(1 for x in storedB if x["reversed_sentence_rank"] <= 9)
    b_fwd_top20_fwdonly = sum(1 for x in fwdB if x["forward_sentence_rank"] <= 20)
    print(f"control B: stored pairs' reversed sentence at rank 1: {b_rev_top1}/30; "
          f"within top 9: {b_rev_top9}/30; forward-only pairs' forward sentence "
          f"in top 20: {b_fwd_top20_fwdonly}/30", flush=True)

    summary = {
        "criterion_fixed_in_F014": (
            "retrieval is specific to the fine-tuned model iff under Control U "
            "the pair's own reversed sentence is the top neighbour in at most "
            f"{CRITERION_TOP1_MAX} of 30 stored-trained pairs (fine-tuned: 26/30)"),
        "controlU_top1_count": u_top1,
        "controlU_top9_count": u_top9,
        "criterion_verdict": ("SPECIFIC to the fine-tuned model"
                              if u_top1 <= CRITERION_TOP1_MAX else
                              "NOT shown specific — untrained model retrieves "
                              "nearly as well"),
        "controlB": {"reversed_rank1": b_rev_top1, "reversed_top9": b_rev_top9,
                     "forward_only_forward_in_top20": b_fwd_top20_fwdonly},
        "finetuned_reference": {"top1": 26, "top9": 30},
    }
    json.dump(summary, open(f"{OUT}/controls_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print("CONTROLS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
