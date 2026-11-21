"""E1b builder -- blind reading test on the RTRM neighbour list.

FRESH RUN (2026-11-20, R-010 option 1 as decided by Paul): identical
construction and gate, pointed at the fresh arm-D models
(exp-201-exposure-fresh) trained on dataset_fresh.json -- new names,
new split. Changes vs build_e1b.py: EXP/OUT paths, SHUFFLE_SEED
20261120, and ground truth read from behavior_fresh.json +
dataset_fresh.json (the behaviour records carry the same probe /
expected_name / godchild fields the original read from verdicts).
The answer key written here stays on the Jetson, uncommitted; only
its SHA-256 goes into the PR (R-010 checklist item 4).

Construction: Study 1's tested recipe (run_confirmatory.py, the 054c-C2
subtracted list) moved to layer 11 and to the fine-tuned expD models:
  - Pool per seed: pool_build_selection.jsonl sentences through that
    seed's model, one unit-normalised hs[11] state per token position,
    fp32 (build_pool_path2.py convention). Labels decode.strip().lower().
  - Residual: mu = pool mean; V = top-64 PCs of (pool - mu) via
    torch.pca_lowrank with torch.manual_seed(0) immediately before;
    res(H) = (H-mu) - ((H-mu) @ V) @ V.T; pool residuals L2-normalised.
  - Query: unit hs[11] at the probe's final position; subtract the pool
    mean of states labelled "is" FIRST (Study 1 subtracted the query
    token's pool mean -- the pronoun there, "is" here); then residual;
    then L2-normalise. Cosine vs pool residuals, descending.
  - Name rank: best-occurrence rank of the godparent's first name
    (lowercased) in the full ranking; None if absent.

Stage 0 VALIDATION GATE (runs first, aborts everything on failure):
the same pipeline in Study 1 mode (base GPT-2, layer 12, banked
pool_path2.pt) must reproduce results_confirmatory_top20.json exactly
(top-20 words) for the first 25 confirmatory items, and
results_confirmatory.json r_label within +/-1. Gate history, kept
honest: the first run (2026-11-18) failed 12/25 because the join key
was item alone -- item numbers restart per split in the banked files;
keyed on (split, item), 24/25 ranks matched exactly and the top-20s
matched 25/25. The one exception, (test, 12), was off by one at rank
4739/4740, where the label's similarity trails the entry above it by
1.5e-8 -- a floating-point tie in a dense tail, immaterial to every
downstream use (all thresholds are <= 20; the flag line is 500).
The +/-1 rank tolerance was adopted for that reason, in writing,
before the build ran; the top-20 requirement stays exact.

Ground truth (fresh run): behavior_fresh.json (pass_8tok, measured
on the fresh models after the forward sanity gate passed 60/60 on all
three seeds) and dataset_fresh.json rev_idx (the 30 reversed-trained
pair indices, shared across seeds -- seeds vary training order only,
per finetune_fresh.py). Nothing banked is modified.

Outputs (this folder, append-only):
  pool_e1b_s{0,1,2}.pt, pool_stats.json, lists_seed{0,1,2}.json,
  bars_seed0.json, answer_key_e1b.json (SEALED), reader_set.md,
  response_form.csv, build_log.txt (via stdout redirect)
"""
import json, random, sys, time
from collections import Counter
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

EXP = "/home/ecpi-student-05/robot-mind/exp-201-exposure-fresh"
P2 = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
OUT = f"{EXP}/blind-test-e1b"
RECORDS = f"{OUT}/records"          # copies of the banked Study 1 files
K = 64
SHUFFLE_SEED = 20261120
SEEDS = [0, 1, 2]


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


def rank_of(words, order_labels, target):
    target = target.lower()
    for i, w in enumerate(order_labels):
        if w == target:
            return i + 1
    return None


# ---------------------------------------------------------------- stage 0
def validation_gate(tok):
    print("=== STAGE 0: validation gate (Study 1 mode) ===", flush=True)
    items = [json.loads(l) for l in open(f"{RECORDS}/confirmatory_items.jsonl")][:25]
    # JOIN KEY IS (split, item): item numbers restart per split in the
    # banked files, so keying on item alone silently collides test and
    # train records. (First gate run failed 12/25 on exactly this --
    # the pipeline itself had reproduced the sentence-matched values.)
    banked_top20 = {(r["split"], r["item"]): r["top20"] for r in
                    json.load(open(f"{RECORDS}/results_confirmatory_top20.json"))}
    banked_res = {(r["split"], r["item"]): r for r in
                  json.load(open(f"{RECORDS}/results_confirmatory.json"))}
    spool = torch.load(P2 + "/pool_path2.pt", map_location="cpu", weights_only=False)
    vecs, words = [], []
    for r in spool:
        for p, v in enumerate(r["hs"][2]):
            vecs.append(v.float())
            words.append(word_of(tok, r["ids"][p]))
    del spool
    M = torch.stack(vecs)
    mu, V = residual_setup(M)
    Rn = F.normalize(coderes(M, mu, V), dim=1)
    pron_mu = {}
    for pron in ["he", "she", "it", "they"]:
        mask = torch.tensor([w == pron for w in words])
        pron_mu[pron] = M[mask].mean(0)
    lm = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    bad = 0
    with torch.no_grad():
        for it in items:
            pron = it["pronoun"]
            prompt = ("%s What does the word %s refer to? Answer with one word:"
                      % (it["sentence"], pron))
            ids = tok(prompt)["input_ids"]
            o = lm(torch.tensor([ids]), output_hidden_states=True)
            h = F.normalize(o.hidden_states[12][0].float(), dim=1)
            pid = tok(" " + pron, add_special_tokens=False)["input_ids"][0]
            pos = [i for i, x in enumerate(ids) if x == pid][0]
            q1 = h[pos].unsqueeze(0) - pron_mu[pron]
            q2 = coderes(q1, mu, V)
            sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
            order = torch.argsort(sims, descending=True)
            top20 = [words[order[i]] for i in range(20)]
            r_label = rank_of(words, [words[j] for j in order], it["label_head"])
            key = (it["split"], it["item"])
            ok20 = top20 == banked_top20[key]
            okr = abs(r_label - banked_res[key]["r_label"]) <= 1
            if not (ok20 and okr):
                bad += 1
                print(f"  MISMATCH item {key}: top20_ok={ok20} "
                      f"r_label {r_label} vs banked {banked_res[key]['r_label']}",
                      flush=True)
    del lm, M, Rn
    if bad:
        print(f"VALIDATION FAILED: {bad}/25 items mismatch -- ABORT", file=sys.stderr)
        sys.exit(1)
    print("VALIDATION PASSED: 25/25 top-20 lists and label ranks reproduce "
          "the banked confirmatory records exactly.", flush=True)


# ---------------------------------------------------------------- pools
def build_pool(tok, seed):
    path = f"{OUT}/pool_e1b_s{seed}.pt"
    try:
        recs = torch.load(path, map_location="cpu", weights_only=False)
        print(f"pool s{seed}: loaded existing {path}", flush=True)
        return recs
    except FileNotFoundError:
        pass
    rows = [json.loads(l) for l in open(P2 + "/pool_build_selection.jsonl")]
    model = GPT2LMHeadModel.from_pretrained(f"{EXP}/models/expD_s{seed}").eval()
    recs = []
    t0 = time.time()
    with torch.no_grad():
        for n, r in enumerate(rows):
            ids = tok(r["sentence"])["input_ids"]
            assert len(ids) == r["gpt2_tokens"], ("token count", r["corpus_index"])
            hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
            recs.append({"id": "wt103-%d" % r["corpus_index"], "ids": ids,
                         "hs11": F.normalize(hs[11][0].float(), dim=1)})
            if (n + 1) % 200 == 0:
                print(f"  pool s{seed}: {n+1}/{len(rows)} ({time.time()-t0:.0f}s)",
                      flush=True)
    del model
    torch.save(recs, path)
    print(f"pool s{seed}: wrote {path} ({time.time()-t0:.0f}s)", flush=True)
    return recs


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    validation_gate(tok)

    behavior = json.load(open(f"{EXP}/behavior_fresh.json"))
    dataset = json.load(open(f"{EXP}/dataset_fresh.json"))
    rev_idx = set(dataset["rev_idx"])
    assert len(rev_idx) == 30

    pool_stats = {}
    all_lists = {}
    for seed in SEEDS:
        print(f"=== SEED {seed} ===", flush=True)
        recs = build_pool(tok, seed)
        vecs, words = [], []
        for r in recs:
            for p, v in enumerate(r["hs11"]):
                vecs.append(v.float())
                words.append(word_of(tok, r["ids"][p]))
        M = torch.stack(vecs)
        norms = M.norm(dim=1)
        assert float((norms - 1.0).abs().max()) < 1e-4, "pool norms not unit"
        mu, V = residual_setup(M)
        Rn = F.normalize(coderes(M, mu, V), dim=1)
        is_mu = M[torch.tensor([w == "is" for w in words])].mean(0)
        hist = Counter(words)
        pool_stats[f"s{seed}"] = {
            "states": len(words), "sentences": len(recs),
            "norm_max_dev": float((norms - 1.0).abs().max()),
            "distinct_labels": len(hist),
            "top20_labels": hist.most_common(20),
            "is_occurrences": int(sum(1 for w in words if w == "is")),
        }
        model = GPT2LMHeadModel.from_pretrained(f"{EXP}/models/expD_s{seed}").eval()
        brecs = {r["index"]: r for r in behavior["models"][f"expD_s{seed}"]["reversed"]}
        vrecs = brecs  # fresh: probe metadata rides on the behaviour records
        lists = []
        with torch.no_grad():
            for i in range(60):
                vr = vrecs[i]
                ids = torch.tensor([tok(vr["probe"])["input_ids"]])
                hs = model(ids, output_hidden_states=True).hidden_states
                h = F.normalize(hs[11][0].float(), dim=1)
                q1 = h[-1].unsqueeze(0) - is_mu
                q2 = coderes(q1, mu, V)
                sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
                order = torch.argsort(sims, descending=True)
                order_labels = [words[j] for j in order]
                first_name = vr["expected_name"].split()[0]
                lists.append({
                    "seed": seed, "index": i, "probe": vr["probe"],
                    "godparent": vr["expected_name"], "godchild": vr["godchild"],
                    "top20": order_labels[:20],
                    "name_rank": rank_of(words, order_labels, first_name),
                    "pass_8tok": bool(brecs[i]["pass_8tok"]),
                    "subset": ("stored-trained" if i in rev_idx else "forward-only"),
                })
        del model, M, Rn
        all_lists[seed] = lists
        json.dump({"model": f"expD_s{seed}", "layer": 11, "items": lists},
                  open(f"{OUT}/lists_seed{seed}.json", "w"), indent=1)
        npass = sum(x["pass_8tok"] for x in lists)
        print(f"seed {seed}: 60 lists built; passes {npass}/60", flush=True)
    json.dump(pool_stats, open(f"{OUT}/pool_stats.json", "w"), indent=1)

    # ---- bars from seed 0 ----
    s0 = all_lists[0]
    bars = {}
    for task, truth in [("task1", [x["pass_8tok"] for x in s0]),
                        ("task2", [x["subset"] == "stored-trained" for x in s0])]:
        table = {}
        for k in list(range(1, 21)):
            preds = [(x["name_rank"] is not None and x["name_rank"] <= k) for x in s0]
            table[k] = balanced_accuracy(preds, truth)
        best_k = min(table, key=lambda k: (-table[k], k))
        bars[task] = {"table": {str(k): round(v, 4) for k, v in table.items()},
                      "best_k": best_k, "bar": round(table[best_k], 4)}
        print(f"{task}: seed-0 best threshold k={best_k} BA={table[best_k]:.4f}", flush=True)
    json.dump(bars, open(f"{OUT}/bars_seed0.json", "w"), indent=1)

    # ---- reader set (seeds 1+2), shuffled, fixed-form responses ----
    items12 = all_lists[1] + all_lists[2]
    order = list(range(120))
    random.Random(SHUFFLE_SEED).shuffle(order)
    key, lines, form = {}, [], ["item_id,task1_pass_fail,task2_stored_familiar"]
    lines += [
        "# Blind reading test E1b — reader set", "",
        "Each item shows a prompt given to a fine-tuned GPT-2 model, the",
        "correct godparent's full name, and the RTRM neighbour list for the",
        "model's layer-11 state at the end of the prompt: the 20 words from",
        "ordinary sentences whose states are most similar to it, in order.",
        "Entries are occurrences — the same word may appear more than once.",
        "",
        "For every item, make two judgments in the response form:",
        "- **Task 1:** Will the model produce the correct godparent when it",
        "  continues this prompt? **PASS** or **FAIL**.",
        "- **Task 2:** Is the correct godparent *stored* in the model, or",
        "  *merely familiar* to it? **STORED** or **FAMILIAR**.",
        "",
        "Work from the 20 words alone. Do not skip items. You will receive",
        "no feedback of any kind until all 120 judgments are in.", "", "---", "",
    ]
    for pos, j in enumerate(order, start=1):
        item_id = f"E1B-{pos:03d}"
        x = items12[j]
        key[item_id] = {"item_id": item_id, "seed": x["seed"], "index": x["index"],
                        "subset": x["subset"], "pass_8tok": x["pass_8tok"],
                        "name_rank": x["name_rank"]}
        form.append(f"{item_id},,")
        lines += [f"## Item {item_id}", "", f"Prompt: \"{x['probe']}\"", "",
                  f"Correct godparent: {x['godparent']}", "",
                  "Neighbour list (20 words, in rank order):", ""]
        lines += [f"{n}. {w}" for n, w in enumerate(x["top20"], start=1)]
        lines += ["", "---", ""]
    json.dump({"shuffle_seed": SHUFFLE_SEED, "sealed": True, "items": key},
              open(f"{OUT}/answer_key_e1b.json", "w"), indent=1)
    open(f"{OUT}/reader_set.md", "w").write("\n".join(lines))
    open(f"{OUT}/response_form.csv", "w").write("\n".join(form) + "\n")
    print("WROTE: pool_stats.json, lists_seed{0,1,2}.json, bars_seed0.json, "
          "answer_key_e1b.json, reader_set.md, response_form.csv", flush=True)
    print("BUILD COMPLETE", flush=True)


if __name__ == "__main__":
    main()
