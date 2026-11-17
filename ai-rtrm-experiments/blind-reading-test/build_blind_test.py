"""Blind reading test set builder (programme item E1; design settled in
referee message R-003 section 5 / R-005 section 3-E1).

Model: expD_s0 ONLY. Seed rule: seed 0 is the first seed -- chosen by
position, not by outcome.

Lens: identical computation to score_verdicts.py / rtrm_probe.probe_layers
-- hs[11] at the final prompt position, manual ln_f (mean/variance
normalisation with the model's ln_f weight and bias), projection through
lm_head. Run on CPU in float32, exactly as the original scoring ran.

VALIDATION GATE (runs before any set is built): the recomputed
lens_rank_hs11 of the correct godparent's first token must match
verdicts_exposure.json (expD_s0, reversed probes) on all 60 items
exactly. Any mismatch aborts the build.

Outputs (this folder, append-only; no banked file is modified):
  top20_extraction.json  -- per pair index: lens rank, band, top-20 tokens
  answer_key.json        -- SEALED ground truth per shuffled item ID
  reader_set.md          -- the 60 shuffled items, reader-facing only
  PROTOCOL_BLIND_TEST.md -- design, locked bars, scoring rules
"""
import json, random, sys, torch
from datetime import datetime, timezone
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP = "/home/ecpi-student-05/robot-mind/exp-201-exposure"
OUT = f"{EXP}/blind-test"
MODEL_KEY = "expD_s0"
SHUFFLE_SEED = 20261117
REVERSED_TRAINED = {1, 2, 5, 6, 7, 10, 11, 15, 16, 19, 21, 22, 23, 24, 25,
                    27, 29, 31, 32, 33, 37, 38, 41, 42, 52, 53, 54, 56, 58, 59}


def band(rank):
    if rank <= 10:
        return "STORED"
    if rank <= 100:
        return "WEAK"
    return "MISSING"


def balanced_accuracy(preds, truths):
    """preds/truths: parallel lists of booleans. Mean of sensitivity and
    specificity. Any constant rule scores exactly 0.500."""
    tp = sum(1 for p, t in zip(preds, truths) if p and t)
    fn = sum(1 for p, t in zip(preds, truths) if not p and t)
    tn = sum(1 for p, t in zip(preds, truths) if not p and not t)
    fp = sum(1 for p, t in zip(preds, truths) if p and not t)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return (sens + spec) / 2.0


def main():
    tok = AutoTokenizer.from_pretrained("gpt2")

    def tids(s):
        return tok(s, add_special_tokens=False)["input_ids"]

    verdicts = json.load(open(f"{EXP}/verdicts_exposure.json"))
    behavior = json.load(open(f"{EXP}/behavior_exposure.json"))
    vrecs = {r["index"]: r for r in verdicts["models"][MODEL_KEY]["reversed"]}
    brecs = {r["index"]: r for r in behavior["models"][MODEL_KEY]["reversed"]}
    assert sorted(vrecs) == list(range(60)) and sorted(brecs) == list(range(60))

    model = AutoModelForCausalLM.from_pretrained(f"{EXP}/models/{MODEL_KEY}").eval()
    lw = model.transformer.ln_f.weight.detach().float()
    lb = model.transformer.ln_f.bias.detach().float()
    W = model.lm_head.weight.detach().float()

    extraction = []
    mismatches = []
    for i in range(60):
        vr = vrecs[i]
        probe = vr["probe"]                      # "{godchild}'s godparent is"
        answer = " " + vr["expected_name"]       # " {godparent}"
        tgt = tids(answer)[0]
        ids = torch.tensor([tids(probe)])
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        h = hs[11][0].float()
        mu = h.mean(-1, keepdim=True)
        var = ((h - mu) ** 2).mean(-1, keepdim=True)
        hn = (h - mu) / torch.sqrt(var + 1e-5) * lw + lb
        v = (hn @ W.T)[-1]
        rank = int((v > v[tgt]).sum()) + 1
        if rank != vr["lens_rank_hs11"] or band(rank) != vr["verdict_hs11"]:
            mismatches.append((i, rank, vr["lens_rank_hs11"]))
        top = torch.topk(v, 20)
        top20 = [{"token_id": int(t), "text": tok.decode([int(t)])}
                 for t in top.indices]
        extraction.append({
            "index": i,
            "probe": probe,
            "godparent": vr["expected_name"],
            "godchild": vr["godchild"],
            "lens_rank_hs11": rank,
            "verdict_hs11": band(rank),
            "top20_hs11": top20,
        })
        print(f"[{i:2d}] rank {rank:6d} {band(rank):7s} "
              f"banked {vr['lens_rank_hs11']:6d} {vr['verdict_hs11']:7s}",
              flush=True)

    if mismatches:
        print(f"VALIDATION FAILED: {len(mismatches)} mismatches: {mismatches}",
              file=sys.stderr)
        sys.exit(1)
    print("VALIDATION PASSED: 60/60 lens_rank_hs11 (and bands) reproduce "
          "verdicts_exposure.json expD_s0 exactly.")

    json.dump({"model": MODEL_KEY, "layer": 11,
               "built_utc": datetime.now(timezone.utc).isoformat(),
               "validation": "60/60 lens_rank_hs11 match verdicts_exposure.json",
               "items": extraction},
              open(f"{OUT}/top20_extraction.json", "w"), indent=1)

    # ---- Ground truth + locked baselines (computed from the key) ----
    rows = []
    for ex in extraction:
        i = ex["index"]
        rows.append({
            "index": i,
            "subset": ("stored-trained" if i in REVERSED_TRAINED
                       else "forward-only"),
            "pass_8tok": bool(brecs[i]["pass_8tok"]),
            "lens_rank_hs11": ex["lens_rank_hs11"],
            "verdict_hs11": ex["verdict_hs11"],
        })
    # Task 1 baseline: lens rank-1 rule (predict pass iff rank == 1)
    ba_rank1 = balanced_accuracy([r["lens_rank_hs11"] == 1 for r in rows],
                                 [r["pass_8tok"] for r in rows])
    # Task 2 comparator: lens band rule (STORED -> stored, else familiar)
    ba_band = balanced_accuracy([r["verdict_hs11"] == "STORED" for r in rows],
                                [r["subset"] == "stored-trained" for r in rows])
    n_pass = sum(r["pass_8tok"] for r in rows)
    n_stored = sum(r["subset"] == "stored-trained" for r in rows)
    print(f"Task 1 rank-1 rule balanced accuracy: {ba_rank1:.4f} "
          f"(passes {n_pass}/60)")
    print(f"Task 2 band rule balanced accuracy:   {ba_band:.4f} "
          f"(stored-trained {n_stored}/60)")

    # ---- Shuffle and assign reader item IDs ----
    order = list(range(60))
    random.Random(SHUFFLE_SEED).shuffle(order)
    by_index = {ex["index"]: ex for ex in extraction}
    row_by_index = {r["index"]: r for r in rows}
    key = {}
    reader_items = []
    for pos, idx in enumerate(order, start=1):
        item_id = f"BT-{pos:02d}"
        ex = by_index[idx]
        key[item_id] = {"item_id": item_id, **row_by_index[idx]}
        reader_items.append((item_id, ex))
    json.dump({"model": MODEL_KEY, "shuffle_seed": SHUFFLE_SEED,
               "sealed": True, "items": key},
              open(f"{OUT}/answer_key.json", "w"), indent=1)

    # ---- Reader set (no subset, no behaviour, no ranks, no bands) ----
    lines = [
        "# Blind reading test — reader set",
        "",
        "Each item below shows a prompt that was given to a fine-tuned GPT-2",
        "model, the correct godparent's full name, and the 20 words the",
        "model's layer 11 ranks highest at the end of that prompt, in rank",
        "order. Nothing else about the item is shown.",
        "",
        "For every item, make two judgments:",
        "",
        "- **Task 1:** Will the model produce the correct godparent when it",
        "  continues this prompt? Circle **PASS** or **FAIL**.",
        "- **Task 2:** Is the correct godparent *stored* in the model, or",
        "  *merely familiar* to it? Circle **STORED** or **MERELY FAMILIAR**.",
        "",
        "Work from the 20 words alone. Do not skip items.",
        "",
        "---",
        "",
    ]
    for item_id, ex in reader_items:
        words = [t["text"].strip() for t in ex["top20_hs11"]]
        lines.append(f"## Item {item_id}")
        lines.append("")
        lines.append(f"Prompt: \"{ex['probe']}\"")
        lines.append("")
        lines.append(f"Correct godparent: {ex['godparent']}")
        lines.append("")
        lines.append("Top 20 words, in rank order:")
        lines.append("")
        for n, w in enumerate(words, start=1):
            lines.append(f"{n}. {w}")
        lines.append("")
        lines.append("Task 1 — the model will produce the correct godparent: "
                     "**PASS / FAIL**")
        lines.append("")
        lines.append("Task 2 — the correct godparent is: "
                     "**STORED / MERELY FAMILIAR**")
        lines.append("")
        lines.append("---")
        lines.append("")
    open(f"{OUT}/reader_set.md", "w").write("\n".join(lines))

    # ---- Protocol with locked bars ----
    proto = f"""# Blind reading test — protocol (LOCKED at build time)

Programme item E1. Design settled in referee message R-003 section 5
and R-005 section 3-E1. Built {datetime.now(timezone.utc).date().isoformat()}
from the banked token-exposure retrain records; no banked file modified.

## Model and items

- Model: **expD_s0 only**. Seed rule: seed 0 is the first seed — chosen
  by position, not by outcome.
- Items: the 60 reversed probes of arm D seed 0. Each probe prompt is
  "{{godchild}}'s godparent is", shown verbatim.
- For each probe, the model's hidden state at layer 11 (hs[11], final
  prompt position) is passed through the lens — the model's final layer
  norm (ln_f) and unembedding — exactly as computed by score_verdicts.py.
  The 20 highest-logit vocabulary tokens, decoded as words, in rank
  order, are the reader's only evidence.

## Validation (before the set was built)

The builder's lens pipeline reproduced the committed
verdicts_exposure.json expD_s0 lens_rank_hs11 values for the correct
names on **60/60** probes exactly (bands likewise). The build aborts on
any mismatch.

## Shuffle

The 60 items were shuffled with `random.Random({SHUFFLE_SEED})` and
given item IDs BT-01..BT-60 in shuffled order. The ground truth is
sealed in answer_key.json, separate from the reader set.

## What the reader sees

Item ID, the probe prompt verbatim, the correct godparent's full name,
and the top-20 words in rank order. No subset, no behaviour, no ranks,
no bands.

## Task 1 — behaviour prediction

For each item the reader predicts whether the model will produce the
correct godparent (PASS) or not (FAIL).

- Ground truth: pass_8tok from behavior_exposure.json (expD_s0).
  In this set: {n_pass} pass, {60 - n_pass} fail.
- Score: **balanced accuracy** (mean of sensitivity and specificity).
  Any fixed rule scores 0.500.
- Baseline, LOCKED: the **lens rank-1 rule** (predict PASS iff the
  correct name sits at hs[11] lens rank 1) scores balanced accuracy
  **{ba_rank1:.4f}** on these 60 items.
- **Bar: the reader passes Task 1 iff his balanced accuracy EXCEEDS
  {ba_rank1:.4f}.**

## Task 2 — stored vs merely familiar

For each item the reader labels the correct godparent STORED or
MERELY FAMILIAR.

- Ground truth: the training subset. "Stored-trained" = the 30 pairs
  whose reversed fact was trained in arm D; "forward-only" = the other
  30, whose names are merely familiar. In this set: {n_stored} stored,
  {60 - n_stored} familiar.
- Score: **balanced accuracy**. Any fixed rule scores 0.500.
- Comparator, LOCKED: the **lens band rule** (STORED band -> stored,
  anything else -> familiar) scores balanced accuracy **{ba_band:.4f}**
  on these 60 items.
- **Bar: the reader passes Task 2 iff his balanced accuracy EXCEEDS
  {ba_band:.4f}.**

## Files

- reader_set.md — what the reader works from.
- answer_key.json — SEALED ground truth (subset, behaviour, lens rank,
  band per item). Not to be opened before the reader finishes.
- top20_extraction.json — the full lens extraction per pair index.
- build_blind_test.py — the builder (validation gate included).
"""
    open(f"{OUT}/PROTOCOL_BLIND_TEST.md", "w").write(proto)
    print("WROTE: top20_extraction.json, answer_key.json, reader_set.md, "
          "PROTOCOL_BLIND_TEST.md")


if __name__ == "__main__":
    main()
