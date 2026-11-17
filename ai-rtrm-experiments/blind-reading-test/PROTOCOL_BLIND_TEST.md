# Blind reading test — protocol (LOCKED at build time)

Programme item E1. Design settled in referee message R-003 section 5
and R-005 section 3-E1. Built 2026-11-17
from the banked token-exposure retrain records; no banked file modified.

## Model and items

- Model: **expD_s0 only**. Seed rule: seed 0 is the first seed — chosen
  by position, not by outcome.
- Items: the 60 reversed probes of arm D seed 0. Each probe prompt is
  "{godchild}'s godparent is", shown verbatim.
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

The 60 items were shuffled with `random.Random(20261117)` and
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
  In this set: 9 pass, 51 fail.
- Score: **balanced accuracy** (mean of sensitivity and specificity).
  Any fixed rule scores 0.500.
- Baseline, LOCKED: the **lens rank-1 rule** (predict PASS iff the
  correct name sits at hs[11] lens rank 1) scores balanced accuracy
  **0.6471** on these 60 items.
- **Bar: the reader passes Task 1 iff his balanced accuracy EXCEEDS
  0.6471.**

## Task 2 — stored vs merely familiar

For each item the reader labels the correct godparent STORED or
MERELY FAMILIAR.

- Ground truth: the training subset. "Stored-trained" = the 30 pairs
  whose reversed fact was trained in arm D; "forward-only" = the other
  30, whose names are merely familiar. In this set: 30 stored,
  30 familiar.
- Score: **balanced accuracy**. Any fixed rule scores 0.500.
- Comparator, LOCKED: the **lens band rule** (STORED band -> stored,
  anything else -> familiar) scores balanced accuracy **0.8000**
  on these 60 items.
- **Bar: the reader passes Task 2 iff his balanced accuracy EXCEEDS
  0.8000.**

## Files

- reader_set.md — what the reader works from.
- answer_key.json — SEALED ground truth (subset, behaviour, lens rank,
  band per item). Not to be opened before the reader finishes.
- top20_extraction.json — the full lens extraction per pair index.
- build_blind_test.py — the builder (validation gate included).
