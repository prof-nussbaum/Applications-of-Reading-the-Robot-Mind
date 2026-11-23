# Older-models run — bars locked before building (R-017 §3)

Locked by Francis, 2026-11-22, before any build on these models. The lock
commit's hash is posted in a PR #18 comment (F-019) before the run starts,
per R-017 §5.

## Question

The seeds 1–2 confirmation (C1–C4, all holding) shares one dataset: the
three E1b seeds differ only in the fine-tuning run. Do the same bars hold
on a **different dataset** — the exp-201 token-exposure arm-D models
(banked 2026-11-16): a different 60 names, a different split, three seeds,
already trained? No new training. Construction identical to the E1b
variant-T build (same code path; the pool is each model's own states over
the WikiText sentences plus that dataset's fine-tuning sentences;
neighbours shown as source sentences). Behaviour is read from the banked
exp-201 records, not re-probed.

Unlike the E1b confirmation, the control legs are **not** already in
hand: this is a different dataset, so Control U and BM25 run fresh on the
exp-201 texts.

## Bars — the C1/C4 bars, applied per seed, all three seeds

- **O1 (= C1).** Fine-tuned variant T: the pair's own reversed sentence is
  the top neighbour in at least **20 of 30** reversed-trained pairs.
- **O2 (= C2).** Control U (untrained base GPT-2, same construction): the
  same count is at most **13 of 30**.
- **O3 (= C3).** BM25 (Okapi, k1 = 1.5, b = 0.75, word tokens lowercased)
  gives the pair's forward and reversed sentences **equal scores** in at
  least **28 of 30** reversed-trained pairs (scored on scores, not ranks).
- **O4 (= C4).** Forward-only pairs: BM25 ranks the pair's forward
  sentence first in at least **28 of 30**; the fine-tuned list has that
  sentence in the top 20 in at most **10 of 30**; the model answers
  **0 of 30** (banked behaviour).

Reported with the bars, as before: the pass-versus-fail split within
reversed-trained pairs, and the same rule-family tables as the seed-0
variant-T records.

## Outcome sentences, fixed now

- **All bars hold on all three seeds** → the confirmation's sentence
  loses its "one dataset" caveat: the retrieval replicates across
  training runs *and* across name sets and splits.
- **O1 fails on any seed** → the retrieval does not transfer across
  datasets as built; report which seed and by how much.
- **O1 holds but a control leg fails** → report exactly which leg and
  what it does to the reading, at the same prominence.
