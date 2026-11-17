# Arm D within-subset specificity (R-003 §1c) — 2026-11-17

**Question (referee R-003 §1c).** The published specificity ranks the
correct godparent name against all 59 others. In arm D the 30
reverse-trained names are the only names that ever follow "is" in
training, so they are lifted as a set: ranking a reverse-trained name
against all 59 mixes in the 30 forward-only names it was never in fair
competition with (and depresses the forward-only figure the same way).
R-003 asks for the correct name ranked only against the other 29 names
of its own subset, per seed.

**Method.** Records only — no model runs. `verdicts_exposure.json`
carries, per reversed item, the hs[11] lens ranks of all 60 godparent
first-name tokens (`ranks60_hs11`, pair-index order;
`score_verdicts.py` lines 62–80). `score_within_subset.py` recomputes
everything from those ranks. Subsets: the 30 reverse-trained pair
indices from `random.Random(201).shuffle(range(60))` (first 30,
sorted: 1, 2, 5, 6, 7, 10, 11, 15, 16, 19, 21, 22, 23, 24, 25, 27, 29,
31, 32, 33, 37, 38, 41, 42, 52, 53, 54, 56, 58, 59); the other 30 are
forward-only.

**Validation (before any within-subset number).** Per item, the stored
spec fields (`spec_best_of_60`, `spec_frac_others_beaten`,
`spec_correct_rank_among_60`) recompute exactly from `ranks60_hs11`,
and the recomputed all-59 aggregates reproduce RESULTS_EXPOSURE.md per
seed: rev-trained best-of-60 15/30, 16/30, 12/30 with mean beaten 0.94;
forward-only best-of-60 0/30 with mean beaten 0.35. PASSED.

## PREDICTED vs ACTUAL

PREDICTED (Fable, R-003 §1c): the rev-trained within-subset beaten
share "should still be near 88%", and the specificity conclusion
should hold.

ACTUAL: rev-trained mean beaten within subset = **0.889 pooled**
(0.889 / 0.891 / 0.887 per seed) — the prediction lands. The
conclusion holds.

## Results

Reverse-trained subset (30 items per seed):

| seed | best within subset | mean beaten within subset | (all-59: best / mean beaten) |
|---|---|---|---|
| expD_s0 | 15/30 = 0.50 | 0.889 | 15/30 / 0.940 |
| expD_s1 | 16/30 = 0.53 | 0.891 | 16/30 / 0.941 |
| expD_s2 | 12/30 = 0.40 | 0.887 | 12/30 / 0.937 |
| pooled | 43/90 = 0.478 | **0.889** | 43/90 / 0.939 |

Forward-only subset (30 items per seed):

| seed | best within subset | mean beaten within subset | (all-59: best / mean beaten) |
|---|---|---|---|
| expD_s0 | 1/30 = 0.033 | 0.523 | 0/30 / 0.350 |
| expD_s1 | 1/30 = 0.033 | 0.525 | 0/30 / 0.349 |
| expD_s2 | 1/30 = 0.033 | 0.538 | 0/30 / 0.351 |
| pooled | 3/90 = 0.033 | **0.529** | 0/90 / 0.350 |

## Reading

- The set-lift inflation in the published figure is real but small:
  0.94 against all 59 becomes 0.889 within the subset. Item-specific
  storage in the reverse-trained subset survives the correction, as
  Fable expected.
- Best-within-subset counts for the reverse-trained subset are
  identical to the best-of-60 counts in every seed (15/16/12): whenever
  a reverse-trained name was beaten at all, it was beaten by a name
  from its own subset.
- The forward-only figure moves the other way and lands on chance:
  within its own subset the correct name beats 0.529 of the others
  (≈ the 0.50 coin flip) and is best in exactly 1/30 per seed — the
  chance rate for best-of-30. The published 0.35 was cross-subset lift
  from the reverse-trained names occupying the top ranks, not any
  within-subset signal. Forward-only names carry no item-specific
  storage, by this measure either.

Files: `score_within_subset.py`, `within_subset_specificity.json`
(per-item detail included). Computed 2026-11-17 from the banked
records; nothing banked was modified.
