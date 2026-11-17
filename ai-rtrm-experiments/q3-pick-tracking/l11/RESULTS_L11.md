# Q3 Amendment 1 — Stage 2: the L11 list at the answer position

Run 2026-11-16, Jetson, per `AMENDMENT_1.md` §"The layer-11 pool"
and `AMENDMENT_1_SUPPLEMENT.md` (the L11 bars, locked before any
answer-position L11 state was extracted). Extraction:
`extract_answer_l11.py` — hs[11] at the last prompt token of the
exact confirmatory prompt, neighbour list against `pool_L11.pt`
with the Q3 answer-position recipe (no pronoun-mean subtraction;
PCA-64, seed 0; pool mean/PCA refit from the L11 pool; best-occurrence
ranks; lean = lower rank). Scoring: `score_answer_l11.py` over the
shared machinery in `stage2_common.py`. Records:
`answer_l11_ranks.json` (617 per-item records), `scores_answer_l11.json`.

## Primary verdict: PASS

**Primary metric — neighbour list at L11, answer position, same-Q
twin pairs, lean vs the model's question-format pick on BOTH twins:
141/217 = 65.0%. Locked bar ≥59.8% (pronoun baseline 108/217 =
49.8%, +10 rule).** The L11 list at the answer position clears its
primary bar by 5.2 points — an improvement of +15.2 points over its
own pronoun baseline. The secondary bar clears too (non-twins,
below). This is the comparison the amendment promised, and it lands
differently from L12: the L12 list's same-Q improvement (+6.4 points)
fell short of its +10 bar; the L11 list's does not.

## PREDICTED (locked bars and baselines) vs ACTUAL

### Primary readout: neighbour list, L11, answer position (vs question-format pick)

| Stratum | Baseline (L11, pronoun) | Locked bar | ACTUAL (answer position) | Bar met |
|---|---|---|---|---|
| Same-Q pairs (217) | 108/217 = 49.8% | ≥59.8% | **141/217 = 65.0%** (both 141, exactly-one 17, neither 59) | **YES** |
| Non-twin items (143) | 66/143 = 46.2% | ≥56.2% | **100/143 = 69.9%** | **YES** |
| Flip-Q pairs (20) | 0/20 (forced) | none | **4/20 = 20.0%**, exact 95% CI [5.7%, 43.7%] (exactly-one 14) | — |

Note the structural change from the pronoun position: at the pronoun,
same-pick pairs are both-or-neither by construction (one shared
state, one lean). At the answer position the twins' states differ,
and 17 same-Q pairs now split exactly-one — the readout is genuinely
reading two different states.

### Against the substitution pick (reported, no bar)

| Stratum | L11 pronoun baseline | ACTUAL (answer position) |
|---|---|---|
| Same-pick pairs (183) | 88/183 = 48.1% | 102/183 = 55.7% (exactly-one 16, neither 65) |
| Flip pairs (54) | 0/54 (forced) | 7/54 = 13.0% (exactly-one 47) |
| Non-twin items (143) | 65/143 = 45.5% | 87/143 = 60.8% |

### The promised comparison: L11 vs L12 at the answer position (vs question-format pick)

| Readout | Same-Q pairs | Non-twins | Flip-Q pairs |
|---|---|---|---|
| **List L11** | **141/217 = 65.0%** vs bar ≥59.8% — **PASS** (+15.2 over its 49.8% baseline) | **100/143 = 69.9%** vs bar ≥56.2% — **PASS** (+23.7 over its 46.2% baseline) | 4/20 = 20.0%, CI [5.7%, 43.7%] |
| List L12 (Amendment 1) | 130/217 = 59.9% vs bar ≥63.5% — FAIL (+6.4 over its 53.5% baseline) | 105/143 = 73.4% vs bar ≥61.0% — PASS (+22.4 over its 51.0% baseline) | 5/20 = 25.0%, CI [8.7%, 49.1%] |

Reading, plainly: at the answer position the layer-11 list tracks
the model's question-format pick better than the layer-12 list does
on the twin pairs — absolutely (65.0% vs 59.9%) and against its own
bar — while the layer-12 list remains higher on non-twin items
(73.4% vs 69.9%; both clear their bars). Two honest qualifications.
First, the bars are layer-relative by the locked +10 rule: L11's
bars sit lower in absolute terms because its pronoun baselines were
lower, exactly as the supplement declared before extraction. Second,
flip-Q pairs stay weak at both layers (4/20 and 5/20, intervals wide
and overlapping): when the model's pick actually flips between the
twins, neither list follows it reliably. The lens result from
Amendment 1 is untouched by this: the pick is fully readable from
the state by layer 11 (lens L11, 91.2% same-Q); the neighbour list
recovers it only partially — but at L11, for the first time in this
test series, partially is enough to clear a predeclared bar on the
primary stratum.

## Validation (all gates passed before scoring)

- **Strata re-derived** from the frozen files (`validate_stage2_l11.py`,
  driver step 1): 237 pairs + 143 non-twins covering the 617 items
  exactly once; amended strata vs the question pick 217 same-Q / 20
  flip-Q; vs the substitution pick 183 / 54 (pairing unchanged).
- **Stage-1 baselines reproduced with the stage-2 scoring code**
  (`stage2_common.pair_stats` on `pronoun_l11_ranks.json`): 108/217,
  0/20 (20 exactly-one), 66/143 vs the question pick; 88/183, 0/54
  (54 exactly-one), 65/143 vs the substitution pick. The scoring
  script repeated every one of these checks, plus a disagreement
  check of the extracted records' picks against the banked joins
  (0 disagreements), before scoring the answer leans.
- **Pool refit fidelity:** the extraction's refit of the L11 pool
  structures reproduced stage 1's top-64 PCA variance (0.5026)
  exactly as asserted in the script.
- **Interval machinery:** the exact (Clopper–Pearson) interval code
  was self-checked against Amendment 1's published L12 flip-Q
  interval — 5/20 → [8.7%, 49.1%] — before use.

## Run record

Driver `run_stage2.sh` under nohup, monitored by `monitor_stage2.sh`
(`monitor_status_stage2.log`): step 1 validation OK 17:53:28 EST,
step 2 extraction OK 17:56:21, step 3 scoring OK 17:56:22,
`stage2.done` written; monitor logged DONE 17:56:42. One launch
fault, on the record: the monitor's first launch failed silently
because the launch command's relative path resolved in the home
directory (the driver's own internal `cd` made the driver immune);
the monitor was relaunched by absolute path at 17:54:42, during
extraction, and heartbeated normally thereafter. The driver and its
logs were unaffected.

## Files (this stage; all in `~/robot-mind/q3-pick-tracking/l11/`)

- `stage2_common.py` — shared strata/pick/pair-stats machinery
- `validate_stage2_l11.py` — pre-extraction validation gate
- `extract_answer_l11.py` — answer-position L11 extraction/readout
- `answer_l11_ranks.json` — per-item answer-position L11 ranks + lean (617)
- `score_answer_l11.py` — gated scoring vs the locked bars
- `scores_answer_l11.json` — all stage-2 numbers, machine-readable
- `run_stage2.sh`, `monitor_stage2.sh` — driver and monitor
- `stage2.log`, `stage2_driver.log`, `monitor_status_stage2.log` — run record


---

## Addendum (2026-11-17, from referee message R-003 §2) — EXPLORATORY

The locked verdicts above stand exactly as recorded. The referee's
recomputation (R-003, script `referee/R-003_checks.py`, verified by
Francis against these records) adds two things the locked bars could
not show.

**Table 2a — a fixed rule clears the bars.** The model's
question-format pick is the second-mentioned noun on 61.9% of items.
Scoring "lean = second noun" on the amended strata:

| Readout | Same-Q pairs, both twins | Non-twins | Flip-Q pairs |
|---|---|---|---|
| List L11, answer position | 141/217 = 65.0% (bar 59.8%) | 100/143 = 69.9% (bar 56.2%) | 4/20 |
| List L12, answer position | 130/217 = 59.9% (bar 63.5%) | 105/143 = 73.4% (bar 61.0%) | 5/20 |
| **Always the second noun** | **137/217 = 63.1%** | **88/143 = 61.5%** | 0/20 |

Baseline-plus-10 was the wrong kind of bar when the baseline sits at
chance and the target is unbalanced. The referee endorsed that rule
in R-002; the miss is shared.

**Table 2b — with the bias removed, the tracking is there.** Balanced
per-item accuracy against the question-format pick (mean of the hit
rates on second-noun picks and first-noun picks; any fixed rule
scores 0.500):

| Readout | Balanced accuracy |
|---|---|
| List L12, pronoun | 0.512 |
| List L11, pronoun | 0.503 |
| List L12, answer position | 0.665 |
| List L11, answer position | 0.668 |
| Lens L9, answer position | 0.647 |
| Lens L10, answer position | 0.840 |
| Lens L11, answer position | 0.928 |

Conclusions the addendum supports: the neighbour list reads the
model's decision at the answer position at about 0.67 against 0.50,
up from chance at the pronoun. **There is no layer dependence for
the list** (0.668 vs 0.665); the L11 PASS and L12 FAIL above differ
because their pronoun baselines differ (49.8% and 53.5%), not
because the readouts do. In same-Q pairs where the model picked the
*first* noun — where the fixed rule scores zero — the L11 list
matches both twins in 45 of 80 pairs and the L12 list in 42 of 80.
The lens reads the same decision far better (0.93 at L11); the pick
becomes readable between L9 and L10.

**Standing audit rule adopted (R-003 §2d):** every lean-versus-pick
figure is printed beside the best fixed rule for the same stratum.
