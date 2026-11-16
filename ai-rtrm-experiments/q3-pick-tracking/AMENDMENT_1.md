# Q3 — Amendment 1 (2026-11-16): the scoring target

Dated amendment to `PROTOCOL_LOCKED.md` (locked 2026-11-15, commit
0d46196). The original text is preserved unchanged; this amendment
governs interpretation and the remaining work. Prompted by the
referee's R-002 (PR #17, posted 2026-11-16 14:02 UTC, before the
extraction ran at 22:16 UTC — the experimenter had not read it; that
miss is on the record in F-003).

## Why

The locked protocol scores answer-position states against the
**substitution pick**. But the answer position is the final token of
the *question-format* prompt: the state there produces the
question-format pick, and the two formats agree on only 65.8% of
items. The locked bars (substitution baselines + 10 points) therefore
sit above the ceiling for a readout that tracks the decision that
state actually makes. For the layer-12 lens the point is exact: the
question-format pick is defined as the larger candidate logit at that
position, and the lens at hs[12] through ln_f and lm_head is those
same logits, so the L12 lens lean equals the question pick on every
item by construction. Its score against the substitution pick is the
format-agreement rate, known in advance (verified by the experimenter
from the banked records, reproducing R-002 exactly):

| Stratum | Q pick = sub pick | Locked bar | L12 lens outcome, known in advance |
|---|---|---|---|
| Same-pick pairs, both twins | 119/183 = 65.0% | ≥67.4% | fail |
| Flip pairs, both twins | 4/54 = 7.4% | ≥50% | fail |
| Non-twin items | 96/143 = 67.1% | ≥72.9% | fail |

The locked FAIL reading in RESULTS_Q3.md ("the readout does not
track the model's pick even with the full sentence in view") is
therefore **withdrawn as an interpretation**. The measurements in
RESULTS_Q3.md stand as computed; they are re-read under this
amendment.

A companion fault, found by the run's own diagnostic (diag_lens.py)
and confirmed by R-002's identity argument: in this environment
(transformers 5.17) hs[12] is already post-ln_f, so the locked lens
computation applies ln_f twice at layer 12. A correct L12 lens would
score 100% against the question pick by construction; as computed it
scores 66.4% (same-Q pairs). The L12 lens is henceforth a
**validation line**, not a result: the identity is proven by the
construction in run_confirmatory.py, and the as-computed figure
documents the fault. hs[11] and below are unaffected.

## The amended test

- **Target: the question-format pick.** Strata re-derived by the same
  pairing rule from that pick: 217 same-pick pairs, 20 flip pairs,
  143 non-twins (the pairing itself is unchanged).
- **Primary: the neighbour list at L12, same-pick pairs.** Bars from
  the pronoun-position baselines against the question pick, computed
  from the banked confirmatory records, same baseline + 10 rule:
  - Pronoun baseline, same-Q pairs: 116/217 = 53.5% → **bar 63.5%**
  - Pronoun baseline, non-twins: 73/143 = 51.0% → **bar 61.0%**
  - Flip-Q pairs (20): no bar; reported with an exact binomial
    interval (pronoun baseline 0/20, forced).
- **Disclosure:** the L12 list's answer-position figure against the
  question pick (59.9%) was visible in the committed q3_scores.json —
  where it sat as a no-bar secondary — before this amendment was
  written. The bars above are derived from pronoun-position baselines
  by the locked +10 rule, not fitted to it. All other amended figures
  were unexamined at the time of writing.
- Substitution-target results stay in RESULTS_Q3.md as **reported, no
  bar**, with the ceiling table above beside them.

## Amended results (same extracted states; no re-extraction)

| Readout | Stratum | Answer position | Bar | Outcome |
|---|---|---|---|---|
| List L12 | Same-Q pairs (217) | 130/217 = 59.9% | ≥63.5% | **FAIL** (+6.4 pts over baseline) |
| List L12 | Non-twins (143) | 105/143 = 73.4% | ≥61.0% | **PASS** (+22.4 pts) |
| List L12 | Flip-Q pairs (20) | 5/20 = 25.0% | none | 95% CI ≈ [8.7%, 49.1%] |
| Lens L11 (secondary) | Same-Q pairs | 198/217 = 91.2% | — | pick readable by L11 |
| Lens L11 (secondary) | Non-twins | 137/143 = 95.8% | — | |
| Lens L6–L10 (secondary) | Same-Q pairs | 52.5% / 52.1% / 53.0% / 59.9% / 80.2% | — | readability emerges L9→L11 |

Amended reading: reading at the answer position **does** improve the
list's tracking of the model's pick — decisively on non-twin items,
which clear their bar — but on the twin pairs, the stratum the test
was built for, the improvement (+6.4 points) falls short of the
predeclared +10 bar. The decision itself is fully present in the
state (the lens identity); the neighbour list recovers it only
partially. The paper's Q3 passage is written to this amended account.

## The layer-11 pool — decided now (R-002 §2.5)

**Yes.** The lens reads the pick best at layer 11, and no L11 list
exists because the banked pool holds layers 0/6/12 only. An L11 list
is added as a **secondary** readout under this amendment, staged so
no number is seen before its bar exists:

1. Build the L11 pool from the same WikiText-103 pool sentences
   (same construction as pool_path2.pt, layer 11).
2. Extract pronoun-position L11 states for the 617 items; compute the
   pronoun baselines against the question pick; lock the L11 bars
   (same +10 rule) in a supplement commit.
3. Only then extract answer-position L11 states and score.

The L12 primary above is unaffected by this addition.
