> **CORRECTION, 2026-11-16 (referee R-001, accepted in F-001):** the
> Fisher exact p = 4.6e-6 reported in Part 2 below is WITHDRAWN. Its
> table treats the 617 items as independent, but 237 twin pairs (474
> items) share one pronoun state with opposite labels, so the items
> are not independent. The honest pair-level figure is p ≈ 0.02. The
> counts themselves (catch 84.0%, false alarms 68.1%, balanced
> accuracy 0.579) are unchanged and reproduce exactly; only the
> significance claim is withdrawn. The paper's final revision carries
> this correction.

# Substitution rescore of the 617 + lens on the control's number items

Exploratory follow-up (not predeclared), run 2026-11-15 at the
referee's suggestion. Two parts, one job:

1. All 617 Path 2 confirmatory DPR items rescored by substitution
   (the sentence rewritten with "the" + each candidate in the
   pronoun's place; the two full sentences compared by total
   log-probability). Output: `subst_rescore_617.json`.
2. A logit-lens comparison at the pronoun on the positive control's
   60 number items (lens ranks of label vs competitor at hs[11] and
   hs[12]). Output: `lens_number_items.json`.
3. `checklist_vs_substitution.py` joins part 1's output with the
   confirmatory records (`results_confirmatory.json` in
   pronoun-resolution/path2-early-warning/) and rescores the locked
   checklist against substitution-defined errors.

Provenance: `subst_rescore.py` is the original script, written to run
on CPU so as not to disturb the token-exposure retrain on the GPU; it
never produced output (the process died at an unknown time, and the
loss went unnoticed for hours — the failure is recorded in the
experimenter's notes). `subst_rescore_gpu.py` is the identical
scoring moved to the GPU on the author's instruction; it completed in
about a minute with the retrain running concurrently and unharmed.
All numbers below were recomputed independently from the output
JSONs and match the run log exactly.

## Part 1 — substitution vs question format, 617 items

- Question-format accuracy: 51.9% (320/617) — reproduces the
  confirmatory number exactly.
- Substitution accuracy: 59.5% (367/617).
- Both right 238; question-only 82; substitution-only 129;
  neither 168. The formats agree on 65.8% (406/617).
- Of the 297 question-format errors, 129 (43.4%) are correct by
  substitution.

## Part 3 — the locked checklist against substitution-defined errors

(Part 2's numbers follow; the join is listed second because it is the
consequential one.)

- Substitution-defined errors: 250; substitution-correct: 367.
- Checklist flag: catches 210/250 errors (84.0%); of the 225
  substitution errors the margin baseline missed, it catches 192
  (85.3%). (Sanity: the same computation against question-format
  errors reproduces the locked headline exactly — 268 margin-missed,
  204 caught, 76.1%.)
- False alarms: 250/367 substitution-correct items flagged (68.1%).
- Balanced accuracy 0.579 (question-format target: 0.518). Fisher
  exact p = 4.6e-6 (flag vs substitution-defined error, all 617
  items; a different table from the confirmatory bar (iv)).
- Margin baseline AUC against substitution-defined correctness:
  0.508 — still chance.

Reading: the ranks carry a weak signal about the model's errors,
buried under format noise in the confirmatory target. The locked
NOT WIN verdict stands (it was defined against the question format);
the interpretation of the null changes from "no information" to "too
weak to use" — the flag still fires on two-thirds of correct items
under either definition of error.

## Part 2 — lens at the pronoun, 60 number control items

- hs[11]: label outranks competitor in 86.7% (52/60).
- hs[12]: 56.7% (34/60).
- Cosine neighbour list (same items): 65.0%.

These are items the model answers correctly by substitution 100% of
the time, so both readouts are measuring stored content
(availability), not decisions. The layer is not a detail: hs[11] and
hs[12] bracket the list from opposite sides.

---

## FLAG (2026-11-17, referee R-003 §3): the hs[12] lens figure above

The "Lens on the control's number items" figures include hs[12]
lean 56.7%. In this environment (transformers 5.17.0),
`hidden_states[12]` is already post-`ln_f`, and the lens code
applies `ln_f` a second time at layer 12 only — so every hs[12]
lens figure in these records is computed through a double
normalisation and should not be read as a layer-12 measurement.
The hs[11] figures (86.7% lean on the number items) are unaffected,
as is every list figure. The referee's R-001 argument from the
hs[11]/hs[12] gap ("one layer changes the answer by 30 points")
is retracted in R-003: the gap was the fault. Valid lens layer
evidence is the Q3 answer-position series: 0.65 at L9, 0.84 at
L10, 0.93 at L11 (balanced accuracy against the question pick).
