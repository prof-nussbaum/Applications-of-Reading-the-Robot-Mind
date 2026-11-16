> **CORRECTION, 2026-11-16 (referee R-001, accepted in F-001):** the
> "Reading (locked)" paragraph on the ceiling probe below is
> WITHDRAWN. The probe's test AUC of 0.316 is not evidence of an
> overfit probe or "a statement about the state": the by-item split
> separated twin pairs — items with identical states and opposite
> labels — across train and test, so the below-chance AUC is twin
> leakage, a split artefact. The AUC as trained (0.316) stands as a
> number; the interpretation is replaced by the leakage account. The
> control's gate verdict (question-format accuracy 60.0% vs the
> locked 80% gate — CONTROL FAILED AS A CONTROL) is unaffected.

# Path 2 — POSITIVE CONTROL RESULTS (2026-11-15)

Protocol: `PROTOCOL_LOCKED.md` (locked 2026-11-15 after the sample-data audit PASSED, seed 0910). Items: 120 new agreement items (60 number, 60 gender) from `items_control.jsonl`. Pool reused unchanged: pool_path2.pt (47,499 states). Question format identical to the Path 2 confirmatory run; substitution format = the 047a method (first pronoun occurrence replaced by "the <candidate>"; total log-probability of the full substituted sentence), as implemented in exp-048/run_048.py.

## GATE (reported first, per the locked protocol)

- Question-format (forced-choice) accuracy: **60.0%** (72 of 120 correct). Gate bar: >= 80%. **FAIL**
- By family (descriptive): number 61.7% (37/60), gender 58.3% (35/60).
- Substitution-format accuracy (descriptive, not gated): **80.0%** (96 of 120); number 100.0%, gender 60.0%. Question/substitution picks agree on 70.0% of items.

## VERDICT: **CONTROL FAILED AS A CONTROL — question-format accuracy 60.0% < 80% gate; per the locked protocol no checklist conclusion is drawn either way.**

## PREDICTED vs ACTUAL (locked predictions, PROTOCOL_LOCKED.md)

PREDICTED:
- Gate passes: question-format accuracy >= 80%.
- MISS < 30, so the four-bar checklist test is UNDERPOWERED by design and must be reported as such.
- Lean-match (secondary): >= 70% if the list tracks the model's decisions where the model demonstrably knows; <= 60% means it does not track even there.
- Ceiling probe on DPR confirmatory states: test AUC between 0.45 and 0.60.

ACTUAL:
- Gate: FAIL — accuracy 60.0%.
- Errors 48 of 120; MISS = 42 (errors the margin baseline did not flag).
- Lean-match: all items 46.7% (56/120); on errors 31.2% (15/48); on correct items (descriptive) 56.9%.
- Ceiling probe (DIAGNOSTIC ONLY — NOT RTRM): test AUC for predicting the model's error = 0.316, bootstrap 95% CI [0.262, 0.377]; train AUC 0.914.

## The four locked bars (pooled, question format)

The gate failed, so per the locked protocol the bars license NO verdict and are not claimed as PASS/FAIL. Descriptive quantities only: RTRM catch of MISS 50.0% (21/42); sensitivity B1 12.5% | RTRM 47.9% | combined 56.2% (gain +43.8 pp); false-alarm B1 8.3% | RTRM 61.1% | combined 62.5% (gain +54.2 pp); Fisher (iv) table (B1-unflagged) [[21, 39], [21, 27]], one-sided p = 0.87.

Secondary exp-056 H1 (flag x error, all items): table [[23, 44], [25, 28]], one-sided Fisher p = 0.947. Appendix: margin AUC as error score 0.612; RTRM flag balanced accuracy 0.434.

Note on the lean-match readout: the locked lean prediction was conditional on the control being a place the model demonstrably knows the answer. The gate failed for the question format (accuracy 60.0%), so that condition is not met and the lean rates above (all items 46.7%; errors 31.2%) are reported descriptively, without the locked either/or reading. For the record they fall in the <= 60% band the protocol associated with 'does not track'. The substitution format, which is not the gated format, scored 80.0%.

## Ceiling probe — labelled reading (DIAGNOSTIC ONLY, NOT RTRM)

A supervised logistic-regression probe (a trained classifier — a different method from the RTRM cosine-similarity method) was trained on the raw L12 pronoun states of the 617 DPR confirmatory items to predict the model's correctness (split-half by item, 308 train / 309 test, seed 0; rank-reproduction check: 617/617 items reproduced exactly). Test AUC for predicting the model's ERROR: **0.316** (bootstrap 95% CI [0.262, 0.377]); test AUC for predicting correctness: 0.316; train AUC (error): 0.914.
Reading (locked): the locked prediction was a test AUC between 0.45 and 0.60; the actual AUC of 0.316 falls BELOW that band, with its bootstrap CI [0.262, 0.377] entirely under chance, while the train AUC is 0.914. That is the signature of an overfit probe (768 features, 308 training items): the direction it learns does not generalise to held-out items. Under the locked reading, a supervised probe with direct access to the state cannot beat chance at predicting errors on held-out items, so no checklist on that state could have — Study 1's null is a statement about the state, not only about the checklist. We report the AUC as trained; flipping its sign after the fact would be a retrofit the protocol forbids. (The probe is a ceiling diagnostic only; it is not an RTRM result and supports no book claim.)

## Deviations and declared choices

- Substitution convention: the protocol names the format 'exp-052 style'; exp-052's own script scored the question format. The battery's substitution method of record is the 047a method (exp-048/run_048.py, exp-049, exp-050b): first pronoun occurrence replaced by "the <candidate>", scored by TOTAL log-probability of the full substituted sentence. That method was used, unmodified, and is declared here and in run_control.py.
- Control records carry clue_class = null (the agreement items have no clue-position class) and a family field (number/gender); all other fields match results_confirmatory.json in shape.
- Ceiling-probe state: the raw per-token L2-normalised L12 state at the pronoun position (the state the checklist recipe starts from), as specified in the task brief.

Files: `results_control.json` (120 per-item records, both formats), `results_control_top20.json` (subtracted-list top-20 per item), `analysis_control.json` (every number above), `ceiling_probe_results.json` + `probe_states_dpr.pt` (Part 2), `SME_CONTROL_ITEMS.md` (Part 3). Nothing here is posted or published.
