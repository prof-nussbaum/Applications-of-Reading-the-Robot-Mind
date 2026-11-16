# AI RTRM experiments — referee exchange, context and current state
Written 2026-11-15 by Francis (the experimenter's agent) so that a referee
joining cold has the whole picture in-repo. Paul (the author) relays
between the referee and Francis; this folder is the shared channel.

## The programme in one paragraph
"Reading the Robot Mind" (RTRM) reads a model's hidden states with a
cosine-similarity neighbour list — words, not statistics — so a domain
expert can inspect what the model has stored. Two predeclared tests of
hidden-state diagnostics were run on GPT-2 small and written up as a
short paper (held, unpublished): Study 1, an RTRM checklist meant to warn
of pronoun-resolution errors before the output margin does; Study 2, a
published logit-lens verdict (STORED / WEAK / MISSING) meant to detect
whether a fine-tuned model stores a fact in reversed form. Both tests
came out negative or inconclusive. The referee's reviews have since
driven three follow-ups, all in this repo: a positive control, true
order-reversed probes, and a token-exposure retrain (running).

## What is settled (referee-verified)
- Study 1 confirmatory (617 DPR items, bars locked in advance): the
  checklist caught 76.1% of margin-missed errors but fired on 72.8% of
  correct items; false-alarm cost +65.9 pts vs a ≤20 bar; Fisher
  p = 0.296. Verdict NOT WIN. The flag is at chance at every pool size
  (balanced accuracy 0.503 / 0.511 / 0.515 / 0.518 at 5k / 15k / 30k /
  full); leg decomposition shows the two active legs each fire at
  near-coin-flip rates on errors and correct items alike. The test bed
  itself was at chance (GPT-2 accuracy 51.9%, Wilson CI 47.9–55.8%).
  Folder: post-book-experiments/pronoun-resolution/path2-early-warning/
  (protocols and reports posted earlier; the item-level records and
  the run/analysis/validation code are added in this PR, so the
  paper's data-availability statement holds. The 439 MB candidate
  pool itself is not posted — GitHub's file-size limit — but its
  build code and build report are in the folder.) Mechanism checks:
  ai-rtrm-experiments/path2-mechanism-checks/.
- Positive control (ai-rtrm-experiments/path2-positive-control/): 120
  agreement-settled items. Locked 80% gate FAILED — question-format
  accuracy 60.0%; verdict CONTROL FAILED AS A CONTROL, bars descriptive
  only. By substitution scoring the same items reach 80.0% (number items
  100%): the question format, not the pronouns, is the weak link. Lean
  (label outranks competitor in the list) 61.7%; flag balanced accuracy
  0.434; margin AUC 0.612. Ceiling probe on the DPR states (supervised,
  NOT RTRM, diagnostic only): test AUC 0.316, CI [0.262, 0.377].
- Class-lift check on the control: number items carry a grammatical-
  number class lift (out-of-sentence same-class median rank 989 vs
  opposite-class 1,395) and the label sits above its class (median
  188.5; beats the class median in 50/60), with sentence-presence a
  confound on that gap; gender items show no class lift and ranks in
  the thousands throughout. The list's top 20 is pronouns: referents
  are findable by rank lookup, not "visible" — the paper's wording is
  being corrected accordingly.
- Reversed probes (ai-rtrm-experiments/exp-201-reversed/): true
  order-reversed probes ("{godchild}'s godparent is") on the exp-201
  models: behaviour 0/60 in all three arms (forward was 60/55/29) — a
  clean reversal curse. BUT the lens verdicts are confounded by token
  exposure (verified from the training corpora): godparent names occur
  mid-sentence 0 times in the no-bridge arms and exactly once per name
  in the bridge arm, so MISSING was forced in two arms and the bridge
  arm's WEAK band is set-wide familiarity (specificity check: the
  correct name beats only 51–55% of the other 59 godparent names in
  every arm; median ranks in specificity2_summary.json).
- Substitution rescore (ai-rtrm-experiments/path2-substitution-rescore/,
  exploratory, not predeclared): the 617 items scored by substitution
  reach 59.5% (question format 51.9%, reproduced exactly); 129 of the
  297 question-format errors (43.4%) are correct by substitution. The
  locked checklist rescored against substitution-defined errors:
  catch 84.0%, false alarms 68.1%, balanced accuracy 0.579 (vs 0.518
  against question-format errors); the margin baseline's AUC stays
  0.508. The NOT WIN verdict stands — it was defined against the
  question format — but the null's reading changes from "no
  information" to "too weak to use". Lens on the control's number
  items: hs[11] lean 86.7%, hs[12] 56.7%, cosine list 65.0%.

## Settled since the kickoff
- Token-exposure retrain (ai-rtrm-experiments/token-exposure-control/,
  RESULTS in folder): **FAMILIARITY CONFIRMED** on the locked bars.
  Forward gate passed (A 60/60/60, B 56/57/57, C 31/30/16 of 60).
  Specificity (correct name best of 60 at layer 11): arms A/B/C at
  0.00–0.07 in all nine arm-seeds (pooled 0.033, chance 0.017); arm D
  reversed-trained pairs 0.50/0.53/0.40 across seeds with the correct
  name beating 94% of the other godparents; arm D forward-only pairs
  0.00 in all three seeds. Two sub-predictions missed and recorded in
  the results file: MISSING share pooled 64% (predicted <50%), and
  arm D reversed behaviour 9–11 of 30 (predicted ≥24 of 30) — the
  lens singles out the trained name, the model still does not say it.

- Q3 pick-tracking test (ai-rtrm-experiments/q3-pick-tracking/,
  RESULTS in folder): **FAIL on all three locked bars.** Reading at
  the answer position does not repair pick-tracking: list at L12 —
  same-pick pairs 99/183 = 54.1% (bar ≥67.4%; pronoun baseline
  57.4%), flip pairs 8/54 = 14.8% (bar ≥50%), non-twins 94/143 =
  65.7% (bar ≥72.9%). L12 lens closest at 63.4% same-pick, still a
  miss; no lens layer met any bar. Per the locked reading, the
  readout does not track the model's pick even with the full
  sentence in view — the revised headline stands strengthened. One
  caveat documented in RESULTS_Q3.md: the L12 lens computation
  double-applies the final LayerNorm in this environment (the banked
  hs[12] control figure carries the same caveat; hs[11] is
  unaffected). The Path 2 NOT WIN verdict is untouched.

## Running now
- Nothing is training or extracting. Next: the paper's final
  revision (workspace document) folds in the retrain, Q3, and the
  R-001 corrections; the blind reading test on arm D awaits Paul's
  timing.
- (The substitution rescore and the lens comparison on the control's
  number items, exploratory when this file was first written, have
  landed — see the settled section above.)

## The framing under review
Referee's synthesis, adopted by the experimenter: Study 2 becomes the
paper's centrepiece ("a published lens verdict confounds familiarity
with storage, and here is the control that exposes it"), Study 1 shrinks
to three plain statements — the question format is an unreliable read of
GPT-2; the neighbour list recovers sentence content and grammatical
number; it carries only a weak item-specific referent signal. The
experimenter's addition: across the robot and language work alike, the
readouts recover what a model HAS (stored content) and not what it will
DO (its decision) — availability, not utilisation. The paper has been
rewritten on this framing (v4, 2026-11-15, held by the author, not
yet in this repo); a final revision folds in the exposure retrain's
results when they land.

## House rules for this exchange
- Referee comments and proposes; Francis runs, verifies, and answers
  in-repo; Paul merges every pull request himself. Nothing here is
  merged without Paul's word.
- A first specificity run (specificity_mix_*.json on the Jetson only)
  was discarded before use: it omitted the lens's final layer-norm and
  its ranks contradicted the saved verdicts. The validated rerun
  (specificity2_*) reproduced the saved verdict ranks exactly and is
  the record.
