# Path 2 POSITIVE CONTROL — PROTOCOL (LOCKED 2026-11-15)

Status: LOCKED 2026-11-15, after the dataset audit PASSED (AUDIT_2026-11-15.md, seed 0910, Python random.Random(910); 120 items: 60 number-agreement, 60 gender-agreement; zero eligibility failures; known asymmetry recorded in the audit: pronoun split number it 37/they 23, gender he 36/she 24). Written under Paul's "You pick. You go." (2026-11-15).
Runs after the exp-201 reversed-probe test (completed 2026-11-15).
No amendments at lock time. Any later change needs a dated amendment file, declared before scoring.


## Why

Path 2's confirmatory null was measured on a chance-level test bed:
GPT-2 accuracy on the 617 DPR items was 51.9% (Wilson 95% CI
47.9–55.8%), margin AUC 0.520, and the neighbour list's lean matched the
model's pick on margin-missed errors at 51.9%. That null cannot by
itself separate "the instrument carries no signal" from "the test bed
carried no signal". A positive control — items the model demonstrably
can do — is the missing leg.

## Items

120 new pronoun items where the referent is settled by grammar:
- Number agreement (60): two candidates differing in number; pronoun
  they/it. Template family, e.g. "The dogs chased the cat because they
  were fast." / "The dogs chased the cat because it was fast."
- Gender agreement (60): candidates are a man-named and a woman-named
  person (or unambiguous role nouns: king/queen, actor/actress);
  pronoun he/she.
Construction: hand-picked good examples + time-of-day-seeded random
draw over the template families (seed = local HHMM, PRNG recorded in
the audit file). Formal sample-data audit per the standing rule:
exact prompt text, label defensibility, expected behaviour, scoring,
vacuous-by-construction risks — dated file, no scoring before it passes.
Pool: reuse pool_path2.pt (WikiText-103, 47,499 states) unchanged;
eligibility = the Path 2 rule (>=10 pool occurrences per candidate).

## Prompt wording (both formats scored on every item)

1. Question format (identical to Path 2): "<sentence> What does the
   word <pronoun> refer to? Answer with one word:"
2. Substitution format (exp-052): the sentence with each candidate
   substituted for the pronoun, scored by model probability of the
   substituted sentence.

## Gate (locked)

GPT-2 forced-choice accuracy (question format) on the 120 items must be
>= 80%. Below the gate, the control has failed as a control: report
that, and no checklist conclusion is drawn either way.

## Scoring and bars (locked; identical in form to Path 2)

Method of record unchanged: 054c-C2 subtracted list at L12, exp-056
flag, margin baseline B1 at <0.35 nats.
- (i) RTRM catch of MISS >= 30%
- (ii) combined sensitivity gain >= +10 pts
- (iii) false-alarm cost <= +20 pts
- (iv) Fisher (flag x error, margin-unflagged) one-sided p < 0.05
- INCONCLUSIVE if MISS < 30 (expected if the model is good: at 80-90%
  accuracy, MISS will be small — if MISS < 30 the control is reported
  as a gate-pass with an underpowered checklist test, NOT stretched).
Verdict taxonomy defined over ALL bars this time: WIN = all four pass;
REFUTED = gate passed, MISS >= 30, and (iii) or (iv) fails; NOT WIN
otherwise, with the failing bars named. (This closes the Path 2
drafting gap.)

## Ceiling check (diagnostic only — NOT RTRM, labelled as such)

Logistic-regression probe on the L12 pronoun state, trained to predict
the model's error, split-half by item (train 308 / test 309 of the DPR
confirmatory records, states recomputed by the confirmatory recipe):
report test AUC with a bootstrap 95% CI. Reading: if a supervised probe
with direct access to the state cannot beat chance at predicting
errors, no checklist on that state could have — Study 1's null is then
a statement about the state. If the probe succeeds where the checklist
failed, the information exists and the checklist is the wrong reader.
Either reading bounds the claim; the probe itself is never presented
as an RTRM result or a book claim.

## PREDICTED (locked 2026-11-15, before scoring)

- Gate passes: GPT-2 forced-choice accuracy >= 80% on the agreement
  items (question format).
- MISS will be < 30 (at >=80% accuracy there are at most 24 errors), so
  the four-bar checklist test will be UNDERPOWERED by design and must be
  reported as such — not stretched.
- Secondary readout, declared now: the neighbour list's lean matches the
  model's pick. On DPR margin-missed errors this was 51.9%. Prediction:
  on control items (all items, and on errors separately) the lean-match
  rate is >= 70% IF the list tracks the model's decisions where the
  model demonstrably knows the answer; <= 60% means it does not track
  even there.
- Ceiling probe on the DPR confirmatory states: test AUC between 0.45
  and 0.60 — near chance. A probe AUC well above 0.60 would refute this
  and mean the error information IS in the state and the checklist is
  simply the wrong reader.

## Data gathered

Per-item records identical in shape to results_confirmatory.json, both
prompt formats; the audit file; SME pages for 6 items (3 errors, 3
correct): the full verbatim prompt, the model's pick, and the
subtracted neighbour list's top 20 words in rank order — the list a
human reader would actually see.
