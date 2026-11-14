# Path 2 — Early warning beyond output confidence: PREDECLARED PROTOCOL

Written 2026-11-13, before any new data is downloaded, audited, or run.
Decision: Path 2 chosen by Francis under Paul's delegation ("You choose,
you go!", 2026-11-13). Path 1 (blind auditing game) is deferred until this
path produces a signal worth handing to people. Nothing here is launched
yet: the sample-data audit gates the pilot, and the pilot gates the run.
No experiment number is assigned — numbers follow completed work.

## The question

On new pronoun-resolution items, does the RTRM subtracted-list checklist
flag model errors that the simplest output baseline — the final answer
margin — does NOT flag? If it only re-flags what a small margin already
shows, RTRM adds nothing here and we say so.

## What we build on (the record, not the hope)

- exp-055 (2026-09-29): 10 new prompts. All 4 model errors were flagged
  by the checklist; 3 false alarms on correct items. Sensitive, not
  specific, n=10.
- exp-056 (2026-10-06): 24 prompts. H1 (errors concentrate in flagged
  items) borderline pass, p=0.050, sensitivity 75%, specificity 67%.
  CORRECTION_056: within-pair residual invariance was vacuous by
  construction — the disambiguating word sat after the pronoun, so the
  causal pronoun state could not contain it. Any scaled design must
  stratify by clue position instead of claiming encoding where encoding
  is impossible.
- exp-057 (2026-10-06): measuring at the sentence-final token (which can
  see the clue) still failed its bar, 8/16 vs >=12/16. The honest summary
  of the arc: a warning signal that is real at pilot scale, unproven at
  any larger scale, with one known vacuity trap and one known positional
  failure mode.
- The reversal-detector lesson (2026-10-11): the final-output readout
  separated trained from untrained MORE cleanly than the internal lens
  (softmax trained max rank 1 / untrained min 8; lens 9 / 13). Internal
  signals must therefore beat the output baseline, not merely agree
  with it. That is the whole design of this protocol.

## Hypothesis and PREDICTED

HYPOTHESIS: the mechanical RTRM flag (below) catches a substantial share
of the errors the margin baseline misses, at a bounded false-alarm cost.

PREDICTED, stated before data:
- GPT-2 errs on roughly 25–45% of eligible DPR items.
- Of the errors the margin baseline does NOT flag (margin >= 0.35 nats),
  the RTRM flag catches at least 30%.
- The combined flag (RTRM OR margin) beats margin-only sensitivity by at
  least 10 percentage points, with false-alarm rate on correct items no
  more than 20 points above margin-only.
- The effect is reported separately for clue-before-pronoun and
  clue-after-pronoun items. For clue-after items the flag can only be a
  structural-prior warning (the pronoun state cannot contain the clue);
  no encoding claim will be made for that class, whatever the numbers.

## Method of record (reused exactly, not reinvented)

Model: GPT-2 small, frozen. We train nothing.

Behavioural prompt, exp-055 format verbatim:
"<sentence> What does the word <pronoun> refer to? Answer with one word:"
Behaviour = next-token forced choice between " <candidate1>" and
" <candidate2>"; margin = logit difference in nats, signed toward the
model's pick; error = pick != dataset label.
 Worked examples of the format, from exp-055 (old items, shown only to
fix the format): "The truck that the car passed was old and it rattled.
What does the word it refer to? Answer with one word:" (label truck);
"The car towed the truck because it was broken. What does the word it
refer to? Answer with one word:" (label truck).

RTRM readout, 054c-C2 / exp-055 / exp-056 recipe exactly:
- Observation layer L12 (final block output, hs[12] convention as in the
  battery record).
- PCA-64 (torch.manual_seed(0) set before pca_lowrank): compress each
  state into 64 patterns, subtract the reconstruction, read the
  remainder ("the subtracted list").
- Query side additionally subtracts the pronoun's mean state.
- Pool: the battery's honest pool (~30,099 states), plus a supplement
  built ONLY if the audit requires it, by the exp-056 supplement recipe
  (filler sentences for candidate nouns absent or thin in the honest
  pool; hygiene log mandatory: norms checked against the honest pool
  before merging, per pool-hygiene rule 22; original pool files
  untouched).

RTRM FLAG (primary, exp-056's predeclared mechanical rule):
FLAG iff (competitor's best-occurrence rank < label's rank) OR
(label rank > 500) OR (label absent from the pool list).
Secondary flag (exp-055's stricter rule — label rank <= 100 AND
competitor below label — reported as CORRECT/RISK) is recorded for
continuity and is NOT the primary; no post-hoc swapping.

BASELINES the flag must beat:
- B1, margin flag (primary baseline): flag iff forced-choice margin
  < 0.35 nats. The 0.35 threshold is taken from exp-055's own record,
  where margins of 0.01, 0.21, and 0.33 were documented as near-tie
  noise. Fixed here, before new data.
- B2 (appendix only): threshold-free comparison (margin as a ranking
  score vs the flag's rank features). Statistics live in the appendix;
  the headline is the per-item table in words.

## Dataset

Primary source: Definite Pronoun Resolution (DPR), Rahman & Ng,
EMNLP-CoNLL 2012 — test split, 564 items (verified 2026-11-13:
TensorFlow Datasets catalog lists test 564 / train 1,322; dataset
homepage at UT Dallas, hlt.utdallas.edu/~vince/data/emnlp12/). Each item:
one sentence, one pronoun, two candidate referents, one label. DPR items
come in twins: two sentences differing in one trigger word, with the
label flipped.

Reserve top-up: if eligibility (below) leaves fewer than 500
confirmatory items, top up from the DPR TRAIN split by a seeded random
draw (we train nothing, so the train split is legitimate test material
for us), reported as its own stratum, never silently pooled.

Eligibility for the confirmatory set (all fixed here, applied at audit):
1. Both candidates are single alphabetic words, distinct, and each is a
   single GPT-2 token in its " <word>" form.
2. Both candidates occur in the sentence at or before the pronoun.
3. The pronoun is the first occurrence of that pronoun token in the
   prompt (exp-055's rule), so the measured state is unambiguous.
4. Pool support: each candidate word has >= 10 occurrences in the
   (possibly supplemented) pool; items failing this are excluded and
   counted, with the words named — thin neighbourhoods are reported,
   never silently carried (exp-055's cat/truck lesson).
5. Label defensibility spot-checked in the audit (below); any item the
   audit judges indefensible is excluded and quoted verbatim in the
   audit file.

Clue-position stratification (mechanical, from the twin structure):
pair each item with its twin; find the first token where the twins
differ; class A if that token sits before the pronoun, class B if after
it, class U if no twin is found. Primary claims are made per class and
pooled; class U is descriptive only.

Contamination, stated plainly: DPR has been public since 2012 and
GPT-2's WebText may contain it; we cannot verify either way. Our
no-leakage rule governs training WE do — we train nothing. Contamination
would inflate the model's behaviour and the margin baseline with it,
biasing AGAINST an incremental RTRM claim, so the win condition below
stays conservative under it.

## Gates, in order (standing checklist applies throughout)

1. SAMPLE-DATA AUDIT, before any run: hand-picked good examples plus a
   time-of-day-seeded random sample (seed = local HHMM at audit start,
   python random.Random, both recorded in the dated audit file). Each
   sampled item checked for: exact prompt text after formatting, label
   defensibility, candidate tokenisation, pool counts, clue class,
   vacuous-by-construction risks. No showcase pictures until it passes.
2. PILOT: 60 eligible items, seeded draw, run end-to-end. Pilot purpose
   is pipeline validation ONLY (files complete, ranks reproduce on a
   re-run of 5 items, margin and flag code paths both exercised). Pilot
   items are EXCLUDED from the confirmatory set.
3. CONFIRMATORY RUN: the remaining eligible items (target >= 500).
   One experiment at a time; append-only folder; log + monitor per the
   self-monitoring rule.
4. Working folder (unnumbered, per the no-earmarking rule):
   Jetson ~/robot-mind/pronoun-battery-v2/path2-early-warning/

## Decision bars (locked here; refutation at full prominence)

Let MISS = a model error the margin baseline B1 did not flag.
- INCONCLUSIVE if fewer than 30 MISS items in the confirmatory set
  (the exp-201 failure mode: the interesting class can be empty, and a
  percentage of nothing is not a result).
- WIN (RTRM adds early warning beyond output confidence) requires ALL:
  (i) RTRM flag catches >= 30% of MISS items;
  (ii) combined (RTRM OR B1) sensitivity exceeds B1-only sensitivity by
       >= 10 percentage points;
  (iii) combined false-alarm rate on correct items exceeds B1-only by
        <= 20 percentage points;
  (iv) one-sided Fisher exact test on flag x error among B1-unflagged
       items, p < 0.05 (appendix statistic supporting the table, never
       the headline).
- REFUTED if >= 30 MISS items and (i) or (ii) fails. A refutation is
  reported with the same prominence and the same per-item table.
- Secondary, reported regardless: the exp-056 H1 table (flag x error on
  all items), and per-class (A/B) versions of every number above. No
  pattern claim will be drawn from any class with < 30 MISS items.

## Reporting (SME-first)

The headline deliverable is the per-item table in words: full prompt
verbatim, label, label rank, competitor rank, lean, RTRM flag, margin,
margin flag, model pick, right or wrong — with the MISS items the RTRM
flag caught shown as worked examples with their subtracted lists.
Statistics are the appendix. PREDICTED vs ACTUAL stated in the summary,
refutations at full prominence. Nothing is posted, pushed, or published
without Paul's explicit word.

## Known limitations, declared in advance

- GPT-2 small only. A win licenses Path 1 (the auditing game) and a
  modern-model follow-up; it does not by itself generalise.
- Forced choice on the first answer token is the battery's established
  behavioural read; it is not free generation.
- Class B items test a structural prior, not encoding (exp-056
  correction). Class A is where an encoding claim could live — and
  exp-057's failure at a post-clue position is the standing warning
  against expecting one.
