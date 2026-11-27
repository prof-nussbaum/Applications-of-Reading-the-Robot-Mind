# Step-4 (Arm A, decoys) — protocol LOCKED — 2026-11-27

LOCKED by Paul's word in chat, 2026-11-27 ("Approve as-is"). The hash
of this commit is posted to Fable in a comment before the run, per
R-017 §5. No scored fine-tune and no retrieval readout of any kind
before Fable's READY on this commit.

Source: Fable's R-017 §6 draft protocol (PR #18 comment 7088008866),
corrected to Arm A only per R-020, with the four pre-lock checks settled
and the pilot result written in, per R-021 conditions 3–4.

## Question

Variant T showed the model's state retrieves the training sentence the
ability came from. But a word search ties it there — the forward and
reversed sentences share the same words, so BM25 scores them equally.
A reader will object that a phrase search would break that tie. The
decoy arm answers that objection by design: the decoy sentence is built
to mislead a word search on purpose, holding the probe's rarest word
plus the godparent's name, and never giving the answer. If the model's
state still finds the true source anyway, that is retrieval beyond
the words.

## Construction (fixed)

- Model: GPT-2 small. Training recipe identical to exp-201 / E1b-fresh:
  SGD lr 5e-4, no momentum, batch 1, 200 epochs.
- Scored dataset: `exp-201-step4/dataset_step4.json` — 60 pairs, 480
  texts. Per pair: target "G is the godparent of C." x6, decoy
  "D met the godchild of G last week." x1, filler x1. No reversed
  sentences — Arm A trains the forward fact only.
- Names, towns, decoy people: fresh throughout. Towns are NOT used
  (they belong to Arm B, dropped by R-020). First-token audit: zero
  overlap with E1, with E1b-fresh, and pilot-vs-scored. Probe strings
  "'s godchild" and "'s godparent" occur 0 times in the training texts;
  the decoy holds "godchild of G", never "G's godchild", so it cannot
  teach the probe continuation; the decoy never names the pair's
  godchild.
- Probe: "G's godchild is" → C. Behaviour pass = the godchild's FULL
  name in the first 8 decoded tokens (the exp-201 / E1b-fresh pass rule).
- Readout (variant T, unchanged): the state at layer 11, at the
  probe's final position; pool = Wikipedia sentences plus every
  distinct fine-tuning sentence; neighbours shown as their source
  sentences. Same construction for the fine-tuned model and for the
  untrained control.
- Three readouts scored on the same pool: the fine-tuned model, the
  untrained model (Control U), and BM25 (Okapi k1 = 1.5, b = 0.75,
  word tokens lowercased — the exact BM25 from the seed-0 controls).
- Seeds: three scored seeds, 0/1/2. (The pilot used its own seed 7
  and its own names; it never enters the scored run.)

## The four pre-lock checks — all settled

1. **Check 4 — BM25 prefers the decoy (2026-11-26).** Exact BM25,
   step-4-shaped corpus (1,756 Wikipedia + 60 targets + 60 decoys +
   60 fillers, no reversed sentences), E1b-fresh names as stand-ins.
   Decoy outranks target 60/60; decoy at rank 1 in 60/60, mean decoy
   rank 1.0 vs target 2.0 (idf godchild 3.466 vs is 2.022). PASS —
   no rewrite needed.
2. **Check 2 — fresh names (2026-11-26).** Pilot and scored datasets
   generated on the Jetson, 60 pairs each; first-token audit zero
   overlap with E1, E1b-fresh, and pilot-vs-scored. Two loud failures
   on the way, both fixed before running: the first draw came up short
   on fresh single-token names (314 against 360 needed), so the name
   pool was expanded; the generator was missing the surname-list
   definition, caught before running.
3. **Check 3 — repetition counts (2026-11-26).** Fixed before the
   pilot: target x6, decoy x1, filler x1 — 480 texts per set, the
   same total as expA. Sample-data audit PASS, 23 checks.
4. **Pilot — does the decoy disturb forward behaviour? (R-021
   conditions 1–4; PASSED 2026-11-27 04:53 EST).** Tolerance committed
   BEFORE training at `ai-rtrm-experiments/step4-armA/PILOT_TOLERANCE.md`
   (commit 35a700d, branch step4-armA): pass iff forward accuracy with
   decoys is within 10 points of without. Pilot models on the pilot's
   own seed and names, training seed 7: A with decoys (480 texts),
   B without (420 texts). Forward probe "G's godchild is", full name
   in the first 8 tokens: **A 58/60 (96.7%) vs B 55/60 (91.7%),
   difference +5.0 points — inside the 10-point tolerance. PASS.**
   First-name scoring 60 vs 56; sentence-completion probe
   ("G is the godparent of" → C) 60/60 on both. No variant-T pool
   built, no retrieval rank computed or looked at. The decoy does not
   disturb forward behaviour; the counts stand. If it had failed,
   Fable would have been told before the decoy or the counts changed.

## Bars — pooled over the three scored seeds, on the probes the model answers correctly

- **A1, finds the source:** the fine-tuned model's state puts the
  target sentence first at least **70%** of the time.
- **A2, beyond the words:** the state's top-1 rate beats BM25's by
  at least **30 points**.
- **A3, from what was learned:** the state's top-1 rate beats the
  untrained model's by at least **30 points**.

Reported with the bars, as before: the pass-versus-fail split within
stored pairs (whether the list separates the pairs the model answers
from the ones it doesn't), and the same rule-family tables as the
seed-0 variant-T records. Every retrieval figure beside its
model-free text-search baseline, per the standing rule from R-013.

## Outcome sentences, fixed now

- **All three bars met** → the paper may say the state found the
  sentence where a word search was misled.
- **A1 and A3 but not A2** → the state finds it, and so does a word
  search — no advantage shown.
- **A1 and A2 but not A3** → it beats the search but does not depend
  on what the model learned.
- **Not A1** → the state does not reliably retrieve the source once
  a decoy is present.

## What this lock does NOT do

- No reader test in this step.
- No scored fine-tune and no retrieval readout of any kind before
  Fable's READY on this lock (standing rule, R-021).
- Arm B (two-hop descriptions) is dropped to later work; the GPT-2
  line freezes for this paper after this one experiment (R-020).

## Verification done before locking (2026-11-27)

1. The scored dataset's recorded draw seed, read on the Jetson:
   `dataset_step4.json` records `draw_seed` 20261126 (60 pairs,
   480 texts), and `dataset_pilot.json` records 20261126 as well.
   The code was right (one `DRAW_SEED = 20261126` for both sets);
   the generator's docstring line 24 wrongly said 20261127 and was
   corrected to 20261126 before this commit.
2. The corrected referee's guide (`guide/READING_A_RECREATION.md`,
   Case 2 rows and cheap-statistics sentence as verified 2026-11-26)
   rides with this same PR, per R-021 §3.
3. This branch carries only this lock, the pilot tolerance (at
   35a700d), the two datasets' builders/audits, the check-4 and pilot
   records, and the guide — nothing else.
