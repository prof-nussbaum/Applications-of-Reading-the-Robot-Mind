# exp-201 Token-Exposure Control — LOCKED PROTOCOL
Locked 2026-11-15, before dataset build, audit, or any training.
Origin: referee's design (relayed by Paul 2026-11-15), adopted in full;
bars and audit gates set by Francis. Folder (unnumbered, per Paul's rule):
Jetson ~/robot-mind/exp-201-exposure/

## Question
Do the logit-lens verdict bands (STORED ≤10 / WEAK 11–100 / MISSING >100,
hs[11]) measure storage of a fact, or familiarity with the answer token?
exp-201's reversed probes were confounded: in the no-bridge arms the
godparent names never occurred mid-sentence in training (verified counts:
201b/201c " G" occurrences = 0; 201a = 60, one per name, from the bridge),
so MISSING was forced by an untrained token, and 201a's WEAK verdicts were
shown by the specificity check to be set-wide (correct name beats only
54.9% of the other 59 godparent names). This experiment removes the
confound and adds the arm exp-201 lacked: pairs actually trained in the
reversed direction.

## Fixed from exp-201 (nothing else changes)
- Same 60 fictitious godparent/godchild pairs, same fact sentences
  "{G} is the godparent of {C}.", same bridge
  "The godchild of {G} is {C}.", same probes.
- Same recipe: GPT-2 small, SGD lr 5e-4, no momentum, batch 1, 200 epochs,
  max_length 64, pad = eos. Per-epoch permutation generator seed =
  (seed+1)*100000 + epoch; torch.manual_seed(seed).
- Seeds 0, 1, 2 for every arm (12 fine-tunes).

## New material
FILLERS (identical in every arm, one per godparent name, x1):
  "The parcel was delivered to {G} this morning."
  — G mid-sentence, no godchild name, no "is {G}" pattern.
REVERSED BRIDGES (arm D only, 30 of 60 pairs, x1):
  "The godparent of {C} is {G}."
  — the exp-201 bridge construction in the opposite direction.
  Pair selection: Python random.Random(201).shuffle of indices 0..59,
  first 30, recorded in the audit.

## Arms (per seed)
- expA = facts x6 + forward bridges x1 + fillers        (480 texts)
- expB = facts x6 + fillers                             (420 texts)
- expC = facts x1 + fillers                             (120 texts)
- expD = facts x6 + fillers + reversed bridges (30 prs) (450 texts)

## Audit gates (BEFORE any training; failure stops the run, loudly)
1. Filler check: 60 texts; each contains its G mid-sentence exactly once;
   no filler contains any godchild name; none matches "is {G}".
2. Leakage: the probe string "{C}'s godparent is" and the substring
   "'s godparent" occur 0 times in every arm's training texts.
3. Exposure recount per arm: mid-sentence " G" counts per name —
   expA 2 (bridge+filler), expB 1, expC 1, expD 1 (+1 after "is" for the
   30 reversed-bridge names). Deviations stop the run.
4. Hand-check: 5 fillers + 5 reversed bridges read by eye in the audit file.

## Scoring (blind order, as exp-201)
1. Lens verdicts + specificity computed and saved, timestamped, BEFORE
   any generation is scored. Lens = ln_f applied, hs[11] primary
   (hs[12] + final rank recorded), bands as exp-201.
2. Behaviour: greedy generation, pass = full expected name within 8
   tokens (50-token window also recorded), forward and reversed probes.

## Metrics
PRIMARY — specificity on reversed probes: fraction of items where the
correct godparent's first token ranks best among the 60 godparent first
tokens at hs[11]. Chance = 1/60 = 1.7%. Also: mean fraction of other
names beaten (chance 50%).
SECONDARY — reversed behaviour pass counts; band counts per arm;
set-wide lift (median hs[11] rank over all 60 godparent names per item);
forward behaviour (sanity).

## PREDICTED (referee's, adopted; Francis concurs)
- expA/B/C: reversed behaviour near 0 (≤3/60 per seed); MISSING share
  collapses vs exp-201 (pooled <50%); specificity near chance (≤10%);
  godparent set median rank in the WEAK zone for expA/B.
- expD reversed-trained pairs: behaviour ≥24/30 per seed, specificity
  ≥50%, STORED common. expD forward-only pairs: like expB (specificity
  ≤10%, behaviour ≤3/30) — skill bleed, if it occurs, will show here
  and is reported either way.

## Decision bars (locked)
- FAMILIARITY CONFIRMED if: expA/B/C specificity ≤10% in ≥8 of 9
  arm-seeds, AND expD reversed-trained specificity ≥50% in ≥2 of 3
  seeds, AND expD forward-only specificity ≤15% in ≥2 of 3 seeds.
- FAMILIARITY REFUTED if: expA/B/C pooled specificity ≥25% (the lens
  carries item-specific reversed information after mere exposure).
- Otherwise MIXED, reported as the numbers stand.
- INVALID if forward behaviour departs from exp-201's pattern by >15
  points in any arm (A ~60/60, B ~55/60, C ~29/60): the retrain itself
  is then broken and reversed results are uninterpretable.

## Reporting
PREDICTED vs ACTUAL, per seed and pooled; SME page with verbatim items
(one reversed-trained D pair, one forward-only D pair, one expB item)
showing probe, correct name's rank, and the other names' ranks.
Append-only: this folder only; exp-201 untouched.

## Amendment 2026-11-15 (post-launch, referee's suggestions, Paul relayed)
1. CHECKPOINTS: finetune_exposure.py now saves an fp16 checkpoint every
   25 epochs (e025–e175; final model already saved at e200) for runs not
   yet started, enabling a future checkpoint sweep without a second
   retrain. Saving only — training math, order, and seeds untouched.
   Original script preserved as finetune_exposure_v1.py. Run expA_s0,
   already in flight, keeps the v1 script and will have no checkpoints.
2. SANITY GATE AS HARD STOP: forward behaviour is evaluated and reported
   FIRST when scoring lands. If the INVALID bar trips (any arm's forward
   behaviour >15 points from exp-201's 60/55/29 pattern), the reversed
   results are reported as uninterpretable and NO familiarity verdict is
   drawn — the analysis stops at the gate.
