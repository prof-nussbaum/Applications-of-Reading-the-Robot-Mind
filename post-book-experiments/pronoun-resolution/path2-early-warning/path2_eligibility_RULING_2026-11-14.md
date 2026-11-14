# Path 2 — eligibility ruling (2026-11-14, before any draw or scoring)

## What happened

The pilot worker stopped at its step 1, as instructed: its materialise
script computed 737 eligible items against the decision note's locked
677, exited nonzero, and wrote nothing. No pilot draw was made, no seed
consumed, no item scored. The assertion did its job.

## Root cause (verified by Francis by direct recomputation)

Case sensitivity in the worker's materialise_draw.py. Audit head words
carry surface case ("Bill", "Debbie"); pool labels are lowercased by the
battery's own convention (run_055.py: tok.decode(...).strip().lower()),
and pool_hygiene.json's below-10 keys are lowercase. The script compared
raw heads against lowercase keys, so capitalised heads never matched
and only 54 items were excluded. The decision note's computation
compared case-insensitively and is correct.

The worker's alternative hypothesis — that the decision note had used
an any-word-of-the-phrase check — was tested and rejected: on this data
the phrase check and the head check exclude identical item sets (no
eligible item has a multiword candidate phrase beyond article + word).

## Ruling

**677 governs** (test 222, train 455; clue A 97, B 578, U 2), exactly as
the pool-count decision note states. R4 comparisons are made
case-insensitively, matching the pool's label convention. The fix is a
one-line change to materialise_draw.py (lowercase the heads before the
below-10 comparison), recorded here and in PILOT_REPORT.md; the script
had produced no data, so nothing is contaminated.

## Minor correction, same neighbourhood

The decision note says 59 words below 10. The machine-readable list in
pool_hygiene.json (candidate_counts_below_10_built) has 60 entries —
it includes "microwave" (9) and "listener" (9), which the BUILD_REPORT
prose list dropped. The json governs; the item counts above were
computed from it and are unaffected.
