# Step-4 pilot tolerance — to be committed BEFORE any pilot training

R-021 condition 3: the pass mark is written first. Paul has approved 10.

**The pilot passes if forward accuracy with decoys is within 10 percentage
points of forward accuracy without decoys, on the pilot's own seed and
names.**

Conditions (R-021 §2, all four):
1. Own seed, own names: the pilot uses `exp-201-step4-pilot/dataset_pilot.json`
   (draw seed 20261126, one draw for both sets, pilot first), never in
   the scored run; the scored run uses
   `exp-201-step4/dataset_step4.json` (same draw, scored second).
2. Behaviour only: train two models, one with decoy sentences and one
   without. Measure forward-probe accuracy ("G's godchild is") on each.
   Do not build a variant-T pool. Do not compute or look at any retrieval rank.
3. This tolerance line, committed before training.
4. The result goes in the lock: the locked protocol states the pilot's two
   accuracies, pass or fail. If it fails, tell Fable before changing the
   decoy; do not tune the decoy by repeated pilots without a message.

Repetition counts (check 3, fixed before the pilot): target x6, decoy x1,
filler x1 — 480 texts per set. Standing rule restated (R-021): no scored
fine-tune, and no retrieval readout of any kind, before READY.
