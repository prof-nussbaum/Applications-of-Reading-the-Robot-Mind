# exp-054f — PCA-dimension robustness sweep

Follow-up to exp-054e. Re-runs the frozen 054e residual recipe with a
single PCA fit (q=500, torch.manual_seed(0), the 054e seed) truncated to 22
values of k from 1 to 500. Not a search for the best k — a check that the
published k=64 sits on a broad plateau rather than a knife-edge.

**Result:** the predeclared U-shape is confirmed. All 4 ceiling referents
(dog, king, car, baby) rank ≤500 contiguously from k=16 to k=288 — an
18-fold range containing the published k=64. Past k=320 the curve degrades
as predicted (king falls back to ~#18,000, near its raw rank #17,061).
k=64 remains the published number; nothing was re-tuned.

**Gate note:** the first run's validation gate demanded exact rank
reproduction at k=64 and failed (king #69 vs published #124). The gate was
miscalibrated — 054e's own notes document draw-to-draw rank variation
(king #50–130). Corrected to the documented draw ranges plus a
residual-energy match; see NOTE_054f.txt. The failed run is kept as the lab
record (append-only).

## Files

- `HYPOTHESIS_054f.txt` — predeclared prediction, metric, and decision bar
- `run_054f.py` — the sweep (single q=500 PCA fit, truncated per k;
  validation gate on 054e's documented draw ranges)
- `results_054f.json` — per-k ranks and residual-energy fractions
  (machine-readable)
- `RESULTS_054f.txt` — the full k table (dog/king/car/baby ranks per k)
- `SUMMARY_054f.txt` — PREDICTED vs ACTUAL write-up, including the pool
  occurrence check (king 3, baby 13, car 18, dog 21 of 30,099 states —
  no simple link between occurrence rate and large-k fragility)
- `NOTE_054f.txt` — documents the gate correction
- `u_shape_054f.png` — plot of referent rank vs k: the U-shape, the 500
  bar, and the k=16–288 plateau containing the published k=64
- `README.md` — this file

## Suggested follow-up comment

Post as a reply to the existing LinkedIn announcement (not a new post),
after this folder is merged to main:

---

Follow-up on the number 64, since it's the obvious objection: it was a
choice, not a derivation. So I re-ran the frozen recipe with that knob
swept across 22 values from 1 to 500, prediction written down first
(U-shape: bad at both ends, good in the middle).

All four test nouns stay visible from k=16 to k=288 — an 18-fold range.
Past 320 it degrades exactly as predicted. The result doesn't live or die
on 64.

Honest footnote: my first validation check demanded the rerun reproduce
the published ranks exactly, and it failed. The failure was in my check,
not the result — ranks wobble between PCA redraws, and the method's own
notes say so. Fixed the check, wrote up why, kept the failed run in the
record.

Full sweep and scripts are in the repo now,
post-book-experiments/pronoun-resolution/exp-054f/.

---
