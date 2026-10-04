# exp-054e — seeded rerun of the 054c recipe (the citable numbers)

**Recipe:** the exp-054c C2 recipe exactly — same 30,099-state pool, same
global-centering, same PCA-64 residual, same pronoun-mean query
subtraction, true cosine ranking. ONE deliberate change:
torch.manual_seed(0), predeclared before torch.pca_lowrank (054c was
unseeded; authorized by Paul 2026-09-29).

**Why:** the render-verification pass showed 054c's exact ranks are
draw-dependent and exactly unreproducible. Public claims need reproducible
numbers.

**Result:** all 4 ceiling referents ≤500 with margin — dog #56, king #124,
car #33, baby #99 (raw: #110, #17,061, #3,729, #1,196). These SEEDED
numbers SUPERSEDE 054c's for any public citation; the 054c folder stays
frozen as the lab record. This run is the source of the six rendered
subtracted-list prompt pages in `pictures/`. exp-054f later verified the
k=64 choice sits on a broad plateau (k=16–288), not a knife-edge.

## Files

- `run_054e.py` — seeded recipe
- `results_054e.json` — per-item ranks (the citable numbers)
- `render_054e.py` — source of the six subtracted-list prompt pages
- `SUMMARY_054e.txt` — PREDICTED vs ACTUAL write-up
- `README.md` — this file
