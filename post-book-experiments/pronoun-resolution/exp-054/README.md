# exp-054 — PCA-as-codec residual (Paul's Kaggle trick, turned inward)

**Idea (Paul's):** use an autoencoder AS A CODEC — subtract its
reconstruction from the original state vector. The residual keeps what the
AE could NOT capture (the small perturbations) with the big carrier
variations removed. Then run the cosine neighbor search on RESIDUALS. This
is his Kaggle power-line trick (AE learns "normal", the residual reveals
the anomaly) turned inward on the model.

**Substitution disclosed:** Paul's pretrained AE weights were not on
either machine, so a linear AE with a 64-dim neck — which IS PCA — was
used instead. Top-64 captures 99.16% of variance; the residual is 0.84%
of the vector's energy.

**Result:** all 4 ceiling referents move under rank 500 (raw: dog #110,
king #17,061, car #3,729, baby #1,196) — but so do wrong nouns. The
residual surfaces drowned NOUN content generally, not the winner. Picking
the winner remains the head's job (see exp-053). Unseeded run; see
exp-054e for the seeded, citable numbers.

## Files

- `run_054.py` — PCA-64 residual + cosine ranking
- `results_054.json` — per-item ranks
- `SUMMARY_054.txt` — PREDICTED vs ACTUAL write-up
- `README.md` — this file
