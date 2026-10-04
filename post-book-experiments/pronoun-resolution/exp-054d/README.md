# exp-054d — mean removal after PCA: same result

**Test:** 054c-C2 did token-mean removal BEFORE PCA. This run swaps the
order: (1) PCA residual with global centering, (2) subtract the pronoun's
mean computed IN RESIDUAL SPACE (mean of the 77 "he" / 70 "it" pool
residuals). Pool side unchanged (exp-054 recipe, word identity intact,
true cosine). Asymmetric throughout — 054b showed symmetric pool-demeaning
deletes word identity.

**Result:** the effect is NOT an artifact of operation order — mean
removal after PCA gives the same result as before it. The instrument is
robust to this swap.

## Files

- `run_054d.py` — swapped-order recipe
- `results_054d.json` — per-item ranks
- `SUMMARY_054d.txt` — PREDICTED vs ACTUAL write-up
- `README.md` — this file
