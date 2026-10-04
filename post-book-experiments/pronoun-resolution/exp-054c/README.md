# exp-054c — asymmetric query-side pronoun demeaning (the working instrument)

**Reading:** exp-054b failed because it demeaned POOL content words by
their own means, deleting word identity. "Subtract mean of that token" is
better read as applying to the QUERY token (the pronoun): strip the
pronoun-typical carrier from the query; leave pool word identity intact.

- Pool side: exp-054 recipe (global-center, PCA-64, residual), true cosine.
- Query C1: q = h − mu_pron. Query C2: q = (h − mu_pron) with the global
  top-64 projected out (pool residuals are ~zero there anyway, so C1 ≈ C2).

**Result:** the working instrument — the C2 recipe used throughout the
study and the SME checklist. Unseeded run: its exact ranks proved
draw-dependent and exactly unreproducible (the original RNG state is
lost). This folder stays frozen as the lab record; exp-054e's seeded
numbers supersede 054c's for any public citation.

## Files

- `run_054c.py` — asymmetric recipe (C1/C2 query variants)
- `results_054c.json` — per-item ranks (unseeded; do not cite)
- `SUMMARY_054c.txt` — PREDICTED vs ACTUAL write-up
- `README.md` — this file
