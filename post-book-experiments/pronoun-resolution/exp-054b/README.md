# exp-054b — symmetric token demeaning: refuted

**Idea (Paul's):** exp-054 subtracted the GLOBAL pool mean before the
PCA-64 codec. Refinement: subtract the mean OF THAT TOKEN — center each
state by its own word's mean occurrence ("how this occurrence deviates
from the typical occurrence of its word"), removing the token-specific
carrier. Two variants: (A) token-demean only, no PCA; (B) token-demean,
then global-center, then PCA-64 codec.

**Result:** REFUTED. Demeaning pool content words by their own means
deletes the word identity being searched for. The failure mode is
instructive — it showed the demeaning must be asymmetric (query side
only), which became the exp-054c recipe. exp-054 is untouched
(append-only).

## Files

- `run_054b.py` — both variants
- `results_054b.json` — per-item ranks
- `SUMMARY_054b.txt` — PREDICTED vs ACTUAL write-up (refutation at full
  prominence)
- `README.md` — this file
