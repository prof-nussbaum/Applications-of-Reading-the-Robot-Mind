# exp-052 — positive controls: behavior vs cosine neighborhoods

Paul's question: not whether the model gets the pronoun right, but whether
the RTRM cosine-similarity method can *view* the referent in the pronoun's
neighborhood when the pronoun is not confused. Do the spider/trophy
pictures mean something, or is the method blind even when resolution
succeeds?

**Method:** 10 audited pronoun prompts (4 ceiling items + 6 pair items).
For each: the model's own substitution behavior (correct vs wrong
referent) alongside the RTRM cosine neighborhood of the pronoun state.

**Result:** geometry and behavior don't track. 3 of the 4 ceiling
referents are invisible to cosine neighborhoods, yet the model's behavior
is decisive (e.g. king: −4.36 vs −11.45). Absence from the neighborhood
does *not* mean absence of resolution. This is the finding that forced the
README's "What we don't claim" section.

## Files

- `run_052.py` — behavior + neighborhood measurement
- `results_052.json`, `results_052b.json` — per-item results
- `AUDIT_052.txt` — sample-data audit (hand-picked + time-seeded draw)
- `SUMMARY_052.txt` — PREDICTED vs ACTUAL write-up
- `exp052_*_prompt_pages_L12.pdf` — the 10 rendered prompt pages (raw-list
  era: the puzzle's starting point; the raw list was later retired as
  scaffolding)
- `README.md` — this file
