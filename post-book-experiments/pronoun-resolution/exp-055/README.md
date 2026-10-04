# exp-055 — prospective test of the SME checklist on 10 new prompts

**Paul's test:** the SME uses ONLY the subtracted list. What should they
look for in new prompts to spot issues? 14 candidates hand-written;
stratified time-seeded draw (seed 1921): 5 clean + 5 tricky. Full audit at
AUDIT_055.txt. SME verdicts mechanical and predeclared. torch.manual_seed(0)
predeclared (same seed as 054e).

**Result:** the model was wrong on 4 of the 10 prompts — and the SME
checklist flags caught ALL 4 (sensitivity 4/4) with 3 false alarms. Two
patterns: a competitor noun above the label pulled the model to it, and a
referent ABSENT from the subtracted list meant a wrong guess. The
head-at-pronoun readout favored the TRUE referent where cosine leaned to
the competitor (exp-053's "head reads what cosine obscures", replicated on
new items). Refuted along the way: clean-prompt accuracy ≥4/5 (actual 1/5)
and mismatch-overturn ≥60% (actual 1/3).

**Note:** results_055.json's pron_logit fields are superseded by
results_055b_pronlogits.json (normalization bug in a cross-check; the RTRM
+ behavioral results are unaffected).

## Files

- `run_055.py`, `run_055b.py` — checklist run + pron-logit correction
- `results_055.json`, `results_055b_pronlogits.json` — per-item results
  (see note above on pron_logit fields)
- `AUDIT_055.txt` — sample-data audit
- `SUMMARY_055.txt` — PREDICTED vs ACTUAL write-up
- `README.md` — this file
