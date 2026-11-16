# exp-201 Token-Exposure Control — RESULTS
Date: 2026-11-15 (analysis 2026-11-16T21:16:50.794128+00:00 UTC)

## VERDICT: FAMILIARITY CONFIRMED

Decision bars (locked): FAMILIARITY CONFIRMED if expA/B/C specificity ≤10% in ≥8 of 9 arm-seeds AND expD reversed-trained specificity ≥50% in ≥2 of 3 seeds AND expD forward-only specificity ≤15% in ≥2 of 3 seeds. REFUTED if expA/B/C pooled specificity ≥25%. INVALID if forward behaviour departs from exp-201 (A~60, B~55, C~29) by >15 in any arm.
- expA/B/C arm-seeds with specificity ≤10%: 9/9
- expA/B/C pooled specificity: 0.033 (chance 0.017)
- expD reversed-trained seeds with specificity ≥50%: 2/3
- expD forward-only seeds with specificity ≤15%: 3/3
- INVALID flags: none

## PREDICTED vs ACTUAL
PREDICTED (referee, adopted): expA/B/C reversed behaviour ≤3/60 per seed; MISSING share collapses (pooled <50%); specificity ≤10%; godparent set median rank in WEAK zone for expA/B. expD reversed-trained: behaviour ≥24/30, specificity ≥50%, STORED common; expD forward-only like expB.

## expA
| seed | rev behav 8tok | fwd behav 8tok | bands S/W/M (rev, hs11) | spec best | mean beaten | set med rank |
|---|---|---|---|---|---|---|---|
| s0 | 0/60 | 60/60 | 0/26/34 | 4/60 (0.07) | 0.52 | 136 |
| s1 | 0/60 | 60/60 | 0/26/34 | 2/60 (0.03) | 0.53 | 134 |
| s2 | 0/60 | 60/60 | 0/27/33 | 1/60 (0.02) | 0.53 | 134 |

## expB
| seed | rev behav 8tok | fwd behav 8tok | bands S/W/M (rev, hs11) | spec best | mean beaten | set med rank |
|---|---|---|---|---|---|---|---|
| s0 | 0/60 | 56/60 | 0/20/40 | 1/60 (0.02) | 0.52 | 131 |
| s1 | 0/60 | 57/60 | 0/18/42 | 0/60 (0.00) | 0.52 | 136 |
| s2 | 0/60 | 57/60 | 0/17/43 | 1/60 (0.02) | 0.52 | 147 |

## expC
| seed | rev behav 8tok | fwd behav 8tok | bands S/W/M (rev, hs11) | spec best | mean beaten | set med rank |
|---|---|---|---|---|---|---|---|
| s0 | 0/60 | 31/60 | 0/19/41 | 4/60 (0.07) | 0.52 | 126 |
| s1 | 0/60 | 30/60 | 0/18/42 | 2/60 (0.03) | 0.51 | 123 |
| s2 | 0/60 | 16/60 | 0/21/39 | 3/60 (0.05) | 0.50 | 121 |

## expD
| seed | rev behav 8tok | fwd behav 8tok | bands S/W/M (rev, hs11) | spec best | mean beaten | set med rank |
|---|---|---|---|---|---|---|---|
| s0 | 9/60 | 49/60 | 18/24/18 | 15/60 (0.25) | 0.64 | 82 |
| s1 | 10/60 | 46/60 | 21/21/18 | 16/60 (0.27) | 0.64 | 78 |
| s2 | 11/60 | 36/60 | 21/21/18 | 12/60 (0.20) | 0.64 | 75 |

Arm D split (30 reversed-trained / 30 forward-only pairs):
| seed | subset | rev behav 8tok | bands S/W/M | spec best | mean beaten |
|---|---|---|---|---|---|
| s0 | revtrained | 9/30 | 18/12/0 | 15/30 (0.50) | 0.94 |
| s0 | fwdonly | 0/30 | 0/12/18 | 0/30 (0.00) | 0.35 |
| s1 | revtrained | 10/30 | 21/9/0 | 16/30 (0.53) | 0.94 |
| s1 | fwdonly | 0/30 | 0/12/18 | 0/30 (0.00) | 0.35 |
| s2 | revtrained | 11/30 | 21/9/0 | 12/30 (0.40) | 0.94 |
| s2 | fwdonly | 0/30 | 0/12/18 | 0/30 (0.00) | 0.35 |

