# Control: list lean vs the model's substitution pick (2026-11-16)

Adopted from referee R-001 (F-001 §4). Computed by `lean_vs_subpick.py`
from `results_control.json`; banked checks reproduce exactly (lean ==
label 74/120 = 61.7%; substitution pick == label 96/120 = 80.0%).

| Comparison | Rate |
|---|---|
| Lean == substitution pick, all 120 | 60.0% (72/120) |
| Lean == substitution pick, number items | 65.0% (39/60) |
| Lean == substitution pick, gender items | 55.0% (33/60) |
| Lean == question-format pick, all 120 | 46.7% (56/120) |
| Lean == label, all 120 | 61.7% (74/120) |

Reading: on the control bed, the list's lean agrees with the model's
substitution pick only 60% of the time, and with the model's
question-format pick less often than chance (46.7%) — while the
substitution pick itself is right 80% of the time. Even where a
reliable pick exists, the lean does not track it. (The Q3 test,
`../q3-pick-tracking/`, asks the same question at the answer position
on the full confirmatory bed, pair-level; it fails there too.)
