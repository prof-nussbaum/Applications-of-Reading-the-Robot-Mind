# Substitution accuracy by candidate type — the 617 (2026-11-16)

Adopted from referee R-001 (F-001 §4). Computed by `name_noun_split.py`
from banked records only; totals reproduce the banked headline exactly
(substitution 367/617, question format 320/617 — asserted in the script).

Classification rule: an item is a **name** item if both candidate heads
are capitalised proper names (Bill, Bob), a **noun** item if neither is
(sniper, terrorist), **mixed** if exactly one is.

| Class | n | Substitution accuracy | Question-format accuracy |
|---|---|---|---|
| Name | 310 | **50.0%** (155/310) | 50.6% (157/310) |
| Common noun | 226 | **73.9%** (167/226) | 52.2% (118/226) |
| Mixed | 81 | 55.6% (45/81) | 55.6% (45/81) |

Reading: the substitution format's advantage over the question format
(+7.6 points overall) is entirely a common-noun phenomenon. On items
whose candidates are proper names, substitution is at exactly chance —
the format change buys the model nothing where the candidates carry no
noun semantics to compare. Consistent with the programme's pattern:
what the formats and readouts recover is word content and class, not
item-specific referent tracking.
