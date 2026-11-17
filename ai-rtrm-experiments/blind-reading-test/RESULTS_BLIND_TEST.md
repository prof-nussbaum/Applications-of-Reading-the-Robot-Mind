# Blind reading test — RESULTS (2026-11-17)

Reader: Paul (the domain expert; also the PI). Model: expD_s0.
Set: the 60 reversed probes of arm D seed 0, shuffled
(random.Random(20261117)); each item showed only the verbatim probe,
the correct godparent's name, and the top-20 lens words at hs[11] in
rank order. Protocol and locked bars: PROTOCOL_BLIND_TEST.md.
Scoring: score_blind_test.py against answer_key.json.

## PREDICTED (locked bars) vs ACTUAL

**Task 1 — predict, from the words alone, whether the model will
produce the correct godparent.** Bar: reader balanced accuracy must
EXCEED the lens rank-1 rule's 0.6471 (fixed rules score 0.500).

ACTUAL: raw 54/60. TP 8, FN 1, TN 46, FP 5. Sensitivity 0.889
(8 of 9 actual passes caught), specificity 0.902.
**Balanced accuracy 0.8954 — PASS, clearing the bar by ~25 points.**

The reader called PASS 13 times, all on stored-trained items whose
correct name sat at lens rank 1–5. The 5 false alarms were names at
rank 1 (×2), 3 (×2) and 5 that the model still did not produce —
the STORED-but-failed pattern, read by eye. One actual pass was
missed.

**Task 2 — sort stored from merely familiar.** Bar: reader balanced
accuracy must EXCEED the lens band rule's 0.8000.

ACTUAL: raw 48/60. TP 18, FN 12, TN 30, FP 0. Sensitivity 0.600,
specificity 1.000. **Balanced accuracy 0.8000 — exactly the band
rule's figure. The bar required exceeding it: Task 2 is a TIE with
the machine rule, scored as NOT PASS under the locked rule.**

The reader never called a merely familiar name stored (0 false
alarms in both tasks), and called 12 of the 30 stored names
familiar — the conservative pattern visible at the halfway check
persisted to the end.

## Disclosures

- **Halfway confirmation:** at the reader's request, aggregate
  scores for the first ~half were reported before he completed the
  set (Task 1 then 28/28 raw with a single actual pass among them;
  Task 2 then 0.667 balanced). No item-level results, no key
  information about unanswered items. The second-half behaviour is
  consistent with the first (same conservative pattern on Task 2;
  Task 1 PASS calls concentrated where the correct name ranked 1–5).
- **Parsing:** the reader marked judgments by hand with
  inconsistent bold markers; the scorer searches the Task lines for
  the judgment words. One Task 2 line (BT-34) reads "STORE" and is
  counted as STORED (the alternatives are STORED / MERELY
  FAMILIAR). All 60 items were answered on both tasks.
- One reader, one model seed, 60 items with 9 behavioural passes:
  a first reading, not a population claim.

## Reading of the result

The book's distinctive claim — that an expert can *see* what the
readout holds by reading the words — gets its first direct support
on Task 1: the reader predicted the model's behaviour from the
lists far better than the mechanical rank rule the lists come with
(0.895 against 0.647). On the storage judgment he matched the
lens's own band rule exactly, with the same conservative shape
(perfect specificity, partial sensitivity): where the lens's WEAK
band blurs stored into familiar, the human reader blurs the same
way. The list is legible for what the model will do; the
familiar/stored boundary is as hard to see as it is to compute.
