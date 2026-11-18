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

---

## Addendum — 2026-11-18 (referee R-006; verified by Francis)

Two corrections to the reading of this test, both verified against
`answer_key.json` by recomputing the referee's check.

**1. The words shown were the lens's, not RTRM's.** The reader's
evidence was the 20 highest-logit vocabulary tokens through ln_f
and the unembedding at hs[11] — a logit-lens list. The test as run
measures the legibility of a lens list. It is not evidence about
the RTRM cosine neighbour list, and must not be cited as support
for the book's distinctive claim about that list.

**2. The locked bars were the weakest fixed rules on the page.**
Each item printed the correct name's rank implicitly (the name
appears in the ranked list or not), so fixed thresholds on that
same information score:

| Task 1 rule: PASS if rank ≤ k | Balanced accuracy |
|---|---|
| k = 1 (the locked bar) | 0.647 |
| k = 2 | 0.905 |
| k = 3 | 0.886 |
| k = 5 | 0.866 |
| k = 20 | 0.843 |
| **Reader** | **0.895** |

| Task 2 rule: STORED if rank ≤ k | Balanced accuracy |
|---|---|
| k = 10 (the band rule, the locked bar) | 0.800 |
| k = 20 (the name appears at all) | 0.917 |
| **Reader** | **0.800** |

No forward-only name ranked better than 37. Fair statement of the
finding: with no calibration, the reader chose a cut-off close to
the best one for predicting behaviour (0.895 vs 0.905), and too
cautious a one for stored-versus-familiar (0.800 vs 0.917). The
locked verdicts above (Task 1 PASS, Task 2 NOT PASS) stand as
recorded; the comparisons in this addendum supersede the "25
points above the rule" framing in the original summary.

**Corrected test (E1b), designed:** the RTRM neighbour list as the
readout, arm D seeds 1 and 2 (120 probes, unseen by the reader),
bars set by the best rank-threshold rule computed on seed 0 with
that readout, no halfway feedback, judgments in a fixed form,
protocol locked and referee-approved (READY) before the reader
opens the set.
