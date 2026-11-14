# DRAFT — Bounded rewording of the exp-055/056 claims (2026-11-14)

Status: **DRAFT for Paul's word only. Nothing here is posted, published, or
pushed.** The repo is untouched. Paul merges any PR himself.

Why this draft exists: at 05:30 UTC on 2026-11-14, after the Path 2
confirmatory result landed, Francis said he would draft the bounded
rewording of the exp-055/056 claims for the bundle either way. This is
that draft.

Sources checked for this draft:
- `~/workspace/path2_RESULTS_confirmatory_2026-11-14.md` (617 items,
  all headline numbers recomputed from the raw records before delivery).
- Live repo read 2026-11-14 12:08 UTC, commit e81befc2:
  `post-book-experiments/pronoun-resolution/README.md` — the exact
  published wording quoted in section 1 below.
- exp-056 record: `~/memory/2026-10-06.md` and `~/memory/2026-10-07.md`
  (H1 borderline pass, CORRECTION_056.txt, supplement-pool caveat).

## 1. What the repo says now (verbatim)

From the pronoun-resolution README, "The arc in five sentences", point 5:

> "On 10 new prompts, drawn by a time-seeded random draw, the checklist's
> flags caught all 4 forced-choice misses (2 robust, 2 near ties) — with
> 3 false alarms. Sensitive, not specific. That is the honest performance
> of the instrument."

From "What we claim":

> "On 10 new prompts, checklist flags caught every forced-choice model
> error (4/4: 2 robust, 2 near ties), with false alarms (3/10)."

From "Limitations":

> "exp-055 is a 10-prompt pilot: thin token neighborhoods (cat: 8 pool
> occurrences, truck: 6), near-tie margins treated as noise, and one soft
> label (T7)."

What the live repo does **not** contain (verified in the same read):
exp-056 is not in the repo at all — the study lists exp-052 through
exp-055 only. Neither is exp-201, nor the exp-320 correction. So the
055 wording above is the only checklist claim a reader can currently see,
and it still reads as a live instrument description.

## 2. What Path 2 found (PREDICTED vs ACTUAL, pooled, n = 617)

The test was predeclared before any scoring, on the DPR set
(Rahman & Ng 2012), GPT-2 small, the 054c-C2 subtracted list at L12,
the exp-056 flag rule, against the final-answer margin baseline
(flag at < 0.35 nats). Confirmatory set: 617 items (test 198, train 419;
clue-before-pronoun 86, clue-after 529, unclassed 2). The 60 pilot items
were excluded.

| | PREDICTED | ACTUAL |
|---|---|---|
| Model error rate | 25–45% | **48.1%** (297 of 617) |
| RTRM catch of margin-missed errors (268 of them) | ≥ 30% | **76.1%** (204 of 268) — PASS |
| Combined sensitivity gain over margin-only | ≥ +10 points | **+68.7 points** (9.8% → 78.5%) — PASS |
| False-alarm cost on correct items | ≤ +20 points | **+65.9 points** (10.6% → 76.6%) — FAIL |
| Flag discriminates errors from correct answers (margin-unflagged) | Fisher p < 0.05 | **p = 0.296** — FAIL |

The flag fires on **72.8% of the items the model got right**
(RTRM-only false-alarm rate). Threshold-free appendix: margin AUC 0.520,
flag balanced accuracy 0.518 — both at chance. Same shape in every split
and both clue classes.

Verdict as locked: **NOT WIN**. The protocol defined REFUTED narrowly
as bars (i) or (ii) failing, so that formal label was not triggered.
That is a drafting gap, owned here, not a loophole to hide behind. In
plain words: **the "early warning at bounded cost" claim is dead.** A
warning that fires on three of every four correct answers is not a
warning.

Why it failed — the mechanism is visible in the records: in the
47,499-state pool the label ranks run into the thousands, so the
rank->500 leg of the exp-056 flag fires constantly. The checklist was
calibrated on small hand-written sets (10 prompts in 055, 24 in 056).
At naturalistic scale it degenerates toward flag-always.

Scope limits, stated with the result: GPT-2 small only; forced choice
on the first answer token, not free generation; DPR has been public
since 2012 and may be in GPT-2's training text, which would inflate
behaviour and the margin baseline together; for clue-after items
(529 of 617) the flag can only be a structural-prior warning — the
pronoun state cannot contain a clue that comes after it — so no
encoding claim is made for that class whatever the numbers.

## 3. Proposed replacement text

### 3a. Replace arc point 5 with:

> 5. On 10 new prompts, drawn by a time-seeded random draw, the checklist's
> flags caught all 4 forced-choice misses (2 robust, 2 near ties) — with
> 3 false alarms. That was a 10-prompt pilot. A predeclared large-scale
> test on 617 natural pronoun items (Path 2, 2026-11-14) did not hold up:
> the flag caught 76.1% of the errors the output margin missed, but it
> also fired on 72.8% of the items the model got right, and it did not
> separate errors from correct answers better than chance (Fisher
> p = 0.296). The checklist as calibrated does not generalise beyond its
> pilot context. See "Update, 2026-11-14" below.

### 3b. Replace the "What we claim" bullet with:

> - On 10 new prompts, checklist flags caught every forced-choice model
> error in that pilot (4/4: 2 robust, 2 near ties), with false alarms
> (3/10). That pilot result did **not** generalise: on 617 natural
> items the same flag caught most margin-missed errors (76.1%) only by
> flagging almost everything, firing on 72.8% of correct items. We no
> longer present the checklist as an early-warning instrument. What
> survives from this arc is the method finding — subtracting the
> predictable part makes noun candidates visible in the neighbour list —
> not the warning claim.

### 3c. Add to "Limitations", and add a new dated section:

> - exp-055 is a 10-prompt pilot: thin token neighborhoods (cat: 8 pool
> occurrences, truck: 6), near-tie margins treated as noise, and one soft
> label (T7). Its checklist result was superseded by the Path 2
> confirmatory test below; see "Update, 2026-11-14".
>
> ## Update, 2026-11-14 — Path 2 confirmatory test (NOT WIN)
>
> PREDICTED vs ACTUAL, pooled, n = 617 (DPR test 198 + train 419;
> clue-before 86, clue-after 529). Error rate predicted 25–45%,
> actual 48.1%. The RTRM flag caught 204 of 268 margin-missed errors
> (76.1%), combined sensitivity beat margin-only by +68.7 points, but
> the false-alarm cost was +65.9 points against a ≤ 20-point bar, and
> the flag did not discriminate errors from correct answers among
> margin-unflagged items (Fisher p = 0.296; margin AUC 0.520, flag
> balanced accuracy 0.518). Mechanism: in a 47,499-state pool the label
> ranks run into the thousands, so the rank->500 leg fires constantly;
> a checklist calibrated on small hand-written sets degenerates toward
> flag-always at naturalistic scale. Consequence: exp-055/056's
> checklist signal does not generalise to naturalistic items at scale,
> and no threshold recalibration will be attempted on this data — any
> recalibrated flag needs a fresh predeclared test on new items.
> Full records: protocol, Amendments 1–2, pool-count decision,
> eligibility ruling, pilot report, confirmatory results with the
> complete 268-item margin-missed table.

## 4. What this rewording does not do

- It does not retrofit a threshold on the Path 2 data. That would be
  dredging, and the protocol forbids it.
- It does not publish exp-056 as a standalone claim. Exp-056's own
  record is borderline (H1 p ≈ 0.050, sensitivity 75%, specificity 67%,
  9 of 12 errors flagged with 4 false alarms) and carries two standing
  caveats: the 2026-10-06 correction (identical within-pair residual
  leans are vacuous by construction — the disambiguating adjective sits
  after the pronoun a causal model cannot see — so the narrow reading
  is a pre-clue structural prior, not semantic blindness) and the
  supplement-pool caveat (30 filler sentences, not the exact 24 test
  sequences). If 056 enters the repo, it enters with both caveats
  attached, as a pilot — not as support for the warning claim.
- It does not change any numbers in the frozen exp-052–055 folders.
  Append-only: this is new dated text, not an edit to old results.

## 5. Still needs Paul's word

1. **Post this rewording to the repo** (PR workflow; Paul merges).
   The two honest assets from last night — this negative result and the
   still-unpublished exp-201 correction (bridge 60/60, no-bridge 55/60,
   facts-once 29/60; detector verdict INCONCLUSIVE because the MISSING
   class was empty in every arm) — belong in the repo record together.
   The exp-320 correction and faithful-run rescore are also still not
   in the repo; the reviewer flagged that too.
2. **Licence decision — still open, no ruling given.** The live main
   README still conditions use on buying a copy of the book for every
   member of the development team (verified 2026-11-14 12:08 UTC).
   Francis's recommendation stands: standard open licence, book as the
   manual. Paul has not ruled on it.
3. **arXiv paper — still open.** One short paper written on Path 2's
   result whichever way it lands, bundled with the exp-201 correction.
   The result has now landed (NOT WIN); the paper decision is Paul's.

No step above happens without Paul's explicit word on that step.
