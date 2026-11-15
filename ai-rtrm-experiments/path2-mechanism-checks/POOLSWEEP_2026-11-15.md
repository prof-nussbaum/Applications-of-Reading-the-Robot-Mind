# Path 2 — EXPLORATORY post-hoc mechanism checks (2026-11-15)

**EXPLORATORY — not a confirmatory test.** These two checks were run after the
confirmatory result (NOT WIN) was known, to test the mechanism offered for it in
the paper: that in the 47,499-state pool the label ranks run into the thousands,
so the flag's rank>500 leg fires constantly. They do not re-open, soften, or
re-score the locked verdict, and no threshold is recalibrated anywhere in them.

## Check 1 — which legs of the exp-056 flag actually fire?

From `results_confirmatory.json` only (no new compute). Legs exactly as coded in
`run_confirmatory.py`: (a) competitor rank < label rank; (b) label rank > 500;
(c) label absent (encoded as null in the records). Reconstructing the flag as
a-or-b-or-c reproduces `rtrm_flag` on all 617 items.

| items | n | flagged | leg a marginal | leg a only | leg b marginal | leg b only | leg c marginal |
|---|---|---|---|---|---|---|---|
| all | 617 | 460 | 305 | 63 | 397 | 155 | 0 |
| model errors | 297 | 227 | 155 | 30 | 197 | 72 | 0 |
| model correct | 320 | 233 | 150 | 33 | 200 | 83 | 0 |

Legs (a) and (b) co-fire on 242 items (errors 125, correct 117). Leg (c) never fires:
no confirmatory item has an absent label — eligibility required >=10 pool
occurrences of each candidate word, so absence was designed out of this test.

Median label rank: **errors 1625**, **correct 1125.0**. Both sit far above the flag's
500 cut, which is why leg (b) fires on roughly two items in three in both groups
(errors 197/297, correct 200/320) — it carries almost no information about whether
the model was right.

## Check 2 — pool-size sweep (the mechanism test)

Each item's subtracted query state was recomputed with the confirmatory recipe
unchanged (PCA-64 fit on the full pool, full-pool pronoun means — nothing refit).
Only the candidate set was subsampled: uniform random rows of the full pool,
seeds 0/1/2. MISS is fixed from the confirmatory records (268 errors with
margin >= 0.35 nats); the margin baseline does not change with pool size.

**Sanity anchor: PASS.** Recomputed full-pool label/competitor ranks match
`results_confirmatory.json` exactly on all 617 items, and the full-pool row
below reproduces the confirmatory numbers (catch 76.1%, FAR 72.8%, p = 0.296).

| pool size | catch of MISS | false-alarm rate (correct) | Fisher p (flag x error, margin-unflagged) | leg (b) firing rate |
|---|---|---|---|---|
| 5000 (mean of seeds 0-2) | 77.5% | 77.1% | 0.497 | 48.0% |
| 15000 (mean of seeds 0-2) | 75.4% | 73.6% | 0.404 | 65.4% |
| 30000 (mean of seeds 0-2) | 75.6% | 72.6% | 0.342 | 64.1% |
| 47,499 (full pool) | 76.1% | 72.8% | 0.296 | 64.3% |

Per-seed values and Fisher tables: `poolsweep_2026-11-15.json`.

Reading (exploratory) — CORRECTED 2026-11-15, same day: the first draft of this
paragraph was written into the generating script before the sweep numbers
existed, and claimed the flag's false-alarm cost falls as the pool shrinks. The
numbers say it does not. The paragraph below replaces it; the correction is
recorded here rather than silently edited, per house rules.

What the sweep actually shows is narrower than the paper's mechanism claim.
The mechanical half holds: label ranks shrink with the candidate pool, and
leg (b) fires less at 5,000 (48.0% of items vs 64.3% at full size). But the
effect is already saturated by 15,000 (65.4%), and shrinking the pool does NOT
reduce the false-alarm rate — at 5,000 it is 77.1%, the worst row in the table —
nor restore discrimination: the Fisher p is worse at every smaller size
(0.497 / 0.404 / 0.342 vs 0.296 at full size), and the catch is flat
(75-78%) at every size. As leg (b) recedes, the remaining legs take over
(at 5,000 the flag still fires on ~74% of all items while leg (b) accounts
for only 48%). So pool size explains why leg (b) misfires, but it does not
explain the flag's failure to discriminate, and it offers no remedy: at no
pool size tested does the flag separate errors from correct answers.
