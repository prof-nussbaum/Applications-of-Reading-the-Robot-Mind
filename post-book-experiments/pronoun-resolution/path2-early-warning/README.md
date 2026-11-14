# Path 2 — early warning beyond output confidence (2026-11-13/14)

## What this was

The pronoun-resolution study in this repo (exp-052 to exp-055) ended
with a small pilot: on 10 new prompts, a checklist built on the RTRM
cosine-similarity method flagged every forced-choice error the model
made, with 3 false alarms. An external review asked the obvious
question: does that checklist still work at scale, on natural items,
against the simplest baseline — the model's own output margin?

Path 2 was that test, predeclared before any scoring. GPT-2 small,
frozen. The method of record was reused exactly: the 054c-C2
subtracted neighbour list at layer 12, PCA-64 (seed 0), pronoun-mean
subtraction, and exp-056's mechanical flag rule (a competitor ranked
above the intended referent, or the referent ranked below 500, or
absent). The baseline was the forced-choice margin: flag any item
where the model's two candidate scores sit within 0.35 nats. The
items were 617 pronoun-resolution problems from the DPR set
(Rahman & Ng 2012), test and train splits, after a formal audit and
two predeclared amendments. The 60 pilot items were scored first,
checked, and permanently excluded from the confirmatory set.

## Verdict: NOT WIN

The flag caught most of the errors the margin missed — by flagging
almost everything. It fired on 72.8% of the items the model got
right. A warning that fires on three of every four correct answers
is not a warning.

### The four locked bars

| Bar | Requirement | Actual | Result |
|---|---|---|---|
| (i) Catch of margin-missed errors | ≥ 30% | 76.1% (204 of 268) | PASS |
| (ii) Combined sensitivity gain over margin-only | ≥ +10 points | +68.7 points (9.8% → 78.5%) | PASS |
| (iii) False-alarm cost on correct items | ≤ +20 points | +65.9 points (10.6% → 76.6%) | FAIL |
| (iv) Flag discriminates errors from correct answers (margin-unflagged items) | Fisher p < 0.05 | p = 0.296 | FAIL |

A win required all four bars. Two failed, so the verdict is NOT WIN.

One honest complication: the protocol defined its REFUTED label
narrowly, as bars (i) or (ii) failing, and neither did — so the
formal REFUTED label was not triggered either. That was a drafting
gap in the protocol, and we are not hiding behind it. In plain
words, the claim this test was built to check — an RTRM early
warning that beats output confidence at a bounded false-alarm
cost — did not hold. The same shape appeared in every split
(test, train) and both clue classes.

### PREDICTED vs ACTUAL

PREDICTED (locked before data): GPT-2 would err on 25–45% of the
items; the RTRM flag would catch at least 30% of the errors the
margin missed; combined sensitivity would beat margin-only by at
least 10 points at no more than 20 points of extra false alarms.

ACTUAL: the model erred on 48.1% (297 of 617) — just above the
predicted band. Errors the margin missed: 268. The flag caught
204 of them (76.1%), and combined sensitivity rose by 68.7 points,
far past the bar. But the false-alarm cost was +65.9 points, more
than three times the allowed cost, and among the items the margin
left unflagged the flag did not separate errors from correct
answers (Fisher p = 0.296). Threshold-free measures say the same:
margin AUC 0.520, flag balanced accuracy 0.518 — both at chance.

Why it failed is visible in the records: in the 47,499-state pool
built for this test, the intended referent's rank in the neighbour
list runs into the thousands, so the flag's "rank below 500" leg
fires constantly. The checklist was calibrated on small
hand-written sets (10 prompts in exp-055, 24 in exp-056). At
natural scale it degenerates toward flag-always.

### What this settles, and what it does not

It settles that the exp-055/056 checklist signal does not
generalise to naturalistic items at scale, in this form. The
parent README's claims have been bounded accordingly (see the
"Update, 2026-11-14" section there).

It does not touch the method finding underneath the checklist:
subtracting the predictable part of a pronoun's state makes noun
candidates visible in the neighbour list. That finding stands on
the exp-052 to exp-055 record.

No threshold rescue will be attempted on this data — tuning the
flag until it passes on the same 617 items would be dredging.
Any recalibrated flag would need a fresh predeclared test on
new items.

## Files

- [path2_early_warning_PROTOCOL_2026-11-13.md](./path2_early_warning_PROTOCOL_2026-11-13.md) — the predeclaration: hypothesis, method, bars, verdict rules, written before any data.
- [path2_early_warning_AMENDMENT1_2026-11-13.md](./path2_early_warning_AMENDMENT1_2026-11-13.md) — pool rebuilt from natural WikiText-103 text; both DPR splits as strata; declared before scoring.
- [path2_early_warning_AMENDMENT2_2026-11-13.md](./path2_early_warning_AMENDMENT2_2026-11-13.md) — state build capped at 45,000 tokens, whole sentences only.
- [path2_poolcount_DECISION_2026-11-14.md](./path2_poolcount_DECISION_2026-11-14.md) — ruling: the built pool's token-label count governs eligibility (114 items fell to the predeclared exclusion rule).
- [path2_eligibility_RULING_2026-11-14.md](./path2_eligibility_RULING_2026-11-14.md) — ruling confirming the locked confirmatory set of 677 before the pilot draw.
- [RESULTS_confirmatory.md](./RESULTS_confirmatory.md) — the full confirmatory results: PREDICTED vs ACTUAL, all four bars, every split and clue class, and the complete 268-item margin-missed table.
- [REWORDING_DRAFT_2026-11-14.md](./REWORDING_DRAFT_2026-11-14.md) — the bounded rewording of the exp-055/056 claims that this result required, with its sources.

The run scripts, the pilot records, and the per-item confirmatory
records live in the lab working folder on the Jetson
(`~/robot-mind/pronoun-battery-v2/path2-early-warning/`). The state
pool (`.pt`) is not included; it regenerates from the protocol's
build rules.
