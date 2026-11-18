# exp-201 REVERSED probes — RESULTS (2026-11-15)

Protocol: `PROTOCOL.md` in this folder, locked 2026-11-15 before any scoring. Sequence followed exactly: leakage audit -> blind verdicts -> behaviour -> anatomy -> analysis. exp-201's files were not edited; everything here is new.

Blind verdicts saved (UTC): **2026-11-15T14:01:35.495933+00:00** — before any behavioural generation ran.

## PREDICTED (locked) vs ACTUAL

| Prediction | Actual |
|---|---|
| Behaviour collapses on reversed probes vs forward probes, most sharply in 201b/201c; 201a drops least | Collapse is total in ALL arms at both windows: 201a 60/60 -> 0/60; 201b 55/60 -> 0/60; 201c 29/60 -> 0/60 (8-token; 50-token identical: 0/60 everywhere). 201a did NOT drop least in absolute terms — it fell furthest (60 -> 0). |
| The MISSING class, empty in exp-201 (0/180), populates here | Yes, overwhelmingly: 153/180 items MISSING under hs[11]; in 201b and 201c it is 60/60. |
| Detector question open by design | Settled only in part — see verdict. The MISSING leg passes its bar; the STORED class never occurs, so the sorting claim cannot be confirmed, and with 0 passes overall the MISSING 'sort' is vacuous as discrimination: MISSING contains everything and everything failed. |

## Verdict (bars applied mechanically)

**INCONCLUSIVE**

- P(fail | MISSING), pooled 201b+201c: 120/120 = 1.00, Wilson 95% CI [0.969, 1.000] — bar >= 0.80 with >= 10 items: **PASS**.
- P(pass | STORED), pooled 201b+201c: class holds **0 items** — **INCONCLUSIVE (class < 10 items)**. No STORED verdict was issued for any item in any arm under hs[11] (or hs[12]).
- CONFIRMED requires both halves; REFUTED requires a filled class below bar. Neither holds: the filled class passes, the other class is empty. Verdict: **INCONCLUSIVE**.
- Base rates, pooled 201b+201c: 0/120 pass (rate 0.000). Because the base pass rate is 0, the one-sided binomial test of the MISSING fail rate against the base fail rate is degenerate (base fail rate = 1.0; no lift is possible) and Fisher exact on STORED vs MISSING is vacuous (table [[0, 0], [0, 120]], p = 1.0). Reported as required; neither number carries information here.

## Verdict counts per arm (reversed probes)

| Arm | hs[11] (primary) | hs[12] | Final-output ranks | Behaviour 8tok rev (fwd) | 50tok rev (fwd) |
|---|---|---|---|---|---|
| 201a (bridge) | MISSING 33, WEAK 27 | MISSING 55, WEAK 5 | MISSING 42, WEAK 18 | 0/60 (60/60) | 0/60 (60/60) |
| 201b (no-bridge) | MISSING 60 | MISSING 60 | MISSING 60 | 0/60 (55/60) | 0/60 (59/60) |
| 201c (facts-x1) | MISSING 60 | MISSING 60 | MISSING 60 | 0/60 (29/60) | 0/60 (30/60) |

## Class tables, pooled 201b+201c (n, of which pass at 8 tokens)

- Lens hs[11] (primary): MISSING: 120 items, 0 pass
- Lens hs[12]: MISSING: 120 items, 0 pass
- Final-output ranks (baseline): MISSING: 120 items, 0 pass

The output baseline does no better and no worse than the lens: on identical bands it too classes all 120 pooled items MISSING. On reversed probes there is no signal for either instrument to sort — behaviour is uniformly absent.

## Failure anatomy (protocol step 4)

STORED-but-failed items under hs[11]: **none — the STORED class is empty**, so the anatomy step has no items. (Every item is MISSING-but-failed; generations are recorded verbatim in the per-arm behaviour JSONs and sampled in SME_ITEMS.md.)

## Deviations

- None to the protocol's sequence, prompts, bands, or bars. One clarification, recorded in SME_ITEMS.md: the SME selection rule's 'worst behaviour' ties across all items (all fail), so the per-arm items were chosen by the stated tie-breaks (worst hs[11] rank; best hs[11] rank; first pair).
- Per-item records: `verdicts_reversed.json` (blind, timestamped), `behavior_mix_201*.json`, `analysis_reversed.json`; audit: `AUDIT_2026-11-15.md` (PASS, 0 leaks in 840 training texts).

## What this settles, in plain words

The reversal curse is real in these models in its true order-swapped form: trained on "G is the godparent of C", none of the three models — bridge arm included — can answer "C's godparent is" with G's name, 0 times in 180. The lens, read blind before behaviour, said MISSING for 153 of those 180 and STORED for none: it never cried wolf in the wrong direction here. But because nothing passed, this run cannot show the detector *sorting* successes from failures — the STORED half of the claim remains untested, exactly as it was after exp-201, now from the other side: forward probes filled STORED and emptied MISSING; reversed probes do the reverse.

---

## FLAG (2026-11-17, referee R-003 §3): the hs[12] column

The hs[12] lens ranks and verdicts in these records were computed
with `ln_f` applied twice (in transformers 5.17.0,
`hidden_states[12]` is already post-`ln_f`; the lens code applies
it again). They are not valid layer-12 measurements and no
conclusion should rest on them. The primary hs[11] verdicts are
unaffected. The "final output" ranks were computed from the
model's own output logits directly (`model(ids).logits` in
verdicts_reversed.py), not through the lens code, and are
unaffected. Confirmed by Francis in F-004.
