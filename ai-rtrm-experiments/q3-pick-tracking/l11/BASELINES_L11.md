# Q3 Amendment 1 — Stage 1: L11 pool + pronoun-position baselines

Run 2026-11-16, Jetson, per `AMENDMENT_1.md` §"The layer-11 pool —
decided now", steps 1–2. **Stage 1 only: no answer-position state was
extracted and nothing at the answer position was scored.** The L11
bars are locked in a supplement commit from these baselines (same +10
rule) before stage 2 exists.

## Validation — the adapted build is the banked build

`build_pool_l11.py` is an adapted copy of the banked
`build_pool_path2.py` (original untouched): same
`pool_build_selection.jsonl` in the same order, same tokenisation,
same per-token unit-normalisation, fp32, GPT-2 on CPU; one forward
pass recording layers 11 and 12. Gate A (`validate_pool_build.py`),
against the banked `pool_path2.pt`:

- Record count 1,756 = banked; record ids identical, same order;
  token ids identical in every record; total states 47,499 = banked.
- **L12 states bit-exact: max abs diff 0.000e+00 over all 1,756
  records** (banked hs[2] vs rebuild hs[0]).
- Raw L12 norm stats identical to the banked `build_stats.json`
  (mean 217.877615, min 23.228270, max 438.479462). Build time 529.5 s
  (banked: 519.6 s).

Gate B (`validate_machinery_l12.py`, the earlier Q3 validation
pattern pointed at the rebuilt L12 pool): the confirmatory recipe
WITH pronoun-mean subtraction reproduces the banked
`results_confirmatory.json` ranks on all **12/12 probed items
exactly** — MACHINERY: VALIDATED. Only after both gates passed did
the same code path run at layer 11.

Scoring self-checks (`score_baselines_l11.py`, run before any L11
number): the strata/pick machinery fed the BANKED L12 pronoun leans
reproduces every published pronoun baseline exactly — vs substitution
pick: 105/183 same-pick both, 0/54 flip both (54/54 exactly-one),
90/143 non-twin; vs question pick: 116/217 same-Q both, 0/20 flip-Q,
73/143 non-twin; strata picks/leans disagree with banked records on
0 items. SELF-CHECKS: ALL OK.

## The L11 pronoun baselines (list, original recipe, WITH subtraction)

Extraction: hs[11] at the pronoun token of the exact confirmatory
prompt, L11 pool (`pool_L11.pt`, 47,499 states), PCA-64 seed 0
(top-64 variance 0.5026 at L11, vs 0.9401 at L12), pronoun means from
214/204/204/200 pool occurrences (he/she/it/they). Per-item ranks and
leans: `pronoun_l11_ranks.json` (617 records).

**Against the question-format pick (primary target under Amendment 1):**

| Stratum | L11 pronoun baseline | L12 pronoun baseline (banked) | +10-rule bar (for the supplement lock) |
|---|---|---|---|
| Same-Q pairs (217) | **108/217 = 49.8%** (both-or-neither: 108 both, 109 neither, 0 exactly-one) | 116/217 = 53.5% | 59.8% |
| Flip-Q pairs (20) | **0/20 = 0.0%** (20/20 exactly-one — forced at the pronoun: one shared state, one lean, two different picks) | 0/20 (forced) | no bar (Amendment 1 pattern: report with exact binomial interval in stage 2) |
| Non-twin items (143) | **66/143 = 46.2%** | 73/143 = 51.0% | 56.2% |

**Against the substitution pick (for the record):**

| Stratum | L11 pronoun baseline | L12 pronoun baseline (banked) |
|---|---|---|
| Same-pick pairs (183) | 88/183 = 48.1% (88 both, 95 neither) | 105/183 = 57.4% |
| Flip pairs (54) | 0/54 = 0.0% (54/54 exactly-one, forced) | 0/54 (forced) |
| Non-twin items (143) | 65/143 = 45.5% | 90/143 = 62.9% |

## Notes

- At the pronoun position the L11 list tracks the picks slightly
  *worse* than the L12 list on every stratum. Nothing in stage 1
  speaks to the answer position; the lens reads the pick best at L11
  (Amendment 1), but the L11 *list* starts from a lower pronoun
  baseline, so its +10 bars (59.8% / 56.2%) sit lower than L12's
  (63.5% / 61.0%) in absolute terms.
- The both-or-neither pattern at the pronoun is structural, not a
  result: twins share one pronoun state, hence one lean, so same-pick
  pairs match on both twins or neither, and flip pairs on exactly one.

## Files

- `build_pool_l11.py` — adapted pool build (layers 11+12, one pass)
- `pool_L11.pt` — the L11 pool (146,624,669 bytes; records
  {"id","ids","hs":[L11]})
- `pool_L12_rebuild.pt` — validation slice only (same format, L12)
- `validate_pool_build.py` — gate A; `validate_machinery_l12.py` — gate B
- `extract_pronoun_l11.py` — pronoun-position L11 extraction/readout
- `pronoun_l11_ranks.json` — per-item L11 pronoun ranks + lean (617)
- `score_baselines_l11.py` — self-checks + baselines
- `baselines_l11.json` — all baseline numbers, machine-readable
- `build_stats_l11.json` — build stats; `stage1.log`, `monitor.log`,
  `monitor_status.log` — run record (driver DONE 2026-11-16 17:49 EST;
  one restart of the monitor script itself at 17:44 is on the record
  in monitor_status.log — the driver and its own monitor.log were
  unaffected)
