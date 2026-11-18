# Q3 Amendment 1 — Supplement: the L11 bars, locked (2026-11-16)

Stage 1 of the layer-11 work is complete (Jetson
`~/robot-mind/q3-pick-tracking/l11/`, full account in
`BASELINES_L11.md`). This supplement locks the L11 bars **before**
any answer-position L11 state is extracted or scored, per the staging
in AMENDMENT_1.md.

## Validation (both gates passed before any L11 readout)

- **Build fidelity:** the adapted pool build, run over the same
  selection file, reproduces the banked pool's L12 states
  **bit-exactly** (max abs diff 0.000e+00 over all 1,756 records;
  47,499 states; ids and token ids identical in order).
- **Machinery:** the confirmatory recipe (with pronoun-mean
  subtraction) against the rebuilt L12 pool reproduced the banked
  confirmatory ranks on 12/12 probed items exactly.
- **Scoring self-checks:** every published pronoun baseline was
  reproduced from the banked L12 leans (105/183, 0/54, 90/143 vs
  substitution; 116/217, 0/20, 73/143 vs question pick) before any
  L11 number was computed.

## L11 pronoun baselines (list, pronoun position, original recipe)

Against the question-format pick (the amended primary target):

| Stratum | L11 baseline | L12 baseline (for comparison) | **L11 bar (+10)** |
|---|---|---|---|
| Same-Q pairs (217) | 108/217 = 49.8% | 116/217 = 53.5% | **≥59.8%** |
| Non-twin items (143) | 66/143 = 46.2% | 73/143 = 51.0% | **≥56.2%** |
| Flip-Q pairs (20) | 0/20 (forced) | 0/20 (forced) | no bar — exact interval |

Against the substitution pick, for the record: same-pick 88/183 =
48.1%, flip 0/54 (forced), non-twins 65/143 = 45.5%.

The L11 list's pronoun baselines sit **below** L12's on every
stratum, so its bars sit lower in absolute terms; the +10 rule is
applied to its own baselines, unchanged. (Noted in BASELINES_L11.md:
PCA top-64 variance at L11 is 0.5026 against 0.9401 at L12 — the
residual construction behaves differently at this layer, which is
part of what stage 2 measures.)

## Stage 2, as locked by this supplement

Extract answer-position hs[11] states for the 617 items; compute the
neighbour list at L11 against the L11 pool with the Q3 recipe (no
pronoun-mean subtraction at the answer position; PCA-64, seed 0);
score pair-level against the question-format pick on the amended
strata. Primary: same-Q pairs vs the 59.8% bar. Secondary: non-twins
vs 56.2%; flip-Q with an exact interval; substitution-target figures
reported with no bar. The L11 pool file (146 MB) stays on the Jetson
at `~/robot-mind/q3-pick-tracking/l11/pool_L11.pt`, like the banked
L12 pool; its build code and stats are in this folder's records.
