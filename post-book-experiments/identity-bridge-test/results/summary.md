# exp-200p results summary

Recreation of the Identity Bridge recipe (Ma et al., arXiv:2602.02470) on the
synthetic 200j domain. GPT-2 small, SGD lr 5e-4, no momentum, batch size 1,
200 epochs, seed 0. Probed 2026-10-13 with the corrected RTRM lens
(`output_hidden_states`, layers 5/7/9/11).

## Training

| Model | Training texts | Epochs | Final mean loss |
|---|---|---|---|
| mix_e3 (bridge) | 420 (60 fwd x6 + 60 bridges) | 200 | 0.2712 |
| mix_e4 (no-bridge baseline) | 360 (60 fwd x6) | 200 | 0.2563 |

Neither model saw any reverse facts in training.

## Reverse-probe results (60 prompts)

| Metric | mix_e3 (bridge) | mix_e4 (no bridge) |
|---|---|---|
| Behaviorally correct | 0/60 | 0/60 |
| L5 median rank (range) | 1,351 (84-21,866) | 4,727 (255-28,467) |
| L7 median rank (range) | 299 (1-3,338) | 3,617 (236-19,857) |
| L9 median rank (range) | 392 (2-2,437) | 1,660 (62-10,508) |
| L11 median rank (range) | 118 (5-321) | 1,971 (11-23,358) |
| L11: STORED (rank <= 10) | 1 | 0 |
| L11: WEAK (rank 11-100) | 26 | 5 |
| L11: MISSING (rank > 100) | 33 | 55 |

## Reading

PREDICTED: the bridge lifts reverse accuracy and shifts the trajectory toward
the learned shape. ACTUAL: 0/60 behaviorally for both -- but the bridge moved
the L11 median rank by more than an order of magnitude (118 vs 1,971) and
shifted 27 items into STORED/WEAK vs 5. The bridge changed the internal
representation substantially without crossing the behavioral threshold.

Caveats: single seed (0); GPT-2 small vs the paper's 1B model; our bridges
were rephrasings, not true two-hop inverse compositions (the synthetic domain
has no inverse relations). Bounds the paper's claim on our turf; does not
refute it.
