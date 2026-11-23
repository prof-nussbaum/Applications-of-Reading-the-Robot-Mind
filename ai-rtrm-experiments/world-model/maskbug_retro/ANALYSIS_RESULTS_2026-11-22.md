# Mask-bug retrospective — automatic results (2026-11-22)

Run under Fable's R-016 READY (`READY: build and analyse per PROTOCOL_MASKBUG.md at a25eee43a64db42771bc2af4c35c0b984802eb85`, 2026-11-23 00:01:55 UTC). Protocol: `ai-rtrm-experiments/world-model/PROTOCOL_MASKBUG.md` at a25eee4. Script `maskbug_retro.py` (repo copy, md5 e4e4daad1b1214dee277808ad8299bd0), run with `MASKBUG_AUG_IMG=/home/ecpi-student-05/robot-mind/recon-panel/augmented` (correction 1). Build wrote 252 items (`maskbug_retro/scores.json`); `analyse` output is reproduced verbatim below.

## Verdict against the locked bars

- **A1 (detects) — PASSES, perfectly.** mse, buggy vs maskfix on the reference pool, all layers: **AUROC 1.000**, same direction in **63/63** pairs (bar: AUROC ≥ 0.85 and ≥ 52/63). Per layer: 21/21 at L8, L16 and L24.
- **A2 (specific) — FAILS.** mse, faithful vs maskfix: **AUROC 0.975** (bar: 0.35–0.65). The benign change separates almost as well as the bug.
- Per-scene: mse is worse than maskfix in **9/9** pairs in every scene for buggy — and also **9/9** in every scene for faithful. The maskfix set has the lowest mse of the three everywhere: the reconstruction detects departure from the reference in either direction, not the bug specifically.
- **A3 (cheap comparators):** on the primary comparison the cheap scores separate as well as mse — h_norm AUROC 0.999 (63/63), mean_shift 1.000 (63/63), top1_cos 0.000 (63/63, opposite direction), same_place 0.000. And h_norm also stays inside A2's band on the benign change (0.482). A simple statistic does as well as the reconstruction.
- **Control (buggy scenes on the buggy pool vs maskfix on the reference pool):** mse AUROC 0.812, 54/63 pairs worse — weaker separation than the primary (1.000, 63/63), as the design expects: against its own buggy reference the bug is much less visible.

## Exploratory, after the fact (added 2026-11-22 at the referee's request, R-018 §2 — no verdict changes)

A2 asked whether the benign change is invisible to mse. It is not, and the A2 verdict above stands. A different question — whether the bug can be told from the benign change — is answered by the same records (all figures re-derived from `scores.json`):

| Mean over 63 items, reference pool | mse | tiles from the right place (same_place) |
|---|---|---|
| maskfix | 166 | 92% |
| faithful (benign change) | 1,292 | 70% |
| buggy | 6,490 | 0.3% |
| control: buggy on the buggy pool | 574 | 84% |

- Buggy against faithful: mse AUROC **0.985**.
- Buggy against both working sets together (the reader's contrast): one cut on mse scores **0.992** (balanced accuracy).
- The bug's effect is about five times the benign change's, and a threshold separates it from both working sets.
- With the bug, **0.3% of tiles come from the right place**: against the good pool the picture is not degraded, it is scrambled. Yet against its own pool the buggy run rebuilds 84% in place. The scene is still in the buggy states; they have moved somewhere the good pool does not cover. A reader shown the scrambled mosaic could wrongly conclude "the model cannot see the scene" — this note says plainly that it can.
- Detection here is close to a mean-shift test. That is why the cheap statistics tie it.

## Outcome row (locked table)

**A1, not A2:** *"The reconstruction detects a departure from the reference; it does not tell this bug from a benign change."*

This is not the last outcome-table row ("Not A1"), so R-016's talk-first condition for that row does not apply. The reader test still decides the legibility claim (R1/R2); the reader set is built only on Fable's second READY, per R-016.

## Full `analyse` output (verbatim)

```
=== PRIMARY: buggy vs maskfix, reference pool, all layers ===
score         AUROC |AUROC-.5|  best cut  pairs worse
mse           1.000      0.500     1.000    63/63 (higher)
same_scene    0.136      0.364     0.849    56/63 (lower)
same_place    0.000      0.500     1.000    63/63 (lower)
top1_cos      0.000      0.500     1.000    63/63 (lower)
h_norm        0.999      0.499     0.992    63/63 (higher)
mean_shift    1.000      0.500     1.000    63/63 (higher)

=== buggy vs maskfix, layer 8 ===
score         AUROC |AUROC-.5|  best cut  pairs worse
mse           1.000      0.500     1.000    21/21 (higher)
same_scene    0.135      0.365     0.833    20/21 (lower)
same_place    0.000      0.500     1.000    21/21 (lower)
top1_cos      0.000      0.500     1.000    21/21 (lower)
h_norm        1.000      0.500     1.000    21/21 (higher)
mean_shift    1.000      0.500     1.000    21/21 (higher)

=== buggy vs maskfix, layer 16 ===
score         AUROC |AUROC-.5|  best cut  pairs worse
mse           1.000      0.500     1.000    21/21 (higher)
same_scene    0.145      0.355     0.857    18/21 (lower)
same_place    0.000      0.500     1.000    21/21 (lower)
top1_cos      0.000      0.500     1.000    21/21 (lower)
h_norm        1.000      0.500     1.000    21/21 (higher)
mean_shift    1.000      0.500     1.000    21/21 (higher)

=== buggy vs maskfix, layer 24 ===
score         AUROC |AUROC-.5|  best cut  pairs worse
mse           1.000      0.500     1.000    21/21 (higher)
same_scene    0.127      0.373     0.881    18/21 (lower)
same_place    0.000      0.500     1.000    21/21 (lower)
top1_cos      0.000      0.500     1.000    21/21 (lower)
h_norm        1.000      0.500     1.000    21/21 (higher)
mean_shift    1.000      0.500     1.000    21/21 (higher)

=== BENIGN CHANGE: faithful vs maskfix (both working), all layers ===
score         AUROC |AUROC-.5|  best cut  pairs worse
mse           0.975      0.475     0.929    63/63 (higher)
same_scene    0.388      0.112     0.651    48/63 (lower)
same_place    0.061      0.439     0.881    63/63 (lower)
top1_cos      0.013      0.487     0.952    63/63 (lower)
h_norm        0.482      0.018     0.651    32/63 (lower)
mean_shift    0.832      0.332     0.825    63/63 (higher)

=== buggy vs faithful, all layers ===
score         AUROC |AUROC-.5|  best cut  pairs worse
mse           0.985      0.485     0.984    62/63 (higher)
same_scene    0.169      0.331     0.857    52/63 (lower)
same_place    0.000      0.500     1.000    63/63 (lower)
top1_cos      0.000      0.500     1.000    63/63 (lower)
h_norm        0.932      0.432     0.865    62/63 (higher)
mean_shift    1.000      0.500     1.000    63/63 (higher)

=== CONTROL: buggy scenes on the BUGGY pool vs maskfix on the reference pool ===
score         AUROC |AUROC-.5|  best cut  pairs worse
mse           0.812      0.312     0.778    54/63 (higher)
same_scene    0.308      0.192     0.802    46/63 (lower)
same_place    0.484      0.016     0.603    36/63 (lower)
top1_cos      0.854      0.354     0.849    57/63 (higher)
h_norm        0.999      0.499     0.992    63/63 (higher)
mean_shift    0.984      0.484     0.929    63/63 (higher)

AUROC near 1 or 0 = the score separates the two sets; near 0.5 = it does not.
"best cut" is the most any single threshold could score on these items.
```
