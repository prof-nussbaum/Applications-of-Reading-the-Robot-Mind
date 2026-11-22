# Mask-bug retrospective: protocol — LOCKED 2026-11-21

Drafted by Claude (referee, R-011 attachment, 2026-11-20). Corrected
against what is on disk by Francis and locked 2026-11-21. The bars and
outcome sentences are the referee's, unchanged. No mosaic is shown to
a reader before the referee's READY on the commit that carries this
file.

## Corrections against disk (Francis, 2026-11-21)

1. **Pixel source.** The rebuild pastes pixels from
   `recon-panel/augmented/` — the image set the pool hidden states
   were extracted from. The script's default (`MASKBUG_AUG_IMG`
   unset) points at `exp-320/data/augmented/`, a *different*
   augmented set (aug01–aug10 differ), and today's copy of
   `exp-320/src/mosaic.py` carries the same wrong default — its
   mtime is 35 minutes after the banked run, a packaging-era edit.
   All runs under this protocol set
   `MASKBUG_AUG_IMG=/home/ecpi-student-05/robot-mind/recon-panel/augmented`.
   The script itself is committed unmodified; the correction is the
   environment variable it already reads.
2. **Banked mosaics** live at `exp-320/figures/mosaic_recons/`
   (63 npz files, k = 300), not under recon-panel.
3. **Validation gate: RUN 2026-11-20, PASSED with correction 1.**
   First attempt (script defaults) reproduced none of the 63 banked
   `mean_mse` values (max |diff| 1116.7). Diagnosis: inputs were not
   the problem (sweep and pool files predate the banked run and are
   untouched); the aug00-in-pool hypothesis was tested and rejected
   (with aug00 in the pool, originals self-match at MSE 0.00, so the
   banked pool had none); the pixel source was the whole failure.
   With correction 1, the control row reproduces all 63 banked
   values: max |diff| 0.0001 (L8), 0.0001 (L16), 0.0002 (L24).
   Gate scripts: `gate_check3.py` (L8), `gate_check4.py` (all
   layers), committed beside this protocol; logs on the Jetson at
   `exp-320/maskbug_gate_check_2026-11-20/`.
4. The script **has** now been run on real H files — the control row
   of the gate, all 63 items. Its build/analyse/reader stages on the
   reference pool remain unrun; step 3 of the order of work is their
   first full run.
5. **Pixel source for the fixed pool: CHECKED 2026-11-22, PASSED**
   (check prescribed by the referee in R-015, PR #19). Correction 3's
   gate validated the pixel folder on the buggy pool only; every
   primary comparison is rebuilt from the fixed pool
   (`pool_layers_maskfix/`). Pool-only check (`poolcheck_r015.py`,
   committed beside this protocol; records in `poolcheck.json`): for
   each pool and layer, every tile state's nearest neighbour among
   the tiles of the *other* augmented copies of the same scene
   (the pool's own basis, k = 300, the retrospective's recipe), and
   the pixel error between the tile's own patch and the neighbour's
   patch, with patches cut from each candidate folder in turn.
   Mean pixel error, recon-panel/augmented vs exp-320/data/augmented:
   fixed pool — L8 380.2 vs 1444.9, L16 407.1 vs 1423.3, L24 378.7
   vs 1451.3; buggy pool (reference) — L8 325.7 vs 1441.4, L16 334.5
   vs 1419.1, L24 328.8 vs 1428.7. The folder named in correction 1
   gives the clearly lower error on the fixed pool at every layer,
   of the same order as on the buggy pool — correction 1 therefore
   stands for both pools, and the script keeps its single picture
   folder. (The absolute values differ from the 2026-11-07 hygiene
   note's 140 vs 552 because the construction differs — pool-internal
   other-copy neighbours here, scene mosaics there; the ordering and
   the roughly fourfold ratio are the signature.) Consistent with
   the exp-320 README's note that the maskfix re-run read latents
   from `recon-panel/augmented_latents/`.

## Question

Fixing the mask bug took the closed-loop pilot from 0 of 10 to 10 of 10.
From the banked hidden states alone, would an RTRM reconstruction have shown
that something was wrong before the robot moved?

## Design

Every banked scene state (7 scenes x 3 prompts x 3 layers = 63 items per
condition) is rebuilt tile by tile from a **reference pool of known-good
states**, by `mosaic.py`'s method at its setting of record (PCA residual,
k = 300, cosine nearest neighbour).

| Condition | Scene states | Role |
|---|---|---|
| buggy | `layer_sweep/` | mask bug present |
| maskfix | `layer_sweep_maskfix/` | same inputs and prompts, mask fixed: **the contrast** |
| faithful | `layer_sweep_faithful/` | a second working configuration: how far does a benign change move the scores? |

Reference pool: `pool_layers_maskfix/`. Control: buggy scenes on the buggy
pool (`pool_layers/`).

**Assumption stated up front.** The method compares a run with a reference
taken when the system worked. The original figures (buggy states on the
buggy pool) looked like good reconstructions, so RTRM cannot detect a fault
that is also in its reference. The control row measures exactly that.

## Scores, per item

RTRM scores (need pool and pictures): `mse` (pixel error, camera frame against
mosaic; the number `mosaic.py` already reports), `same_scene`, `same_place`.

Cheap comparators (need no pictures): `top1_cos`, `h_norm`, `mean_shift`.

## Order of work

1. **Validation gate.** DONE 2026-11-20 — see correction 3. Passed.
2. **Lock** this protocol (this commit), with `scores.json` **withheld**.
3. `analyse`, commit the output.
4. `reader --seed N` builds 189 shuffled items (camera frame beside mosaic).
   Commit the set and the key's SHA-256 only. Referee READY. Then Paul reads.
5. Commit judgments, then the key, then the scoring.

## Bars (locked as drafted)

**Automatic, primary: buggy against maskfix, reference pool, all layers, 63 pairs.**

- **A1, detects.** `mse` separates the sets with AUROC at or above 0.85 (or at
  or below 0.15) and moves the same way in at least 52 of 63 pairs.
- **A2, specific.** On faithful against maskfix the same score stays between
  0.35 and 0.65. If it does not, the finding is "detects a change from the
  reference", not "detects the bug".
- **A3, against the cheap comparators.** RTRM adds detection value only if no
  cheap comparator separates buggy from maskfix at least as well while also
  passing A2. Otherwise the supportable claim is legibility, and the reader
  test decides it.

`same_scene` and `same_place` are secondary and reported in full.

**Reader: 189 single items, each judged "looks right" or "looks wrong".**
Broken = buggy; working = maskfix and faithful.

- **R1, legible.** Balanced accuracy at or above 0.80.
- **R2, better than a statistic.** At or above the best single-cut balanced
  accuracy any cheap comparator reaches on the same 189 items. (That cut is
  chosen with the answers in view, which favours the comparator.)

## How each outcome will be written

| Result | Sentence the note may carry |
|---|---|
| A1, A2, A3 and R1 | An RTRM reconstruction against a known-good reference showed the bug, a reader could see it, and simple statistics did not do as well. |
| A1, A2, R1, not A3 or not R2 | The bug is visible in the reconstruction to a reader with no statistics set up; a simple statistic detects it equally well. |
| A1, not A2 | The reconstruction detects a departure from the reference; it does not tell this bug from a benign change. |
| Not A1 | The mask bug is not visible in these reconstructions. The 0-of-10 failure left the picture the model holds intact. |

The last row is a finding too: it would say the bug damaged what the policy
does with the scene, not what it holds of the scene.

## Disclosures to carry into any write-up

- **Seven scenes.** The three prompts of a scene share one image, so the 63
  items are not independent. Report per scene as well as pooled.
- **What was run closed-loop.** The 0-of-10 to 10-of-10 result is the mask fix
  in the pilot protocol. The buggy and maskfix sets here both use a zero body
  vector; faithful is the set closest to the working pilot.
- **Reader familiarity.** Paul has seen the earlier mosaics of these scenes
  (buggy states on the buggy pool). No mosaic in this design has been built
  or seen by anyone.
- **k.** k = 300 is primary because it is the script of record. `MASKBUG_K=0`
  into a second output folder is reported as secondary.
- All outputs go to a fresh folder; the script refuses to overwrite and never
  writes into a banked folder.
