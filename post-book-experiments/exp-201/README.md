# exp-201 — the bridge test redone properly (godparent domain)

## Why this folder exists

The identity-bridge study published in this repository
(`../identity-bridge/`, exp-200p/200q) drew a fair criticism in
external review: in its mentor-domain showcase, the "bridge"
sentence nearly restates the tested fact, and the probe direction
never truly reverses. The review asked for the bridge study to be
redone properly — a fresh domain, true inverse bridges, and probe
wording that appears nowhere in training.

This folder is that redo. It was run on 2026-10-23, before the
review arrived, and is published now as the correction it is.
Nothing in the earlier study's folders has been edited; this is a
new folder beside them.

## What was done

A new relation domain: godparent/godchild, with new invented names.
Training states the forward fact ("X is the godparent of Y"); the
bridge sentences state the true inverse ("The godchild of X is Y");
the probe asks in the inverse direction ("X's godchild is"). The
probe prompt phrasing appears in no training text — verified for
all 60 items in all arms by the sample-data audit
(`audit_2026-10-23.md`, seed 1431).

Three arms, same recipe as exp-200q (GPT-2, SGD lr 5e-4, 200 epochs,
seed 0):

- **201a, bridge:** forward facts ×6 plus bridge sentences.
- **201b, no-bridge:** forward facts ×6 only.
- **201c, facts-once:** forward facts ×1 only.

The detector under test is the published instrument, unchanged:
the L11 logit-lens verdict (`rtrm_probe.py`, bands STORED ≤ 10,
WEAK 11–100, MISSING > 100), computed per item, blind — each arm's
verdict files were saved and timestamped before that arm's
behavioural results were examined. Behaviour: greedy generation
from the probe prompt; pass if the generation contains the full
expected godchild name.

The hypothesis, metric, and decision bar were locked in writing
before any training, probing, or behavioural testing
(`HYPOTHESIS_201.txt`).

## PREDICTED vs ACTUAL

PREDICTED: items the lens calls MISSING fail the behavioural test
at a rate ≥ 80%; items it calls STORED pass at a rate ≥ 80%
(pooled over arms 201b + 201c; WEAK items excluded; each verdict
class needs at least 10 items, or that half of the claim is
declared INCONCLUSIVE rather than dropped).

ACTUAL — the behaviour replicated the earlier study almost
exactly:

| Arm | Behavioural pass (8-token) | exp-200q counterpart |
|---|---|---|
| 201a bridge | 60/60 | q3: 60/60 |
| 201b no-bridge | 55/60 | q4: 53/60 |
| 201c facts-once | 29/60 | q5: 19/60 |

ACTUAL — the detector verdict is **INCONCLUSIVE**, by the locked
bar:

- **P(fail | MISSING) is undefined: 0/0.** The MISSING class was
  empty in all three arms. Across 180 items the published probe
  never once verdicted MISSING — not even in arm 201c, where 31
  of 60 items failed behaviourally. The half of the claim that
  mattered most — does the instrument flag the failures — could
  not be evaluated, because the instrument did not flag anything.
- **P(pass | STORED) = 58/72 = 80.6%** pooled over 201b + 201c,
  which clears its 80% bar (per arm: 201b 42/46 = 91.3%,
  201c 16/26 = 61.5%). Fourteen STORED-but-failed disagreement
  items are listed in full in `RESULTS_201.txt`.

## The honest read

The behaviour is real and it replicates: bridges teach the
inverse direction, facts seen once mostly do not. The detector
claim does not survive contact with new data in its published
form: on a domain it played no part in developing, the instrument
under-flags — it calls nearly everything STORED or WEAK, so it
cannot sort the items that will fail from the items that will
pass. That is a calibration finding about the published
thresholds, stated here at the same prominence a confirmation
would have received.

Full record: `RESULTS_201.txt`.

## Files

- `HYPOTHESIS_201.txt` — hypothesis, instrument, metric, and
  decision bar, locked 2026-10-23 before any run. Unedited.
- `RESULTS_201.txt` — the results as recorded the day of the run,
  including all 14 disagreement items. Unedited.
- `audit_2026-10-23.md`, `audit_201.py` — the sample-data audit
  (PASS, seed 1431).
- `build_201.py`, `finetune_201.py`, `behavior_201.py`,
  `tabulate_201.py`, `render_visuals_201.py`, `make_listing.py` —
  the run scripts.
- `dataset_201.json`, `pairs_201.json`, `inputs_201_listing.txt` —
  the exact items and training texts.
- `probe_mix_201a/b/c_results.json` — the blind per-item lens
  verdicts, with their run logs (`probe_run_mix_201*.log`).
- `behavior_mix_201a/b/c.json`, `tabulation_201.json`,
  `train_summary_201.txt` — behavioural records and tabulation.
- `results_table_mix_201a/b/c.png` — per-arm results tables.
- `showcase_*.png` — three worked items from arm 201b: a
  stored-and-passed item, a disagreement (STORED, rank 5, but the
  model answered with the wrong name), and a WEAK item.

Not included: the fine-tuned model weights (~1.4 GB on the lab
machine) and the training logs; the scripts and
`train_summary_201.txt` record the recipe exactly.
