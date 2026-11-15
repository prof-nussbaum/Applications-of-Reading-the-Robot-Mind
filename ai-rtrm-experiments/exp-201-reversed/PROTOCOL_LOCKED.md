# exp-201 REVERSED probes — PROTOCOL (locked 2026-11-15, before any scoring)

Paul's delegation, 2026-11-15: "You pick. You go." Francis picked this run
first (cheapest decisive test; saved models; no retraining), positive
control second. This protocol is locked before the leakage audit runs.

## Why

exp-201's probes kept the training name order: fact "G is the godparent
of C", probe "G's godchild is" -> C. What reversed was the relation noun,
not the entity order. Berglund et al.'s reversal test swaps the order.
This run applies TRUE order-reversed probes to the same saved models.

## Items and prompts (verbatim)

All 60 godparent pairs from dataset_201.json, all three saved arms
(201a bridge, 201b no-bridge, 201c facts-x1).

- Reversed probe: "{godchild}'s godparent is" -> expects " {godparent}"
  Example: "Andreas Ravensworth's godparent is" -> " Zelda Yarnton"
- Expected first token: first GPT-2 token of " {godparent full name}".

## Sequence (order is part of the protocol)

1. LEAKAGE AUDIT, dated, before scoring: assert every reversed-probe
   string occurs 0 times in all training texts of all three mixes;
   assert the possessive template "{name}'s godparent is" is absent.
   If any probe leaks, stop and report — do not score.
2. BLIND VERDICTS, saved and timestamped before any behavioural scoring:
   - Primary instrument: published probe (rtrm_probe.py), hs[11]
     convention, bands STORED <=10 / WEAK 11-100 / MISSING >100, rank of
     expected first token at the probe's last position.
   - Secondary readout (no bar attached): same rank at hs[12].
   - Output baseline: rank of the expected first token in the final-layer
     next-token distribution at the same position.
3. BEHAVIOUR: greedy generation, 8 tokens, pass = generation contains the
   full godparent name (primary, identical to exp-201); 50-token variant
   recorded as secondary.
4. FAILURE ANATOMY: for STORED-but-failed items, record whether the
   expected first token was the argmax / in the top 5 of the final
   distribution, and the full generation verbatim.

## Decision bars (locked; pooled over 201b + 201c, as in exp-201)

- CONFIRMED iff P(fail | MISSING) >= 0.80 AND P(pass | STORED) >= 0.80,
  each class holding >= 10 items.
- A class with < 10 items makes that half INCONCLUSIVE (no verdict on it).
- Classes filled and a rate below bar: REFUTED, at full prominence.
- Report alongside, always: per-arm base pass rates; each class rate with
  a Wilson 95% interval and a one-sided binomial test against its arms'
  pooled base rate; Fisher exact (verdict x behaviour); the same class
  table computed on final-output ranks with identical bands (the output
  baseline's chance to beat the lens on its own terms).

## PREDICTED (locked)

- Behaviour collapses on reversed probes relative to exp-201's forward
  probes, most sharply in 201b and 201c (Berglund pattern); 201a drops
  least (its bridges name the godchild relation explicitly).
- The MISSING class, empty in exp-201 (0/180), populates here.
- Detector question is open by design: IF MISSING populates and sorts
  failures at the bar, the instrument is vindicated in its proper test;
  IF behaviour collapses and MISSING stays empty or does not sort, the
  instrument is finished as a failure detector.

## Data gathered

- Per-item JSON per arm: pair, probe verbatim, lens rank hs[11], lens
  rank hs[12], final-output rank, verdicts under both indexings,
  generation (8 and 50 tokens) verbatim, pass/fail.
- SME recreations: for 3 items per arm, the training sentences that arm
  actually saw for that pair (forward fact; bridge sentence for 201a),
  the reversed probe verbatim, the model's full generated continuation,
  and the three ranks — words and verbatim text, no paraphrase.
- RESULTS file: PREDICTED vs ACTUAL, verdict per the bars, all tables.

## Rules

Append-only: new folder ~/robot-mind/exp-201-reversed/ on the Jetson;
exp-201's files are never edited. No threshold refit on this data. Fail
loudly the moment a step fails.
