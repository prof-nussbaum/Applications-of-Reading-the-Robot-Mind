# Confirmation on seeds 1–2 — bars LOCKED before building

Proposed by the referee in R-014 §4 (PR #18, 2026-11-22). Approved
by Paul 2026-11-22 ("I'm OK with Fable's proposal if you are. Go!").
Locked by Francis in this file, in this commit, BEFORE any seed 1–2
variant-T build is run. The bars are the referee's, with the
clarifications below; nothing here was changed after seeing any
seed 1–2 number (none exists yet).

## The bars — each must hold on BOTH seeds 1 and 2

- **C1, finds the source.** Fine-tuned model, variant T: the pair's
  own reversed sentence is the top neighbour in at least 20 of the
  30 stored-trained pairs. (Seed 0: 26 of 30.)
- **C2, from what was learned.** Control U (untrained base GPT-2,
  variant-T construction): the pair's own reversed sentence is the
  top neighbour in at most 13 of the 30 stored pairs — the F-014
  criterion. (Seed 0: 0 of 30.)
- **C3, beyond the words.** BM25 gives the pair's forward and
  reversed sentences equal scores in at least 28 of the 30 stored
  pairs. (Seed 0: 30 of 30.)
- **C4, follows the model, not the text.** Forward-only pairs: BM25
  ranks the pair's forward sentence first in at least 28 of 30,
  while the fine-tuned model's variant-T list has that sentence in
  the top 20 for at most 10 of 30, and the model answers none of
  the 30 forward-only probes. (Seed 0: BM25 30 of 30 at rank 1;
  fine-tuned list 5 of 30 in the top 20; answers 0 of 30.)

## Clarifications (Francis, fixed with the lock)

1. **C2 and the BM25 legs of C3/C4 are seed-independent, and are
   already run.** The three seeds share one dataset
   (`dataset_fresh.json`: same 60 pairs, same 30 reversed bridges,
   same fillers) and differ only in the fine-tuning run. The
   untrained model has no seed, and BM25 reads only the texts, so
   the seed-0 Control U and Control B records
   (`controls/controlU_seed0.json`, `controls/controlB_seed0.json`,
   `controls/controlB_scores.json`) ARE the control records for
   seeds 1 and 2: C2 = 0 of 30, C3 = 30 of 30 equal scores, C4's
   BM25 leg = 30 of 30 at rank 1. What remains to be built per seed
   is the fine-tuned variant-T list (C1, and C4's list and
   behaviour legs).
2. **Ranks.** Variant-T ranks are first-occurrence ranks over the
   neighbour order of the extended pool, exactly as in the seed-0
   build (`build_variants_seed0.py`, records at ec36f83): the rank
   of a sentence is the rank of its best-placed state. BM25 ranks
   are over the 1,906 pool sentences under the stable sort of the
   seed-0 control (ties keep corpus order); C3 is scored on the
   BM25 *scores*, not the ranks, so the tie-break cannot enter it.
3. **Behaviour leg of C4** is read from `behavior_fresh.json` for
   the seed's own model (the 8-token pass on the reversed probe),
   the same records the seed-0 build used.
4. **Built-in gate, per seed:** before any variant is scored, the
   recomputed current-token top-20s must reproduce the banked
   `lists_seed{1,2}.json` exactly (60/60), as the seed-0 build's
   gate did for seed 0. A mismatch aborts the build.

## Reported whatever the bars do

Per R-014 §4: the pass-versus-fail split within stored pairs (the
reversed-sentence ranks for the pairs the model answers vs the
pairs it does not), and the same rule-family tables as ec36f83,
for each seed.

## Outcome sentences (the referee's table, R-014 §4)

| Result | Sentence the paper may carry |
|---|---|
| C1 to C4 hold | In a small model fine-tuned on invented facts, a nearest-neighbour readout of the model's state retrieved the training sentence the ability came from, where a word search could not distinguish it and an untrained model's state did not retrieve it. |
| C1, C2 hold; C3 or C4 fails | The state retrieves its source and the retrieval depends on fine-tuning; no advantage over a word search was shown. |
| C1 or C2 fails | Seed 0 did not replicate. |
