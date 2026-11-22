# Q3 next-word re-analysis — variant N labelling at the answer position (L11)

Run 2026-11-22, Jetson, per R-012 §4 (step 2 of the R-013 sequence;
step 3 in R-014's updated numbering). **Exploratory: no Q3 verdict
changes.** Script: `q3_nextword_l11.py`. Records: `nextword_ranks.json`
(per item, both labellings), `nextword_scores.json`. The raw
answer-position states are banked on the Jetson at
`q3-pick-tracking/nextword_2026-11-22/answer_states_l11.pt`, so no
future relabelling needs re-extraction.

## What was done

The banked Amendment-1 stage-2 extraction (617 confirmatory items,
answer position = last prompt token, hs[11], pool_L11.pt, PCA-64
seed 0, no pronoun-mean subtraction) was repeated exactly, with one
change: each pool neighbour is labelled by the token that FOLLOWED
it in its source sentence (variant N labelling, as in the E1b
redesign), instead of by its own token. Sentence-final pool states
have no next token; they keep their rank positions and never match
a head word. The lean is the run_confirmatory rule (head with the
lower first-occurrence rank) under each labelling.

Gate, passed before scoring: the current-token ranks and leans
computed in this run reproduce `l11/answer_l11_ranks.json` on all
617 items (pool top-64 variance 0.5026, as stage 1).

Scoring footing (R-003 §2, verified against the banked records
before running): per-item balanced accuracy against the
question-format pick over all 617 items, positive class = the pick
is head2. On the banked records this footing gives current-token
list 0.6681, lens L11 0.9277, fixed rules 0.500 — the comparators
R-012 §4 named (0.668 / 0.928 / 0.500).

## PREDICTED vs ACTUAL

PREDICTED (the labelling diagnosis behind variant N): at the answer
position the state is about to produce the answer word, so
neighbours labelled by the word that followed them should expose
that word, and the list's balanced accuracy should rise well above
the current-token 0.668, toward the lens's 0.928.

ACTUAL: it does not rise. Variant N scores **0.6604**, a hair below
the current-token list's 0.6681 — the labelling change makes no
material difference at the answer position.

| Readout (L11, answer position) | Balanced accuracy vs question-format pick |
|---|---|
| Variant N list (next-token labels) | 0.6604 |
| Current-token list (banked recipe) | 0.6681 |
| Lens | 0.9277 |
| Best fixed rule (always head2 / always head1) | 0.500 |

Native Q3 strata (pair-level, vs the question-format pick) tell the
same story with texture: same-Q pairs both-match 128/217 (59.0%)
under variant N vs 141/217 (65.0%) current-token; flip-Q pairs
both-match 1/20 vs 4/20; non-twin items 105/143 (73.4%) vs 100/143
(69.9%). The small non-twin gain does not offset the twin-pair drop,
and no stratum moves toward the lens.

## Reading

The labelling was not what binds the list at the answer position.
This is consistent with the settled Q3 account and changes none of
it: the neighbour list recovers sentence content and leans weakly
toward the model's question-format pick (balanced accuracy ≈ 0.66
under either labelling), while the lens reads the same pick at
0.93. Where variant N's sibling fix mattered in E1b, the binding
constraint was the pool (variant T: the source sentences were not
in it), not the neighbour labels — variant N missed there too
(0.5625 / 0.5167 against the 0.65 gate).
