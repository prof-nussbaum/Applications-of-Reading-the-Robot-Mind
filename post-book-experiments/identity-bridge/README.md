# The Identity Bridge, tested: does the bridge actually bridge?

## The paper's claim

"Breaking the Reversal Curse in Autoregressive Language Models via Identity
Bridge" (Ma et al., UC Berkeley, arXiv:2602.02470) proposes a data recipe for
the reversal curse: add identity-bridge facts phrased as compositions ("the
wife of A's husband is A"), up-weight the forward facts 6x. They report a
fine-tuned Llama-3.2-1B going from ~0% to ~40-50% on reverse questions.

We recreated the recipe twice on synthetic domains with GPT-2 small, and
checked it with the RTRM probe. This bundle is the full arc: the claim, the
instruments, the results, and what to look for next time.

## The instruments

Two ways of reading what a layer holds, applied at layers 5, 7, 9, 11 of
GPT-2 small:

**1. The logit lens.** Take the hidden state at the last input position,
clean it with the final LayerNorm, score all 50,257 vocabulary tokens with
the unembedding matrix, and rank them. The rank of the expected answer's
first piece tells how accessible the answer is at that layer.

**2. Cosine brute force.** Skip the LayerNorm and the unembedding. Compute
the cosine similarity between the raw hidden state and every one of the
50,257 token embeddings, and rank them all. The nearest tokens — the top
few — are the evidence; watch how they evolve across layers.

Both methods read the **last input position**. GPT-2 is a causal model:
each position sees only the tokens to its left, so the hidden state at the
last input position is the entire basis for the next token. The first
generated token is chosen from that state alone. Probing it layer by layer
is watching the decision get built.

Every record shows the **prompt tokens** — the exact tokenized input — so
there is no ambiguity about what was tested.

**Verdicts** (from the lens L11 rank): rank 1-10 = STORED, 11-100 = WEAK,
over 100 = MISSING.

## Test 1 — exp-200p: rephrased bridges (the unfair test)

Domain: 60 synthetic forward facts, no inverse relations at all.
- mix_e3 (bridge): forwards x6 + 60 bridge sentences that were rephrasings,
  not true compositions. No reverse facts.
- mix_e4 (no-bridge baseline): forwards x6. No bridges, no reverse facts.

**PREDICTED:** the bridge lifts reverse accuracy and shifts the trajectory
toward the learned shape.

**ACTUAL:** 0/60 behaviorally for both. But the lens moved: L11 median rank
118 (bridge) vs 1,971 (baseline); verdicts 1 STORED + 26 WEAK vs 0 + 5.
The bridge changed the internal representation by an order of magnitude
without crossing the behavioral threshold — a more confident wrong answer.
This bounds the paper's claim on our turf; it does not refute it, because
our "bridges" were never the paper's compositions.

## Test 2 — exp-200q: true two-hop bridges (the fair test)

Domain: 60 mentor/apprentice pairs — genuine inverse relations.
- Forward: "Odette Rook is the mentor of Gustav Aasen."
- Bridge (true composition): "The apprentice of Odette Rook is Gustav Aasen."
- Probe (never trained): "Odette Rook's apprentice is" -> [Gustav]
- mix_q3 (bridge): 60 forwards x6 + 60 bridges.
- mix_q4 (no-bridge baseline): 60 forwards x6.
- mix_q5 (isolation): 60 forwards x1, no bridges, no up-weighting.

All: GPT-2 small, SGD lr 5e-4, 200 epochs, seed 0. Formal sample-data audit
passed before training.

**PREDICTED:** bridge >= 25% behavioral vs baseline <= 5%; L11 rank gap >= 5x.

**ACTUAL:**

| | q3 (6x+bridges) | q4 (6x, no bridges) | q5 (1x, no bridges) |
|---|---|---|---|
| Behavioral | 60/60 (100%) | 53/60 (88%) | 19/60 (32%) |
| Lens L11 median | 5 | 49 | 50.5 |
| Cosine L11 median | 1 | 4,462 | 13,859 |

The honest surprise, at full prominence: **the baseline learned the reverse
at 88% from forward-only training.** Six-fold repetition of "M is the mentor
of A" taught GPT-2 small to answer "M's apprentice is" — no bridges needed.
The isolated effects:

- **Repetition** (q4 minus q5): +56 points behavioral. The biggest
  behavioral driver.
- **Bridges** (q3 minus q4): +12 points behavioral — but the internal
  representation transforms: cosine rank 4,462 -> 1.

And the caveat the data forced on us: the lens L11 median barely differs
between q4 and q5 (49 vs 50.5) despite the 56-point behavioral gap. The
lens caught the bridge effect and missed the repetition effect; the cosine
tracked both. Neither instrument is the whole story.

## One worked example (q3, bridge model)

Prompt: "Odette Rook's apprentice is" — expected [Gustav]

Prompt tokens: ['O','d','ette',' R','ook',"'s",' apprentice',' is']

Lens:   L5:[now](4918) -> L7:[Marcus](26) -> L9:[Gustav](1) -> L11:[Corn](3)
Cosine: L5:[a](14214) -> L7:[a](818) -> L9:[Frank](6) -> L11:[Gustav](1)

Generated: "Gustav Aasen." — correct. Both methods converge on the answer
by layer 9; the tokens are the evidence, the ranks are the summary.

## What the SME should look for in text results, going forward

1. **Watch the tokens evolve, not just the rank.** Generic words ("now",
   "located") giving way to names by layer 7-9 is the signature of an
   answer being built. A trajectory stuck on generic words through layer 11
   means the layer never found it.
2. **Demand both methods.** When the lens and the cosine agree, trust the
   reading. When they disagree (as with q4 vs q5), say so — the
   disagreement is itself a finding about what each instrument sees.
3. **Check the prompt tokens.** They are printed with every record. If you
   cannot see exactly what went in, you cannot trust what came out.
4. **Behavior is the ground truth; the probe explains.** A STORED verdict
   with wrong generation (we saw one: rank 5, wrong name out) means
   accessibility is not the same as knowledge.
5. **Beware the attractor.** When every prompt in a family gets the same
   wrong answer ("Ostara. Ostara is the capital of Ostara." x8 in exp-200p),
   the model has no reverse link — it is emitting its strongest
   pattern-match and confabulating the justification after.
6. **Separate repetition from structure.** Our three-way split showed
   repetition drives behavior while bridges drive representation. Any
   future "fix" for the curse should be tested against a repetition-only
   baseline before claiming the mechanism.

## Files

- `results/all_results.md` — all 300 per-item records (200p: 120 with lens
  windows; 200q: 180 with lens + cosine windows, prompt tokens, both
  generation lengths).
- `results/probe_*.json` — machine-readable records.
- `figures/trajectory_200p_lens.png` — exp-200p lens trajectories.
- `figures/trajectory_200q_threeway.png` — exp-200q lens + cosine medians,
  three models.
- `figures/trajectory_odette_rook_threeway.png` — one prompt
  ("Odette Rook's apprentice is") through all three models: lens and
  cosine top tokens per layer, with the expected [Gustav] rank.
- `figures/bridge_layers_q3.svg` — vertical layer-stack diagram for the
  q3 bridge model on the worked example, in the style of the
  reversal-curse-detector post.
- `figures/probe_diagram_bridge.svg` — "where the probe looks": the
  GPT-2 small architecture with the four probe layers, and the worked
  example showing both instruments (logit lens and cosine nearest) at
  every layer.

## Provenance and limits

Trained and probed October 2026 on a Jetson Orin Nano (GPT-2 small, CUDA).
Single seed (0) throughout — the gaps are large, but seed robustness is
untested. GPT-2 small (124M), not the paper's 1B model. Synthetic domains
throughout; nothing here is claimed about production models. Experiment
records live append-only in `ai-problem-analyses/exp-200/exp-200p/` and
`exp-200q/`; this bundle is the published summary.
