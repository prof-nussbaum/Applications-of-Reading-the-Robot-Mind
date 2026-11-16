# Testing the Identity Bridge: does the bridge actually bridge?

**Question.** A published paper proposes a data recipe that fixes the reversal
curse. We recreated the recipe on our synthetic domain and checked it with the
RTRM probe.

**The paper.** "Breaking the Reversal Curse in Autoregressive Language Models
via Identity Bridge" (Ma et al., UC Berkeley, arXiv:2602.02470). Their recipe:
add identity-bridge facts phrased as compositions ("the wife of A's husband is
A"), up-weight the forward facts 6x. They report a fine-tuned Llama-3.2-1B
going from ~0% to ~40-50% on reverse questions.

**What we did.**

- **mix_e3** (bridge model): 50 expedition-forward + 10 discovery-forward
  facts, each repeated 6x, plus 60 bridge sentences. No reverse facts.
- **mix_e4** (no-bridge baseline): the same 60 forward facts repeated 6x.
  No bridges. No reverse facts.
- Both: GPT-2 small, SGD lr 5e-4, no momentum, batch size 1, 200 epochs,
  seed 0. Final mean loss 0.2712 (mix_e3) / 0.2563 (mix_e4).
- Probed with the RTRM probe (`../reversal-curse-detector/rtrm_probe.py`):
  60 reverse prompts, rank of the expected first piece at layers 5/7/9/11,
  plus greedy generation.

## PREDICTED

The bridge model beats the no-bridge baseline on reverse prompts, both
behaviorally and in the layer trajectory.

## ACTUAL

Behavioral accuracy: **0/60 for both models.** The bridge did not fix the
curse behaviorally. This refutes the prediction. It does not refute the paper
(see caveats).

But the probe caught movement that the accuracy number calls zero:

| Layer | mix_e3 (bridge) median rank | mix_e4 (no bridge) median rank |
|---|---|---|
| L5 | 1,351 | 4,727 |
| L7 | 299 | 3,617 |
| L9 | 392 | 1,660 |
| L11 | 118 | 1,971 |

L11 verdicts: mix_e3 = 1 STORED, 26 WEAK, 33 MISSING; mix_e4 = 0 STORED,
5 WEAK, 55 MISSING. Behavioral: 0/60 vs 0/60.

## One worked example (mix_e3, bridge model)

Prompt: "The founder of Lumenworks Collective is"

Expected first piece: [ Helena]

L5: [ now] rank 5865 -- L7: [ Nathan] rank 1644 -- L9: [ Nathaniel] rank 280
-- L11: [ glac] rank 67

Generated: "Nathaniel Webb. Webb is the inventor of" -- wrong.

The rank improved by two orders of magnitude across the layers, and the model
still produced a confident wrong name.

## Reading

The bridge moved the internal representation substantially -- more than an
order of magnitude at the final layer -- without crossing the behavioral
threshold. A more confident wrong answer, measured across all 60 items.
Accuracy says "nothing happened." The layers say "something real happened,
not enough."

## Caveats (honest)

1. Our synthetic domain has no inverse relations (nothing like husband/wife),
   so our bridge sentences were rephrasings, not true two-hop compositions
   like the paper's. This bounds the paper's claim on our turf; it does not
   refute it.
2. GPT-2 small (124M parameters), not the paper's 1B model.
3. Single seed (0). The gap is large, but seed robustness is untested.
4. The mix_e3 trajectory is not monotonic (L7 median 299, L9 median 392).
   Real data, reported as-is.

## What this is for

The paper proposes a fix; the probe checks whether the fix actually landed,
and shows where the confidence went instead of the answer. We have not proved
anything about the curse in general. We have a hallucination detector, and
this is what it caught.

## Files

- `results/probe_mix_e3.json`, `results/probe_mix_e4.json` -- all 60 items
  each: prompt, expected first piece, per-layer ranks, top tokens, generated
  text, behavioral correctness.
- `results/summary.md` -- aggregate tables.
- `figures/trajectory_bridge_vs_baseline.png` -- per-layer median rank,
  bridge vs no-bridge baseline.

## Provenance

Trained and probed 2026-10-12/13 on a Jetson Orin Nano (GPT-2 small, CUDA).
Probe used the corrected lens (`output_hidden_states`, same as the published
probe). Experiment records live in `ai-problem-analyses/exp-200/exp-200p/`
(append-only; this folder is the published summary).
