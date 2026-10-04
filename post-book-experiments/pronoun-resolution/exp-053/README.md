# exp-053 — logit lens: the head reads what cosine obscures

**Question:** where does the model's own readout (the final dense layer +
softmax) favor the correct referent — at the pronoun position, or only at
the final token? For each of the 10 audited exp-052 prompts, every layer's
hidden state was pushed through the model's own unembedding, and the logit
margin (correct minus wrong candidate) recorded at the pronoun token and at
the final token.

**Result:** the equating IS in the pronoun state in the head's linear
channel. Read at the pronoun, the trained output head favors the correct
referent on all 4 ceiling items (e.g. king by +5.64 nats). Cosine
neighborhoods obscure what the head's linear readout recovers — Paul's
small-perturbations hypothesis, confirmed on the decision layer.

## Files

- `run_053.py` — logit-lens measurement across layers
- `results_053.json` — per-item, per-layer logit margins
- `SUMMARY_053.txt` — PREDICTED vs ACTUAL write-up
- `README.md` — this file
