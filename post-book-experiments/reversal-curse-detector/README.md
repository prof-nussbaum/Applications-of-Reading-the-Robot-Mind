# RTRM Answer-Storedness Probe

A minimal audit tool from the *Reading the Robot Mind* toolkit: given a
language model and a list of question/answer pairs, it flags answers the
model never actually stored — by reading what each layer kept and what it
threw away.

## The idea: a window into the layers

A language model rewrites information at each of its layers. RTRM
(Reading the Robot Mind) is a family of techniques for watching that
rewriting happen. Two matter here:

- **The logit lens** (this tool): project any single layer's internal
  state — the 768 numbers it produced — through the model's output matrix
  and rank all 50,257 tokens. Whatever ranks near the top is what that
  layer has on its mind. Sweep across layers and you watch an answer
  emerge — or never emerge.
- **Brute-force recreation** (companion technique): take one layer's
  activations and recreate the *input* tokens from them by trying every
  vocabulary token (all 50,257 — that's the brute force) and seeing which
  match best, including nearby tokens. If `[ Meridian]` is recreatable
  from layer 3 but gone by layer 10, that layer discarded it.

Together they show precisely what each layer keeps and what it discards.
This probe uses the lens at layer 11, where the answer's fate is decided.

## What it does

For each `(prompt, expected_answer)` pair, the probe reads the model's
internal state at the final prompt token (layer 11), applies the logit
lens, and reports the rank of the answer's first piece.

- Rank near 1: the layer built the link — the answer's onset is
  retrievable.
- Rank in the hundreds or thousands: the layer never built it — expect a
  wrong answer or a hallucination.

On the validation set (60 facts, half reverse-trained): every trained
fact ranked 9 or better (30/30 behaviorally correct); every untrained
fact ranked 13 or worse (0/30 correct). Zero overlap — including on a
sentence template the method was never tuned on.

**Honest note:** for pure flagging, the model's plain final-output ranks
work too — and separate even more cleanly (trained: all rank 1; untrained:
rank 8+). The lens is the microscope that found the mechanism (a
first-piece retrieval failure, localized across layers 5–11); the detector
itself can just read the output.

## Installation

See [INSTALL.md](INSTALL.md) — venv-based setup for two machines:

- **Jetson Orin 8 GB, JetPack 6** (GPU; ~1 minute for 60 pairs). Note:
  install PyTorch from NVIDIA's Jetson wheel, not plain
  `pip install torch` (which gives a CPU-only build on aarch64).
- **Windows 11, no GPU** (CPU; slower, but fine for a few pairs).

Quick start once the venv is active:

```bash
python rtrm_probe.py --model gpt2 --pairs demo_pairs.json --out my_results.json
```

No GPU required for small batches. And if you want the published
validation numbers without running anything, they are precomputed in
`results/validation_60.json` — `python3 show_validation.py` prints the
table (stdlib only, no torch needed).

## Usage

1. Create a JSON file of pairs:

```json
[
  {"prompt": "The capital of France is", "answer": " Paris"},
  {"prompt": "The lead cartographer of the Meridian Expedition is",
   "answer": " Odette Marlowe"}
]
```

Note: `answer` is the expected continuation; only its first piece is
probed. Include the leading space (GPT-2 tokenization).

2. Run:

```bash
python rtrm_probe.py --model gpt2 --pairs pairs.json --out results.json
```

`--model` accepts any HuggingFace causal-LM id or local path.

3. Read the verdicts:

```
STORED  rank      2 :: 'The capital of France is' -> [ Paris]
MISSING rank   1488 :: 'The lead cartographer of the Meridian Expedition is' -> [ Od]
```

Add `--layers 5,7,9,11` for the window — each layer's top token in
brackets with the target's rank in parentheses:

```
MISSING rank    675 :: 'The founder of Lumenworks Collective is' -> [ Helena]
         window: L5:[ now](17299) L7:[ located](6224) L9:[ Vincent](2784) L11:[ Elena](675)
```

Verdict thresholds: rank ≤ 10 → STORED, ≤ 100 → WEAK, else MISSING.

## Reproducing the validation

The `exp-200n`/`exp-200o` validation used a GPT-2 model fine-tuned on 90
statements (50 forward / 25 reverse / 10 discovery-forward / 5
discovery-reverse, 200 epochs, SGD lr 5e-4). The probe was run over all 60
reverse prompts:

| group | n | median rank | behavioral |
|---|---|---|---|
| expedition, reverse-trained | 25 | 1.0 | 25/25 correct |
| expedition, reverse-untrained | 25 | 675 | 0/25 correct |
| discovery, reverse-trained | 5 | 1.0 | 5/5 correct |
| discovery, reverse-untrained | 5 | 587 | 0/5 correct |

Trained max rank: 9. Untrained min rank: 13. No overlap.

## Limitations

- The probe reads the *first* answer piece only. Once the onset is
  retrieved, the model usually completes the rest correctly (exp-200l);
  a STORED verdict does not guarantee the full answer is perfect.
- Thresholds (10/100) were calibrated on GPT-2-small; other models may
  need recalibration.
- This is a detection instrument, not a fix — and not a proof of
  "knowledge." Flagged pairs are candidates for review or RAG coverage.
