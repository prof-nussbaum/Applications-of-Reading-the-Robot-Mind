"""RTRM answer-storedness probe (audit tool).

One technique from Reading the Robot Mind (RTRM), the family of methods
for watching what a language model's layers keep and what they discard.

This tool uses the logit lens: it takes the internal state (768 numbers)
at a single layer, projects it through the model's output matrix, and
ranks all 50,257 vocabulary tokens. Whatever ranks near the top is what
that layer has on its mind. Sweeping across layers shows an answer
emerging -- or never emerging.

The companion RTRM technique is brute-force recreation: take one layer's
activations and recreate the *input* tokens from them by trying every
vocabulary token (all 50,257 -- that's the brute force), including nearby
tokens, to see what the layer preserved vs. transformed. Together, the
lens (what's coming next) and brute-force recreation (what survived from
the input) reveal precisely what each layer keeps and discards.

Here, for each (prompt, expected_answer) pair, we report the logit lens
rank of the answer's first piece at the final prompt position (layer 11).
Low rank = the layer built the retrievable link; high rank = it did not.

Validated in exp-200i/200j/200k/200n/200o:
  - fwd_e2 (cursed): median rank ~2400, 0/50 behavioral
  - ctl_e2 (control): median rank 1, 50/50 behavioral
  - within-model Spearman(rank, answer log-prob) = -0.65
  - mix_e2 flagger test: 30/30 trained -> rank 1-9, 30/30 untrained -> rank 13+,
    zero overlap, including a novel template (200o)

Honest note: for pure flagging, the model's final-output ranks work too
(trained: all rank 1; untrained: rank 8+). The lens is the microscope that
found the mechanism; the detector can just read the output.

Usage:
    python rtrm_probe.py --model <hf-id-or-path> --pairs pairs.json
    python rtrm_probe.py --model <hf-id-or-path> --pairs pairs.json --layers 5,7,9,11
pairs.json: [{"prompt": "...", "answer": "..."}, ...]
    "answer" is the expected continuation; only its first piece is probed.

Output: JSON list with rank + verdict per pair, plus per-layer ranks/tops.
With --layers, the console also prints the window: each layer's top token
in brackets with the target's rank, e.g.
    window: L5:[ now](5119) L7:[ now](466) L9:[ Steven](225) L11:[ Marcus](3)
Verdicts: rank <= 10 "STORED", <= 100 "WEAK", else "MISSING".
"""
import argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def tids(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def probe_layers(model, tok, lw, lb, W, prompt, answer, layers):
    """Return {layer: (target_rank, top_token_text)} for each layer."""
    ids = torch.tensor([tids(tok, prompt)])
    tgt = tids(tok, answer)[0]
    out = {}
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
        for L in layers:
            h = hs[L][0].float()
            mu = h.mean(-1, keepdim=True)
            var = ((h - mu) ** 2).mean(-1, keepdim=True)
            hn = (h - mu) / torch.sqrt(var + 1e-5) * lw + lb
            v = (hn @ W.T)[-1]
            rank = int((v > v[tgt]).sum()) + 1
            top = int(v.argmax())
            out[L] = (rank, tok.decode([top]))
    return out, tok.decode([tgt])


def probe(model, tok, lw, lb, W, prompt, answer, layer=11):
    out, piece = probe_layers(model, tok, lw, lb, W, prompt, answer, [layer])
    return out[layer][0], piece


def verdict(rank):
    if rank <= 10:
        return "STORED"
    if rank <= 100:
        return "WEAK"
    return "MISSING"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default="probe_results.json")
    ap.add_argument("--layers", default="11",
                    help="comma-separated layers for the window, e.g. '5,7,9,11'")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(a.model).eval()
    lw = model.transformer.ln_f.weight.detach().float()
    lb = model.transformer.ln_f.bias.detach().float()
    W = model.lm_head.weight.detach().float()

    pairs = json.load(open(a.pairs))
    out = []
    for p in pairs:
        per_layer, piece = probe_layers(model, tok, lw, lb, W,
                                        p["prompt"], p["answer"], layers)
        rank = per_layer[layers[-1]][0]
        rec = {"prompt": p["prompt"], "answer": p["answer"],
               "first_piece": piece, "rank": rank,
               "verdict": verdict(rank),
               "layer_ranks": {str(L): per_layer[L][0] for L in layers},
               "layer_tops": {str(L): per_layer[L][1] for L in layers}}
        out.append(rec)
        print(f"{verdict(rank):7s} rank {rank:>6} :: {p['prompt']!r} -> [{piece}]")
        if len(layers) > 1:
            win = " ".join(f"L{L}:[{per_layer[L][1]}]({per_layer[L][0]})"
                           for L in layers)
            print(f"         window: {win}")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
