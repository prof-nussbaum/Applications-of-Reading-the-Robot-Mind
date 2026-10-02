"""exp-053: logit lens on the exp-052 pronoun items (method 3).

QUESTION: where/when does the model's own readout (final dense layer) favor the
correct referent -- at the pronoun position, or only at the final token?

PREDICTED (2026-09-29, before running):
- At the pronoun position: the logit margin (correct - wrong) stays near zero /
  does not track the right answer at any layer. Built-in control: pair1a/pair1b
  (and 2a/2b, 3a/3b) have bit-identical pronoun states (same left context), so
  their pronoun-margin curves MUST be identical -- yet the correct answer flips.
  The pronoun lens therefore cannot be right for both members of a pair.
- At the final token: the correct candidate pulls ahead in late layers, and the
  block-12 lens reproduces the exp-052 behavioral margins exactly (sanity check).
- Ceiling items at the pronoun: open -- a single unambiguous noun might already
  bind; report as observed.

No new dataset: the 10 items are the audited exp-052 set (AUDIT_052.txt).
Lens: verified 2026-09-29 that hidden_states has 13 entries =
(emb, block1..block11 raw, ln_f(block12)). I.e. our saved L12 IS the head's
input (post final LayerNorm); L0/L6 are pre-norm. Lens(k<12) = lm_head(ln_f(hs[k]));
lens(12) = lm_head(hs[12]) with no extra norm. The block-12 lens must reproduce
the exp-052 behavioral margins exactly (sanity check).
"""
import json
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/exp-053"
items = [
    {"id": "pair1a", "sentence": "The dog chased the cat because it was barking.",
     "pronoun": "it", "correct": "dog", "wrong": "cat"},
    {"id": "pair1b", "sentence": "The dog chased the cat because it was meowing.",
     "pronoun": "it", "correct": "cat", "wrong": "dog"},
    {"id": "pair2a", "sentence": "The man lifted the boy because he was strong.",
     "pronoun": "he", "correct": "man", "wrong": "boy"},
    {"id": "pair2b", "sentence": "The man lifted the boy because he was light.",
     "pronoun": "he", "correct": "boy", "wrong": "man"},
    {"id": "pair3a", "sentence": "The car hit the tree because it was speeding.",
     "pronoun": "it", "correct": "car", "wrong": "truck"},
    {"id": "pair3b", "sentence": "The car hit the tree because it had deep roots.",
     "pronoun": "it", "correct": "tree", "wrong": "car"},
    {"id": "ceil1", "sentence": "The dog barked because it was excited.",
     "pronoun": "it", "correct": "dog", "wrong": "cat"},
    {"id": "ceil2", "sentence": "The king ruled for years because he was wise.",
     "pronoun": "he", "correct": "king", "wrong": "farmer"},
    {"id": "ceil3", "sentence": "The car broke down because it needed repairs.",
     "pronoun": "it", "correct": "car", "wrong": "truck"},
    {"id": "ceil4", "sentence": "The baby cried because it was hungry.",
     "pronoun": "it", "correct": "baby", "wrong": "child"},
]

def qa(it):
    return it["sentence"] + " What does the word %s refer to? Answer with one word:" % it["pronoun"]

tok = GPT2TokenizerFast.from_pretrained("gpt2")
dev = "cuda" if torch.cuda.is_available() else "cpu"
lm = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
ln_f, W = lm.transformer.ln_f, lm.lm_head.weight

rows = []
with torch.no_grad():
    for it in items:
        ids = tok(qa(it), return_tensors="pt")["input_ids"].to(dev)
        hs = lm(ids, output_hidden_states=True).hidden_states
        assert len(hs) == 13, len(hs)
        tids = ids[0].tolist()
        ptok = tok(" " + it["pronoun"], add_special_tokens=False)["input_ids"][0]
        pos = tids.index(ptok)
        tc = tok(" " + it["correct"], add_special_tokens=False)["input_ids"][0]
        tw = tok(" " + it["wrong"], add_special_tokens=False)["input_ids"][0]
        pm, fm = [], []
        for l in range(13):
            h = hs[l][0]
            lp_ = (ln_f(h[pos]) if l < 12 else h[pos]) @ W.T
            lf_ = (ln_f(h[-1]) if l < 12 else h[-1]) @ W.T
            pm.append(round(float(lp_[tc] - lp_[tw]), 3))
            fm.append(round(float(lf_[tc] - lf_[tw]), 3))
        rows.append({"id": it["id"], "pron_pos": pos,
                     "pron_margin": pm, "final_margin": fm})
json.dump(rows, open(D + "/results_053.json", "w"), indent=1)
for r in rows:
    print(r["id"], "pron_pos", r["pron_pos"])
    print("  pron :", r["pron_margin"])
    print("  final:", r["final_margin"], flush=True)
