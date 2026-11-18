"""Diagnostic (post-scoring, interpretation only): why does the L12 lens
lean agree with the banked question-format pick on only ~66% of non-twin
items, when the lens at the last layer should be close to the model's own
output? Compares, on 8 items: banked model_pick; fresh model logits pick;
manual-ln_f lens pick on hidden_states[12] (the subst_rescore computation
used by run_q3.py); lm_head on hidden_states[12] with NO ln_f; and the
ln_f MODULE applied to hidden_states[12]. Changes nothing in the results."""
import json, torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

import os
ROBOT_MIND = os.environ.get("ROBOT_MIND", "/home/ecpi-student-05/robot-mind")  # Jetson working tree (data lives outside the repo); override with $ROBOT_MIND

EW = ROBOT_MIND + "/pronoun-battery-v2/path2-early-warning"
items = [json.loads(l) for l in open(EW + "/confirmatory_items.jsonl")][:8]
res = {(r["split"], r["item"]): r for r in json.load(open(EW + "/results_confirmatory.json"))}
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().cuda()
lw = lm.transformer.ln_f.weight.detach().float().cpu()
lb = lm.transformer.ln_f.bias.detach().float().cpu()
W = lm.lm_head.weight.detach().float().cpu()
lnf = lm.transformer.ln_f

with torch.no_grad():
    for it in items:
        n1, n2 = it["head1"], it["head2"]
        t1 = tok(" " + n1, add_special_tokens=False)["input_ids"][0]
        t2 = tok(" " + n2, add_special_tokens=False)["input_ids"][0]
        prompt = "%s What does the word %s refer to? Answer with one word:" % (it["sentence"], it["pronoun"])
        ids = tok(prompt)["input_ids"]
        o = lm(torch.tensor([ids]).cuda(), output_hidden_states=True)
        pick_logits = n1 if float(o.logits[0, -1, t1]) > float(o.logits[0, -1, t2]) else n2
        h = o.hidden_states[12][0, -1].float().cpu()
        hn = (h - h.mean()) / (h.var(unbiased=False) + 1e-5).sqrt() * lw + lb
        lg = W @ hn
        pick_manual = n1 if float(lg[t1]) > float(lg[t2]) else n2
        lg2 = W @ h
        pick_nolf = n1 if float(lg2[t1]) > float(lg2[t2]) else n2
        hm = lnf(o.hidden_states[12][0, -1].cuda()).float().cpu()
        lg3 = W @ hm
        pick_module = n1 if float(lg3[t1]) > float(lg3[t2]) else n2
        print("%s-%s banked=%s logits=%s manual=%s nolf=%s module=%s" %
              (it["split"], it["item"], res[(it["split"], it["item"])]["model_pick"],
               pick_logits, pick_manual, pick_nolf, pick_module))
print("DIAG_DONE")
