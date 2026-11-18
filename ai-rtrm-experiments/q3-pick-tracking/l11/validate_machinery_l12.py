"""Q3 Amendment 1, stage 1, gate B: reproduce the banked pronoun-position
L12 list ranks from results_confirmatory.json on 12 probed items, using
the confirmatory recipe WITH pronoun-mean subtraction and the REBUILT
L12 pool (pool_L12_rebuild.pt) instead of the banked pool. Pattern copied
from ~/robot-mind/q3-pick-tracking/validate_machinery.py; the only change
is the pool source (hs[0] of the rebuild = hidden_states[12]). If these
ranks match exactly, the adapted build + readout chain is proven at
layer 12, and the same code path at layer 11 is trustworthy.
"""
import json, torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

import os
ROBOT_MIND = os.environ.get("ROBOT_MIND", "/home/ecpi-student-05/robot-mind")  # Jetson working tree (data lives outside the repo); override with $ROBOT_MIND

EW = ROBOT_MIND + "/pronoun-battery-v2/path2-early-warning"
L11 = ROBOT_MIND + "/q3-pick-tracking/l11"
K = 64
PRONOUNS = ["he", "she", "it", "they"]

items = [json.loads(l) for l in open(EW + "/confirmatory_items.jsonl")][:12]
res = {(r["split"], r["item"]): r for r in json.load(open(EW + "/results_confirmatory.json"))}

tok = GPT2TokenizerFast.from_pretrained("gpt2")
def word_of(i): return tok.decode([i]).strip().lower()

spool = torch.load(L11 + "/pool_L12_rebuild.pt", map_location="cpu", weights_only=False)
vecs, words = [], []
for r in spool:
    for p, v in enumerate(r["hs"][0]):  # rebuild hs[0] = layer 12
        vecs.append(v.float()); words.append(word_of(r["ids"][p]))
del spool
M = torch.stack(vecs)
mu = M.mean(0)
torch.manual_seed(0)
U, S, V = torch.pca_lowrank(M - mu, q=K)
def coderes(H):
    Hc = H - mu
    return Hc - (Hc @ V) @ V.T
Rn = F.normalize(coderes(M), dim=1)
pron_mu = {}
for pron in PRONOUNS:
    mask = torch.tensor([w == pron for w in words])
    pron_mu[pron] = M[mask].mean(0)
print("pool %d states" % len(M), flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(dev)

def rank_of(target, sims):
    target = target.lower()
    order = torch.argsort(sims, descending=True)
    for i in range(len(order)):
        if words[order[i]] == target: return i + 1
    return None

bad = 0
with torch.no_grad():
    for it in items:
        pron = it["pronoun"]
        prompt = "%s What does the word %s refer to? Answer with one word:" % (it["sentence"], pron)
        ids = tok(prompt)["input_ids"]
        o = lm(torch.tensor([ids]).to(dev), output_hidden_states=True)
        h = F.normalize(o.hidden_states[12][0].float().cpu(), dim=1)
        pid = tok(" " + pron, add_special_tokens=False)["input_ids"][0]
        pos = [i for i, x in enumerate(ids) if x == pid][0]
        q2 = coderes(h[pos].unsqueeze(0) - pron_mu[pron])
        sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
        got = (rank_of(it["head1"], sims), rank_of(it["head2"], sims), rank_of(it["label_head"], sims))
        r = res[(it["split"], it["item"])]
        want = (r["r_head1"], r["r_head2"], r["r_label"])
        ok = got == want
        bad += 0 if ok else 1
        print("%s-%s got=%s want=%s %s" % (it["split"], it["item"], got, want, "OK" if ok else "** MISMATCH **"), flush=True)
print("MACHINERY:", "VALIDATED" if bad == 0 else "FAILED on %d items" % bad)
raise SystemExit(1 if bad else 0)
