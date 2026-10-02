"""exp-054d: mean removal AFTER PCA (Paul's extra test).

054c-C2 order: (1) subtract pronoun token-mean, (2) global-center + PCA-64.
This run swaps it: (1) PCA residual with global centering, (2) subtract the
pronoun's mean computed IN RESIDUAL SPACE.

Algebra: let mu = global mean, V = top-64 PCs, mu_pron = pronoun token mean.
  C2:  q2 = (h - mu_pron - mu) - V V^T (h - mu_pron - mu)
  054d: r = (h - mu) - V V^T (h - mu); m = residual-space pronoun mean
        = (mu_pron - mu) - V V^T (mu_pron - mu)
        q_new = r - m = h - mu_pron - V V^T (h - mu_pron)
  => q_new - q2 = mu - V V^T mu =: c   (constant across queries!)

So the order-swap differs from C2 only by the fixed vector c, the
PCA-residual of the global mean. If the top-64 captures the mean direction,
c ~= 0 and the order is immaterial.

PREDICTED (before running): 054d closely matches 054c-C2 -- all four ceiling
referents <= 500, per-item rank shifts small vs the 054->054c gains. Report
||c|| relative to query norms as the check.
"""
import json
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/exp-054d"
E = "/home/ecpi-student-05/robot-mind/experiments"
B = "/home/ecpi-student-05/robot-mind/pronoun-battery"
BV = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2"
K = 64

ITEMS = [
    ("pair1a", "it", ("dog", "cat")),
    ("pair1b", "it", ("cat", "dog")),
    ("pair2a", "he", ("man", "boy")),
    ("pair2b", "he", ("boy", "man")),
    ("pair3a", "it", ("car", "tree")),
    ("pair3b", "it", ("tree", "car")),
    ("ceil1", "it", ("dog", "cat")),
    ("ceil2", "he", ("king", "farmer")),
    ("ceil3", "it", ("car", "truck")),
    ("ceil4", "it", ("baby", "child")),
]

tok = GPT2TokenizerFast.from_pretrained("gpt2")
def word_of(i):
    return tok.decode([i]).strip().lower()

spool = (torch.load(E + "/exp-032d-all-pools/pool_all.pt", map_location="cpu", weights_only=False)
         + torch.load(E + "/exp-037-pronoun-binding/pool_037.pt", map_location="cpu", weights_only=False)
         + torch.load(B + "/exp-047/pool_047.pt", map_location="cpu", weights_only=False))
vecs, words = [], []
for r in spool:
    for p, v in enumerate(r["hs"][2]):
        vecs.append(v.float())
        words.append(word_of(r["ids"][p]))
M = torch.stack(vecs)

mu = M.mean(dim=0)
Mc = M - mu
U, S, V = torch.pca_lowrank(Mc, q=K)
def coderes(H):
    Hc = H - mu
    return Hc - (Hc @ V) @ V.T
R = coderes(M)
Rn = F.normalize(R, dim=1)

# residual-space pronoun means (query side only; pool keeps word identity)
is_he = torch.tensor([w == "he" for w in words])
is_it = torch.tensor([w == "it" for w in words])
mu_he_r = R[is_he].mean(dim=0)
mu_it_r = R[is_it].mean(dim=0)
print("n_he=%d n_it=%d" % (int(is_he.sum()), int(is_it.sum())), flush=True)

# the constant offset between the two orders
c = mu - (mu @ V) @ V.T
print("||c||=%.6f  ||mu||=%.4f  ratio=%.6f" % (float(c.norm()), float(mu.norm()), float(c.norm() / mu.norm())), flush=True)

tpool = torch.load(BV + "/exp-052/pool_052.pt", map_location="cpu", weights_only=False)
by_id = {r["id"]: r for r in tpool}

def rank_of(sims, target):
    order = torch.argsort(sims, descending=True)
    return next(int(i) + 1 for i in range(len(order)) if words[order[i]] == target)

# C2 queries for the ||c|| sanity comparison
mu_w_he = M[is_he].mean(dim=0, keepdim=True)
rows = []
for iid, pron, (c1, c2) in ITEMS:
    rec = by_id[iid]
    ids = rec["ids"]
    pos = ids.index(tok(" " + pron, add_special_tokens=False)["input_ids"][0])
    h = rec["hs"][2][pos].float().unsqueeze(0)
    r_q = coderes(h)
    mu_pr = mu_he_r if pron == "he" else mu_it_r
    q = r_q - mu_pr.unsqueeze(0)
    s = (F.normalize(q, dim=1) @ Rn.T)[0]
    row = {"id": iid, "referent": c1, "distractor": c2,
           "D_rank": rank_of(s, c1), "D_wrong": rank_of(s, c2),
           "qnorm": round(float(q.norm()), 4)}
    rows.append(row)
    print("%(id)s %(referent)s D #%(D_rank)d (wrong #%(D_wrong)d)" % row, flush=True)

json.dump({"K": K, "c_norm": float(c.norm()), "mu_norm": float(mu.norm()),
           "rows": rows}, open(D + "/results_054d.json", "w"), indent=1)
print("wrote results_054d.json", flush=True)
