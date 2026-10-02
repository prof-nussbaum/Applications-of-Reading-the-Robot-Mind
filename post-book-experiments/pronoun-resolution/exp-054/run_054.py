"""exp-054: PCA-as-codec residual test (stand-in for Paul's pretrained AE).

Paul's idea (2026-09-29): use an autoencoder AS A CODEC -- subtract its
reconstruction from the original state vector. The residual holds what the
AE could NOT capture (the small perturbations), with the big carrier
variations removed. Then run the cosine neighbor search on RESIDUALS instead
of raw states. This is his Kaggle power-line trick (AE learns 'normal', the
residual reveals the anomaly) turned inward on the model.

His pretrained AE weights are NOT on either machine (searched Jetson +
workspace 2026-09-29: no weights anywhere, only the book's decoder-to-tokens
code which outputs vocab logits, not vectors, so it cannot serve as a codec).
This run uses the no-training equivalent: a LINEAR autoencoder with a k-dim
neck IS PCA. Codec(h) = mu + V V^T (h - mu); residual = h - codec(h), with
(V, mu) fit on the search pool only. If Paul produces his AE weights, rerun
with those.

PREDICTED (2026-09-29, before running): if decision-relevant information lives
in small perturbations drowned by big carrier variations, removing the top-64
principal components (the 'big variations') lets the referent surface: ALL FOUR
ceiling referents rank <= 500 by cosine in residual space (the same bar
exp-052's kill used). Raw exp-052 ranks were dog #110, king #17,061,
car #3,729, baby #1,196 -- the raw ranks are recomputed here as a pipeline
validation and must reproduce.

Pair items: the two members have bit-identical pronoun states, so identical
residuals; the ceiling items carry the test (same as exp-052).
"""
import json
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/exp-054"
E = "/home/ecpi-student-05/robot-mind/experiments"
B = "/home/ecpi-student-05/robot-mind/pronoun-battery"
BV = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2"
K = 64  # neck size

ITEMS = [
    ("pair1a", "The dog chased the cat because it was barking.", "it", ("dog", "cat")),
    ("pair1b", "The dog chased the cat because it was meowing.", "it", ("cat", "dog")),
    ("pair2a", "The man lifted the boy because he was strong.", "he", ("man", "boy")),
    ("pair2b", "The man lifted the boy because he was light.", "he", ("boy", "man")),
    ("pair3a", "The car hit the tree because it was speeding.", "it", ("car", "tree")),
    ("pair3b", "The car hit the tree because it had deep roots.", "it", ("tree", "car")),
    ("ceil1", "The dog barked because it was excited.", "it", ("dog", "cat")),
    ("ceil2", "The king ruled for years because he was wise.", "he", ("king", "farmer")),
    ("ceil3", "The car broke down because it needed repairs.", "it", ("car", "truck")),
    ("ceil4", "The baby cried because it was hungry.", "it", ("baby", "child")),
]

tok = GPT2TokenizerFast.from_pretrained("gpt2")
def word_of(i):
    return tok.decode([i]).strip().lower()

pool_all = torch.load(E + "/exp-032d-all-pools/pool_all.pt", map_location="cpu", weights_only=False)
p37 = torch.load(E + "/exp-037-pronoun-binding/pool_037.pt", map_location="cpu", weights_only=False)
p47 = torch.load(B + "/exp-047/pool_047.pt", map_location="cpu", weights_only=False)
spool = pool_all + p37 + p47
vecs, words = [], []
for r in spool:
    for p, v in enumerate(r["hs"][2]):
        vecs.append(v.float())
        words.append(word_of(r["ids"][p]))
M = torch.stack(vecs)
print("pool", tuple(M.shape), "mean unit-norm:", round(float(M.norm(dim=1).mean()), 4), flush=True)

mu = M.mean(dim=0)
Mc = M - mu
U, S, V = torch.pca_lowrank(Mc, q=K)
var_cap = float((S ** 2).sum() / (Mc ** 2).sum())
print("top-%d variance captured: %.4f" % (K, var_cap), flush=True)

def codec_residual(H):
    Hc = H - mu
    return Hc - (Hc @ V) @ V.T

R = codec_residual(M)
resid_frac = float((R ** 2).sum() / (Mc ** 2).sum())
print("residual energy fraction: %.4f" % resid_frac, flush=True)

tpool = torch.load(BV + "/exp-052/pool_052.pt", map_location="cpu", weights_only=False)
by_id = {r["id"]: r for r in tpool}
Rn = F.normalize(R, dim=1)
rows = []
for iid, sent, pron, (c1, c2) in ITEMS:
    rec = by_id[iid]
    ids = rec["ids"]
    ptok = tok(" " + pron, add_special_tokens=False)["input_ids"][0]
    pos = ids.index(ptok)
    h = rec["hs"][2][pos].float().unsqueeze(0)
    rh = codec_residual(h)
    sims_raw = (F.normalize(h, dim=1) @ M.T)[0]
    sims_res = (F.normalize(rh, dim=1) @ Rn.T)[0]
    out = {"id": iid, "referent": c1}
    for name, sims in (("raw", sims_raw), ("resid", sims_res)):
        order = torch.argsort(sims, descending=True)
        out[name + "_rank"] = next(int(i) + 1 for i in range(len(order)) if words[order[i]] == c1)
    order = torch.argsort(sims_res, descending=True)
    out["resid_top5"] = [[words[int(i)], round(float(sims_res[int(i)]), 3)] for i in order[:5]]
    rows.append(out)
    print(iid, c1, "raw #%-6d resid #%d" % (out["raw_rank"], out["resid_rank"]), "top5:", out["resid_top5"], flush=True)

json.dump({"K": K, "var_captured": round(var_cap, 4),
           "residual_energy_fraction": round(resid_frac, 4), "rows": rows},
          open(D + "/results_054.json", "w"), indent=1)
print("wrote results_054.json", flush=True)
