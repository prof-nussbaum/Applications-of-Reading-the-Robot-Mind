"""exp-054e: SEEDED reproducibility rerun of exp-054c (Paul authorized 2026-09-29).

Same C2 recipe and same items as exp-054c, ONE deliberate change:
torch.manual_seed(0) is predeclared BEFORE torch.pca_lowrank.

Why: the unseeded 054c run produced exact ranks (king #51, cat #2,217, ...)
that a 2026-09-29 render-verification pass showed are draw-dependent and
exactly unreproducible (king 50-124, cat 847-4062 across redraws). Public
claims need reproducible numbers, so this seeded rerun's numbers supersede
054c's for citation. exp-054c's folder stays frozen as the lab record.

PREDICTED (predeclared, before running):
  (a) all 4 ceiling referents (dog, king, car, baby) rank <= 500 in the
      subtracted (C2) list -- the exp-054 bar.
  (b) referent ranks fall within the diagnostic draw ranges observed
      2026-09-29: king 50-130, dog 50-80, car 29-45, baby 55-90.
      Distractor ranks are draw-dependent; recorded as-is, no prediction.
"""
import json
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/exp-054e"
E = "/home/ecpi-student-05/robot-mind/experiments"
B = "/home/ecpi-student-05/robot-mind/pronoun-battery"
BV = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2"
K = 64
SEED = 0

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
RENDER_ITEMS = ["ceil2", "ceil1", "ceil3", "ceil4", "pair3a", "pair3b"]

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
print("pool", M.shape[0], flush=True)
assert M.shape[0] == 30099, "pool size changed vs 054c!"

# pool side: exp-054 recipe, true cosine -- SEEDED (the one deliberate change)
mu = M.mean(dim=0)
Mc = M - mu
torch.manual_seed(SEED)   # <-- predeclared seed; 054c was unseeded
U, S, V = torch.pca_lowrank(Mc, q=K)


def coderes(H):
    Hc = H - mu
    return Hc - (Hc @ V) @ V.T


R = coderes(M)
Rn = F.normalize(R, dim=1)
resid_frac = float((R ** 2).sum() / (Mc ** 2).sum())
print("pool residual energy fraction: %.4f" % resid_frac, flush=True)

# pronoun token means (query side only)
uniq = sorted(set(words))
w2i = {w: i for i, w in enumerate(uniq)}
wi = torch.tensor([w2i[w] for w in words])
counts = torch.bincount(wi)
mu_w = torch.zeros(len(uniq), M.shape[1]).index_add_(0, wi, M) / counts.unsqueeze(1)

tpool = torch.load(BV + "/exp-052/pool_052.pt", map_location="cpu", weights_only=False)
by_id = {r["id"]: r for r in tpool}


def rank_of(sims, target):
    order = torch.argsort(sims, descending=True)
    return next(int(i) + 1 for i in range(len(order)) if words[order[i]] == target)


rows, renders = [], {}
for iid, pron, (c1, c2) in ITEMS:
    rec = by_id[iid]
    ids = rec["ids"]
    pos = ids.index(tok(" " + pron, add_special_tokens=False)["input_ids"][0])
    h = rec["hs"][2][pos].float().unsqueeze(0)
    mu_q = mu_w[w2i[pron]].unsqueeze(0)
    q1 = h - mu_q
    q1c = q1 - mu
    q2 = q1c - (q1c @ V) @ V.T
    s1 = (F.normalize(q1, dim=1) @ Rn.T)[0]
    s2 = (F.normalize(q2, dim=1) @ Rn.T)[0]
    row = {"id": iid, "referent": c1, "distractor": c2,
           "C1_rank": rank_of(s1, c1), "C1_wrong": rank_of(s1, c2),
           "C2_rank": rank_of(s2, c1), "C2_wrong": rank_of(s2, c2)}
    rows.append(row)
    print("%(id)s %(referent)s C1 #%(C1_rank)d (wrong #%(C1_wrong)d) "
          "C2 #%(C2_rank)d (wrong #%(C2_wrong)d)" % row, flush=True)
    if iid in RENDER_ITEMS:
        order = torch.argsort(s2, descending=True)
        top = [(words[int(order[i])], round(float(s2[order[i]]), 4)) for i in range(200)]
        renders[iid] = {
            "prompt": tok.decode(ids),
            "pronoun": pron, "pron_pos": pos,
            "referent": c1, "distractor": c2,
            "C2_rank": row["C2_rank"], "C2_wrong": row["C2_wrong"],
            "top200": top,
        }

json.dump({"K": K, "seed": SEED, "residual_energy_fraction": resid_frac,
           "note": "seeded rerun of 054c; numbers here supersede 054c for public citation",
           "rows": rows, "renders": renders},
          open(D + "/results_054e.json", "w"), indent=1)
print("wrote results_054e.json", flush=True)
