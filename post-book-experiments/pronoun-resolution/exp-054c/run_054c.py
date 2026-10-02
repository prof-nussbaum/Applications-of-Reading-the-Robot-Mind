"""exp-054c: asymmetric token-mean (Paul's suggestion, better reading).

exp-054b-B applied token-demeaning SYMMETRICALLY (every pool state demeaned
by its own word's mean) and failed the bar (king #9,202). Mechanism: for
pool CONTENT words, the word-mean CONTAINS the word's identity -- subtracting
it deletes the very king-ness we search for. Paul's 'subtract mean of that
token' is better read as applying to the QUERY token (the pronoun): remove
the pronoun-typical carrier from the query, but leave pool word identity
intact.

Variant C (asymmetric):
  Pool side: exp-054 recipe (global-center, PCA-64, residual) with TRUE
    cosine (both sides normalized; see 054b note on the half-precision
    norm wobble in exp-054's raw path).
  Query side C1: q = h - mu_pron (pronoun token-mean only).
  Query side C2: q = (h - mu_pron) - V V^T (h - mu_pron) (also project out
    the global top-64; pool residuals are ~zero there anyway, so C1 ~= C2).

PREDICTED (before running): C2 puts all four ceiling referents <= 500
(same bar). C1 exploratory; correct-vs-wrong gaps exploratory.
"""
import json
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/exp-054c"
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
print("pool", M.shape[0], flush=True)

# pool side: exp-054 recipe, true cosine
mu = M.mean(dim=0)
Mc = M - mu
U, S, V = torch.pca_lowrank(Mc, q=K)
def coderes(H):
    Hc = H - mu
    return Hc - (Hc @ V) @ V.T
R = coderes(M)
Rn = F.normalize(R, dim=1)
print("pool residual energy fraction: %.4f" % float((R ** 2).sum() / (Mc ** 2).sum()), flush=True)

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

rows = []
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
    print("%(id)s %(referent)s C1 #%(C1_rank)d (wrong #%(C1_wrong)d) C2 #%(C2_rank)d (wrong #%(C2_wrong)d)" % row, flush=True)

json.dump({"K": K, "rows": rows}, open(D + "/results_054c.json", "w"), indent=1)
print("wrote results_054c.json", flush=True)
