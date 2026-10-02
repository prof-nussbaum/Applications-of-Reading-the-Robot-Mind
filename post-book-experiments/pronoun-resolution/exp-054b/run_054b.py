"""exp-054b: token-mean centering refinement (Paul's suggestion, 2026-09-29).

exp-054 subtracted the GLOBAL pool mean before the PCA-64 codec. Paul's
refinement: subtract the mean OF THAT TOKEN -- center each state by its own
word's mean occurrence, so the residual is 'how this occurrence deviates
from the typical occurrence of its word'. This removes the token-specific
carrier (the 'typical pronoun' component that dominated exp-054's residual
neighborhoods: top-5 neighbors were all pronouns) instead of only the
global common mode.

Two variants:
  A: token-demean only, no PCA. r = h - mu_word(h).
  B: token-demean, then global-center, then the PCA-64 codec as in exp-054.
     (Paul's refinement + the exp-054 machinery.)

PREDICTED (before running): variant B reproduces exp-054's visibility bar --
ALL FOUR ceiling referents rank <= 500 by cosine in residual space. Variant A
exploratory (no directional prediction); correct-vs-wrong gaps exploratory in
both variants.

Pool states whose residual is near-zero (rare words where the word-mean ~=
the state itself) are excluded from the search pool; the count is reported.
Test prompts are not in the pool (no self-match issue).
"""
import json
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/exp-054b"
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
N = M.shape[0]
print("pool", N, flush=True)

uniq = sorted(set(words))
w2i = {w: i for i, w in enumerate(uniq)}
wi = torch.tensor([w2i[w] for w in words])
counts = torch.bincount(wi)
mu_w = torch.zeros(len(uniq), M.shape[1]).index_add_(0, wi, M) / counts.unsqueeze(1)
mu_tok = mu_w[wi]                       # per-state token mean
print("distinct words:", len(uniq), "min count:", int(counts.min()),
      "median count:", float(counts.median()), flush=True)
for t in ("he", "it"):
    print("count[%s] = %d" % (t, int(counts[w2i[t]])), flush=True)

# variance accounting (denominator matches exp-054: globally-centered energy)
Mc = M - M.mean(dim=0)
var_tot = float((Mc ** 2).sum())
D_t = M - mu_tok                        # token-demeaned pool (variant A space)
mu_D = D_t.mean(dim=0)
Dc = D_t - mu_D
var_demeaned = float((Dc ** 2).sum())
print("variance removed by token-demeaning: %.4f" % (1 - var_demeaned / var_tot), flush=True)

U, S, V = torch.pca_lowrank(Dc, q=K)
var_cap = float((S ** 2).sum() / (Dc ** 2).sum())
print("top-%d of demeaned captures: %.4f" % (K, var_cap), flush=True)

def codecB(H, mu_q):
    d = H - mu_q - mu_D
    return d - (d @ V) @ V.T

R_B = codecB(M, mu_tok)
print("variant-B residual energy fraction: %.4f" % float((R_B ** 2).sum() / (Dc ** 2).sum()), flush=True)

def kept_normed(X):
    nrm = X.norm(dim=1)
    keep = nrm > 1e-6
    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    return F.normalize(X[keep], dim=1), [words[int(i)] for i in idx], int((~keep).sum())

An, wordsA, exclA = kept_normed(D_t)
Bn, wordsB, exclB = kept_normed(R_B)
print("excluded near-zero residuals: A=%d B=%d" % (exclA, exclB), flush=True)

tpool = torch.load(BV + "/exp-052/pool_052.pt", map_location="cpu", weights_only=False)
by_id = {r["id"]: r for r in tpool}
Mn = F.normalize(M, dim=1)

def rank_of(sims, wl, target):
    order = torch.argsort(sims, descending=True)
    return next(int(i) + 1 for i in range(len(order)) if wl[order[i]] == target)

rows = []
for iid, pron, (c1, c2) in ITEMS:
    rec = by_id[iid]
    ids = rec["ids"]
    pos = ids.index(tok(" " + pron, add_special_tokens=False)["input_ids"][0])
    h = rec["hs"][2][pos].float().unsqueeze(0)
    mu_q = mu_w[w2i[pron]].unsqueeze(0)
    rA_q = h - mu_q
    rB_q = codecB(h, mu_q)
    s_raw = (F.normalize(h, dim=1) @ Mn.T)[0]
    s_A = (F.normalize(rA_q, dim=1) @ An.T)[0]
    s_B = (F.normalize(rB_q, dim=1) @ Bn.T)[0]
    row = {"id": iid, "referent": c1, "distractor": c2,
           "raw_rank": rank_of(s_raw, words, c1),
           "A_rank": rank_of(s_A, wordsA, c1),
           "B_rank": rank_of(s_B, wordsB, c1),
           "A_wrong": rank_of(s_A, wordsA, c2),
           "B_wrong": rank_of(s_B, wordsB, c2)}
    order = torch.argsort(s_B, descending=True)
    row["B_top5"] = [[wordsB[int(i)], round(float(s_B[int(i)]), 3)] for i in order[:5]]
    rows.append(row)
    print("%(id)s %(referent)s raw #%(raw_rank)d A #%(A_rank)d (wrong #%(A_wrong)d) B #%(B_rank)d (wrong #%(B_wrong)d)" % row, flush=True)

json.dump({"K": K,
           "var_removed_by_token_demean": round(1 - var_demeaned / var_tot, 4),
           "topK_of_demeaned": round(var_cap, 4),
           "excluded": {"A": exclA, "B": exclB},
           "rows": rows},
          open(D + "/results_054b.json", "w"), indent=1)
print("wrote results_054b.json", flush=True)
