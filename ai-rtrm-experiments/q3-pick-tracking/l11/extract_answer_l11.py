"""Q3 Amendment 1, stage 2: ANSWER-position neighbour-list readout at
LAYER 11 for the 617 confirmatory items, against pool_L11.pt, with the
Q3 answer-position recipe (run_q3.py) — the protocol's ONE deviation:
NO pronoun-mean subtraction. Per the locked supplement
(AMENDMENT_1_SUPPLEMENT.md):

- prompt/tokenisation identical to run_q3.py and stage 1: the exact
  confirmatory prompt, GPT2TokenizerFast, one forward pass per item;
- answer position = hidden_states[11] at the LAST prompt token;
- pool structures refit from the L11 pool: mu = pool mean;
  torch.manual_seed(0) immediately before torch.pca_lowrank(M - mu,
  q=64); pool residuals L2-normalised;
- query: unit-normalise the raw state (pool-identical), subtract the
  POOL mean only, PCA-64 residual, L2-normalise, cosine vs pool
  residuals; best-occurrence rank (1-based) of each head word;
  lean = run_confirmatory rule (head with the lower rank).
Behaviour is NOT re-scored: question_pick / sub_pick are joined from
the banked records. Output: answer_l11_ranks.json.
"""
import json, torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

EW = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
L11 = "/home/ecpi-student-05/robot-mind/q3-pick-tracking/l11"
K = 64

items = [json.loads(l) for l in open(EW + "/confirmatory_items.jsonl")]
res = {(r["split"], r["item"]): r for r in json.load(open(EW + "/results_confirmatory.json"))}
sub = {(s["split"], s["item"]): s for s in json.load(open(EW + "/subst_rescore_617.json"))}
print("items:", len(items), flush=True)
assert len(items) == 617

tok = GPT2TokenizerFast.from_pretrained("gpt2")
def word_of(i): return tok.decode([i]).strip().lower()

# ---- L11 pool structures (refit, seed 0; identical construction to stage 1) ----
spool = torch.load(L11 + "/pool_L11.pt", map_location="cpu", weights_only=False)
vecs, words = [], []
for r in spool:
    for p, v in enumerate(r["hs"][0]):  # hs[0] = layer 11
        vecs.append(v.float()); words.append(word_of(r["ids"][p]))
del spool
M = torch.stack(vecs)
assert len(M) == 47499, len(M)
mu = M.mean(0)
torch.manual_seed(0)  # predeclared seed, immediately before the PCA call
U, S, V = torch.pca_lowrank(M - mu, q=K)
def coderes(H):
    Hc = H - mu
    return Hc - (Hc @ V) @ V.T
Rn = F.normalize(coderes(M), dim=1)
top64 = float((S**2).sum() / ((M - mu)**2).sum())
print("pool %d states; V %s; top-64 var %.4f (stage 1: 0.5026)" % (len(M), tuple(V.shape), top64), flush=True)
assert abs(top64 - 0.5026) < 0.001, top64
del M, vecs

def rank_of(target, sims):
    target = target.lower()
    order = torch.argsort(sims, descending=True)
    for i in range(len(order)):
        if words[order[i]] == target: return i + 1
    return None

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, flush=True)
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(dev)

out = []
with torch.no_grad():
    for n, it in enumerate(items):
        pron, n1, n2, label = it["pronoun"], it["head1"], it["head2"], it["label_head"]
        comp = n2 if label == n1 else n1
        prompt = "%s What does the word %s refer to? Answer with one word:" % (it["sentence"], pron)
        ids = tok(prompt)["input_ids"]
        o = lm(torch.tensor([ids]).to(dev), output_hidden_states=True)
        h_raw = o.hidden_states[11][0, -1].float().cpu()  # answer position: last token
        hq = F.normalize(h_raw, dim=0).unsqueeze(0)       # pool-identical normalisation
        q = F.normalize(coderes(hq), dim=1)               # pool-mean subtraction ONLY (no pronoun mean)
        sims = (q @ Rn.T)[0]
        r1, r2 = rank_of(n1, sims), rank_of(n2, sims)
        rL, rC = rank_of(label, sims), rank_of(comp, sims)
        lean = n1 if (r1 is not None and (r2 is None or r1 < r2)) else n2
        out.append({"split": it["split"], "item": it["item"], "sentence": it["sentence"],
                    "pronoun": pron, "head1": n1, "head2": n2, "label": label,
                    "competitor": comp,
                    "question_pick": res[(it["split"], it["item"])]["model_pick"],
                    "sub_pick": label if sub[(it["split"], it["item"])]["sub_right"] else comp,
                    "r_label": rL, "r_head1": r1, "r_head2": r2, "r_comp": rC,
                    "lean_l11_answer": lean})
        if (n + 1) % 100 == 0:
            print("extracted %d/617" % (n + 1), flush=True)

json.dump(out, open(L11 + "/answer_l11_ranks.json", "w"), indent=1)
print("saved answer_l11_ranks.json records:", len(out), flush=True)
print("EXTRACTION_DONE", flush=True)
