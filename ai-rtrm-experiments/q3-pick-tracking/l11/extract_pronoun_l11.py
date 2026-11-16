"""Q3 Amendment 1, stage 1: PRONOUN-position neighbour-list readout at
LAYER 11 for the 617 confirmatory items, using the ORIGINAL confirmatory
recipe (run_confirmatory.py) WITH pronoun-mean subtraction. The Q3
no-subtraction deviation applied only to the answer position and is NOT
used here. No behaviour is scored; no answer-position state is read.

Recipe (identical to run_confirmatory.py except layer 12 -> 11 and the
pool file): pool = pool_L11.pt (hs[0]); mu = pool mean; PCA-64 via
torch.pca_lowrank with torch.manual_seed(0) immediately before;
coderes(H) = (H-mu) - ((H-mu) @ V) @ V.T; pool residuals L2-normalised;
pronoun mean = mean of RAW pool states whose word label equals the
pronoun, subtracted from the raw unit-normalised query state FIRST;
query = hs[11] at the pronoun token (first token equal to
tok(" "+pronoun)) of the exact confirmatory prompt, per-token
L2-normalised exactly as the pool was stored; ranks = best-occurrence
1-based rank of each head (targets lowercased, pool labels lowercased);
lean = head1 if r1 is not None and (r2 is None or r1 < r2) else head2.
Output: pronoun_l11_ranks.json (ranks + lean per item; picks are joined
from banked records by the scoring script, never recomputed).
"""
import json, torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

EW = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
L11 = "/home/ecpi-student-05/robot-mind/q3-pick-tracking/l11"
K = 64
PRONOUNS = ["he", "she", "it", "they"]

items = [json.loads(l) for l in open(EW + "/confirmatory_items.jsonl")]
print("items:", len(items), flush=True)
assert len(items) == 617

tok = GPT2TokenizerFast.from_pretrained("gpt2")
def word_of(i): return tok.decode([i]).strip().lower()

spool = torch.load(L11 + "/pool_L11.pt", map_location="cpu", weights_only=False)
vecs, words = [], []
for r in spool:
    for p, v in enumerate(r["hs"][0]):  # hs[0] = layer 11
        vecs.append(v.float()); words.append(word_of(r["ids"][p]))
del spool
M = torch.stack(vecs)
mu = M.mean(0)
torch.manual_seed(0)  # predeclared seed for pca_lowrank (run_055 / protocol)
U, S, V = torch.pca_lowrank(M - mu, q=K)
def coderes(H):
    Hc = H - mu
    return Hc - (Hc @ V) @ V.T
Rn = F.normalize(coderes(M), dim=1)
pron_mu = {}
for pron in PRONOUNS:
    mask = torch.tensor([w == pron for w in words])
    pron_mu[pron] = M[mask].mean(0)
    print("pronoun mean %-5s from %d pool occurrences" % (pron, int(mask.sum())), flush=True)
print("pool %d states; V %s; top-64 var %.4f" % (len(M), tuple(V.shape),
      float((S**2).sum() / ((M - mu)**2).sum())), flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, flush=True)
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(dev)

def rank_of(target, sims):
    target = target.lower()
    order = torch.argsort(sims, descending=True)
    for i in range(len(order)):
        if words[order[i]] == target: return i + 1
    return None

out = []
with torch.no_grad():
    for it in items:
        pron, n1, n2, label = it["pronoun"], it["head1"], it["head2"], it["label_head"]
        prompt = "%s What does the word %s refer to? Answer with one word:" % (it["sentence"], pron)
        ids = tok(prompt)["input_ids"]
        sent_len = len(tok(it["sentence"])["input_ids"])
        o = lm(torch.tensor([ids]).to(dev), output_hidden_states=True)
        h = F.normalize(o.hidden_states[11][0].float().cpu(), dim=1)  # pool-identical recipe
        pid = tok(" " + pron, add_special_tokens=False)["input_ids"][0]
        pos = [i for i, x in enumerate(ids) if x == pid][0]
        assert pos < sent_len, ("pronoun position outside sentence", it, pos, sent_len)
        q1 = h[pos].unsqueeze(0) - pron_mu[pron]  # pronoun mean first (run_055 exact)
        q2 = coderes(q1)                          # then pool-mean subtraction + PCA residual
        sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
        r1, r2, rL = rank_of(n1, sims), rank_of(n2, sims), rank_of(label, sims)
        comp = n2 if label == n1 else n1
        rC = rank_of(comp, sims)
        lean = n1 if (r1 is not None and (r2 is None or r1 < r2)) else n2
        out.append({"split": it["split"], "item": it["item"], "sentence": it["sentence"],
                    "pronoun": pron, "head1": n1, "head2": n2, "label": label,
                    "competitor": comp, "r_label": rL, "r_head1": r1, "r_head2": r2,
                    "r_comp": rC, "lean_l11_pronoun": lean})
        if len(out) % 100 == 0:
            print("extracted %d/617" % len(out), flush=True)

json.dump(out, open(L11 + "/pronoun_l11_ranks.json", "w"), indent=1)
print("saved pronoun_l11_ranks.json records:", len(out), flush=True)
print("EXTRACTION_DONE", flush=True)
