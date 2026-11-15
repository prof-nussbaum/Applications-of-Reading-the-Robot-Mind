"""Path 2 POSITIVE CONTROL — CEILING PROBE (PROTOCOL_LOCKED.md, 2026-11-15).

*** DIAGNOSTIC ONLY — THIS IS NOT RTRM. ***
A supervised logistic-regression probe is a trained classifier on the
activations: a different method from the RTRM cosine-similarity method.
It appears in this folder ONLY as a ceiling check that bounds the
checklist claim (per the locked protocol's reading). It is never an
RTRM result and never a book claim.

Method:
- Recompute the L12 pronoun states for the 617 DPR confirmatory items
  with the confirmatory recipe (mirror of run_confirmatory.py). The
  state used is the raw query state BEFORE rank computation: the
  per-token L2-normalised L12 hidden state at the pronoun position in
  the formatted question prompt (the state the checklist recipe starts
  from, before pronoun-mean subtraction / PCA residual).
- VERIFICATION GATE: the same forward pass also recomputes r_label and
  r_comp through the full checklist recipe; these must reproduce
  results_confirmatory.json exactly on at least 20 items. If fewer
  match, this script writes ceiling_probe_STOP.json and exits WITHOUT
  training anything.
- Probe: sklearn LogisticRegression(C=1.0), StandardScaler fit on the
  training half only. Split-half BY ITEM, stratified on model_right,
  seed 0: train 308 / test 309 of 617.
- Target: model_right (per protocol). Reported: test AUC for predicting
  model_right, and the equivalent error-prediction AUC (score negated),
  with a bootstrap 95% CI (1000 resamples of the test set, rng seed 0,
  percentile method) on the error-prediction AUC. Train AUC for context.

Outputs (new files, control folder only): probe_states_dpr.pt,
ceiling_probe_results.json (or ceiling_probe_STOP.json on gate failure).
"""
import json, sys, torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

EW = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
PC = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-positive-control"
K = 64
PRONOUNS = ["he", "she", "it", "they"]

items = [json.loads(l) for l in open(EW + "/confirmatory_items.jsonl")]
saved = {(r["split"], r["item"]): r for r in json.load(open(EW + "/results_confirmatory.json"))}
assert len(items) == 617 and len(saved) == 617
print("items:", len(items), flush=True)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
def word_of(i): return tok.decode([i]).strip().lower()

spool = torch.load(EW + "/pool_path2.pt", map_location="cpu", weights_only=False)
vecs, words = [], []
for r in spool:
    for p, v in enumerate(r["hs"][2]):
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
print("device:", dev, flush=True)
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(dev)

def rank_of(target, sims):
    target = target.lower()
    order = torch.argsort(sims, descending=True)
    for i in range(len(order)):
        if words[order[i]] == target: return i + 1
    return None

states, keys, match = [], [], 0
with torch.no_grad():
    for it in items:
        pron, label = it["pronoun"], it["label_head"]
        prompt = "%s What does the word %s refer to? Answer with one word:" % (it["sentence"], pron)
        ids = tok(prompt)["input_ids"]
        sent_len = len(tok(it["sentence"])["input_ids"])
        o = lm(torch.tensor([ids]).to(dev), output_hidden_states=True)
        h12 = o.hidden_states[12][0].float().cpu()
        h = F.normalize(h12, dim=1)
        pid = tok(" " + pron, add_special_tokens=False)["input_ids"][0]
        pos = [i for i, x in enumerate(ids) if x == pid][0]
        assert pos < sent_len
        hq = h[pos]  # raw query state before rank computation
        states.append(hq)
        keys.append((it["split"], it["item"]))
        # verification: full recipe ranks
        q1 = hq.unsqueeze(0) - pron_mu[pron]
        q2 = coderes(q1)
        sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
        comp = it["head2"] if label == it["head1"] else it["head1"]
        rL, rC = rank_of(label, sims), rank_of(comp, sims)
        sv = saved[(it["split"], it["item"])]
        if rL == sv["r_label"] and rC == sv["r_comp"]:
            match += 1

print("rank reproduction: %d / 617 items exact (r_label and r_comp)" % match, flush=True)
if match < 20:
    json.dump({"STOP": "rank reproduction failed", "matched_items": match,
               "required": ">= 20 exact matches of r_label and r_comp vs results_confirmatory.json"},
              open(PC + "/ceiling_probe_STOP.json", "w"), indent=1)
    print("STOP: reproduction gate failed; no probe trained.", flush=True)
    sys.exit(1)

X = torch.stack(states).numpy().astype(np.float64)
y_right = np.array([1 if saved[k]["model_right"] else 0 for k in keys])
torch.save({"keys": keys, "states": torch.stack(states)}, PC + "/probe_states_dpr.pt")
print("saved probe_states_dpr.pt", X.shape, flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

sss = StratifiedShuffleSplit(n_splits=1, test_size=309, random_state=0)
tr, te = next(sss.split(X, y_right))
assert len(tr) == 308 and len(te) == 309, (len(tr), len(te))
sc = StandardScaler().fit(X[tr])
clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(sc.transform(X[tr]), y_right[tr])
p_right_te = clf.predict_proba(sc.transform(X[te]))[:, 1]
p_right_tr = clf.predict_proba(sc.transform(X[tr]))[:, 1]

auc_right = float(roc_auc_score(y_right[te], p_right_te))
y_err = 1 - y_right
s_err = 1.0 - p_right_te
auc_err = float(roc_auc_score(y_err[te], s_err))
auc_err_train = float(roc_auc_score(y_err[tr], 1.0 - p_right_tr))

rng = np.random.default_rng(0)
boots = []
yte, ste = y_err[te], s_err
for b in range(1000):
    idx = rng.integers(0, len(te), len(te))
    if len(np.unique(yte[idx])) < 2:
        continue
    boots.append(roc_auc_score(yte[idx], ste[idx]))
lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
print("test AUC (predict error) = %.3f  95%% CI [%.3f, %.3f]; train AUC = %.3f" % (auc_err, lo, hi, auc_err_train), flush=True)

res = {
  "label": "CEILING PROBE — DIAGNOSTIC ONLY, NOT RTRM. A supervised probe is a trained classifier, a different method from the RTRM cosine-similarity method; it bounds the checklist claim and is never an RTRM result or a book claim.",
  "states": "raw L12 pronoun states (per-token L2-normalised) at the pronoun position in the question-format prompt, recomputed with the confirmatory recipe for the 617 DPR confirmatory items",
  "rank_reproduction_check": {"items_exact_match_r_label_and_r_comp": match, "of": 617, "required": ">= 20"},
  "split": {"method": "StratifiedShuffleSplit, by item, stratified on model_right, seed 0", "train": 308, "test": 309},
  "model": "sklearn LogisticRegression(C=1.0, max_iter=1000), StandardScaler fit on train half only",
  "target": "model_right from results_confirmatory.json",
  "test_auc_predict_right": auc_right,
  "test_auc_predict_error": auc_err,
  "test_auc_predict_error_bootstrap95CI": [lo, hi],
  "bootstrap": "1000 resamples of the test set, rng seed 0, percentile method; %d valid resamples" % len(boots),
  "train_auc_predict_error": auc_err_train,
  "locked_prediction": "test AUC between 0.45 and 0.60 (near chance)",
}
json.dump(res, open(PC + "/ceiling_probe_results.json", "w"), indent=1)
print("saved ceiling_probe_results.json", flush=True)
