import json, torch, statistics as st
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

import os
ROBOT_MIND = os.environ.get("ROBOT_MIND", "/home/ecpi-student-05/robot-mind")  # Jetson working tree (data lives outside the repo); override with $ROBOT_MIND

EW = ROBOT_MIND + "/pronoun-battery-v2/path2-early-warning"
PC = ROBOT_MIND + "/pronoun-battery-v2/path2-positive-control"
K = 64
PRONOUNS = ["it", "they", "he", "she"]

items = json.load(open(PC + "/results_control.json"))
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
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval()  # CPU: do not disturb training

# word -> sorted positions in the ranking are computed per item from sims;
# precompute word -> pool row indices for fast rank lookup
from collections import defaultdict
wrows = defaultdict(list)
for i, w in enumerate(words):
    wrows[w].append(i)

def rank_of(target, sims):
    rows = wrows.get(target.lower())
    if not rows:
        return None
    return int(sims[rows].argmax()) if False else None

def best_rank(target, order_pos):
    # order_pos: array mapping pool row -> rank (1-based)
    rows = wrows.get(target.lower())
    if not rows:
        return None
    return int(min(order_pos[r] for r in rows))

# class panels per family: class key = (family, class-of-label)
# number: singular if pronoun it (label singular) etc. Determine each head's class:
head_class = {}
for it in items:
    lab, comp, pron, fam = it["label"], it["competitor"], it["pronoun"], it["family"]
    if fam == "number":
        lab_class = "singular" if pron == "it" else "plural"
        comp_class = "plural" if pron == "it" else "singular"
    else:
        lab_class = "male" if pron == "he" else "female"
        comp_class = "female" if pron == "he" else "male"
    head_class[lab] = (fam, lab_class)
    head_class[comp] = (fam, comp_class)

rows_out = []
with torch.no_grad():
    for it in items:
        pron = it["pronoun"]
        prompt = it["prompt"]
        ids = tok(prompt)["input_ids"]
        o = lm(torch.tensor([ids]), output_hidden_states=True)
        h12 = o.hidden_states[12][0].float()
        h = F.normalize(h12, dim=1)
        pid = tok(" " + pron, add_special_tokens=False)["input_ids"][0]
        pos = [i for i, x in enumerate(ids) if x == pid][0]
        hq = h[pos].unsqueeze(0)
        q1 = hq - pron_mu[pron]
        q2 = coderes(q1)
        sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
        order = torch.argsort(sims, descending=True)
        order_pos = torch.empty(len(order), dtype=torch.long)
        order_pos[order] = torch.arange(1, len(order) + 1)
        fam, lab_class = head_class[it["label"]]
        same = [w for w, (f, c) in head_class.items()
                if f == fam and c == lab_class and w not in (it["label"], it["competitor"])]
        opp = [w for w, (f, c) in head_class.items()
               if f == fam and c != lab_class and w not in (it["label"], it["competitor"])]
        r_same = [best_rank(w, order_pos) for w in same]
        r_opp = [best_rank(w, order_pos) for w in opp]
        r_same = [r for r in r_same if r is not None]
        r_opp = [r for r in r_opp if r is not None]
        rows_out.append({
            "item": it["item"], "family": fam, "pronoun": pron,
            "label": it["label"], "competitor": it["competitor"],
            "r_label": best_rank(it["label"], order_pos),
            "r_comp": best_rank(it["competitor"], order_pos),
            "r_label_saved": it["r_label"], "r_comp_saved": it["r_comp"],
            "median_same_class": st.median(r_same) if r_same else None,
            "median_opp_class": st.median(r_opp) if r_opp else None,
            "n_same": len(r_same), "n_opp": len(r_opp)})

mism = sum(1 for r in rows_out if r["r_label"] != r["r_label_saved"] or r["r_comp"] != r["r_comp_saved"])
print("validation: rank mismatches vs saved results_control.json:", mism, "of", len(rows_out))
for fam in ["number", "gender"]:
    sub = [r for r in rows_out if r["family"] == fam]
    print(f"\n== {fam} (n={len(sub)}) ==")
    print("median r_label:        ", st.median(r["r_label"] for r in sub))
    print("median same-class rank:", st.median(r["median_same_class"] for r in sub))
    print("median opp-class rank: ", st.median(r["median_opp_class"] for r in sub))
    print("median r_comp:         ", st.median(r["r_comp"] for r in sub))
    bl = sum(1 for r in sub if r["r_label"] < r["median_same_class"])
    print(f"label beats same-class median in {bl}/{len(sub)} items (chance ~50%)")
json.dump(rows_out, open(PC + "/class_lift_check.json", "w"), indent=1)
print("\nDONE")
