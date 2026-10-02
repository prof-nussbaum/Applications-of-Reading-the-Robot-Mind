"""exp-055: SME error-spotting on 10 NEW pronoun prompts (Paul's test, 2026-09-29).

Question: the SME uses ONLY the subtracted list. What should they look for
in new prompts to spot issues, before seeing the model's answer?

Method (Paul's framing): lossy compression via PCA. PCA finds the biggest
features that differentiate one token-state from another at the observation
layer (L12). We compress each state into 64 patterns, subtract the
reconstruction, and read the remainder. Query side additionally subtracts
the pronoun's mean state (054c-C2 recipe). torch.manual_seed(0) set before
pca_lowrank (predeclared).

SME indicators (predeclared, applied MECHANICALLY - no human judgment):
  I1 referent presence: label's token rank <= 100 in the subtracted list?
  I2 competitor check: the other sentence noun ranked BELOW the label?
  clean verdict:  CORRECT if I1 and I2 else RISK
  tricky verdict: lean = better-ranked of the two nouns;
                 LEAN-OK if lean == label else LEAN-MISMATCH
Behavior (computed after, reported separately): next-token forced choice
between ' n1' / ' n2' after 'Answer with one word:'. I4 cross-check: the
model's own logits at the pronoun (053 recipe: normalized hs[12] @ W.T).

PREDICTED (before running):
  clean: SME CORRECT on >=4/5; model right on 5/5.
  tricky: lean == first noun on >=4/5 (replicates the 054c first-noun prior
    on new items); on LEAN-MISMATCH items the model overturns correctly on
    >=60% (mismatch flags sensitive, not specific - the cautious hypothesis).
  Primary metric: sensitivity of SME flags for model errors
    = P(flagged | model wrong). If the model makes zero errors the
    error-spotting claim is unevaluable (vacuous); the finding is then
    about lean visibility and the flag false-alarm rate.
Items: see AUDIT_055.txt (seed 1921, stratified draw 5 clean + 5 tricky).
"""
import json, torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/exp-055"
E = "/home/ecpi-student-05/robot-mind/experiments"
B = "/home/ecpi-student-05/robot-mind/pronoun-battery"
K = 64
torch.manual_seed(0)  # predeclared seed for pca_lowrank

# (id, type, pronoun, noun1=first-mentioned, noun2, label, prompt)
ITEMS = [
 ("C1","clean","he","man","boy","man",
  "The man who met the boy was tired and he rested. What does the word he refer to? Answer with one word:"),
 ("C4","clean","it","cat","dog","cat",
  "The cat that the dog chased was tired and it slept. What does the word it refer to? Answer with one word:"),
 ("C5","clean","it","car","truck","car",
  "The car that passed the truck was old and it rattled. What does the word it refer to? Answer with one word:"),
 ("C6","clean","it","truck","car","truck",
  "The truck that the car passed was old and it rattled. What does the word it refer to? Answer with one word:"),
 ("C7","clean","he","man","son","man",
  "The man who saw the son was happy and he smiled. What does the word he refer to? Answer with one word:"),
 ("T1","tricky","he","man","boy","man",
  "The man couldn't lift the boy because he was weak. What does the word he refer to? Answer with one word:"),
 ("T4","tricky","it","dog","cat","cat",
  "The dog bit the cat because it was hurt. What does the word it refer to? Answer with one word:"),
 ("T5","tricky","it","car","truck","car",
  "The car towed the truck because it was powerful. What does the word it refer to? Answer with one word:"),
 ("T6","tricky","it","truck","car","truck",
  "The car towed the truck because it was broken. What does the word it refer to? Answer with one word:"),
 ("T7","tricky","he","man","son","man",
  "The man praised the son because he was proud. What does the word he refer to? Answer with one word:"),
]

tok = GPT2TokenizerFast.from_pretrained("gpt2")
def word_of(i): return tok.decode([i]).strip().lower()
spool = (torch.load(E+"/exp-032d-all-pools/pool_all.pt", map_location="cpu", weights_only=False)
  + torch.load(E+"/exp-037-pronoun-binding/pool_037.pt", map_location="cpu", weights_only=False)
  + torch.load(B+"/exp-047/pool_047.pt", map_location="cpu", weights_only=False))
vecs, words = [], []
for r in spool:
    for p, v in enumerate(r["hs"][2]):
        vecs.append(v.float()); words.append(word_of(r["ids"][p]))
M = torch.stack(vecs)                      # 30099 x 768
mu = M.mean(0)
U, S, V = torch.pca_lowrank(M - mu, q=K)   # V: 768 x 64
def coderes(H):
    Hc = H - mu
    return Hc - (Hc @ V) @ V.T
Rn = F.normalize(coderes(M), dim=1)
is_he = torch.tensor([w == "he" for w in words]); is_it = torch.tensor([w == "it" for w in words])
mu_he, mu_it = M[is_he].mean(0), M[is_it].mean(0)
print("pool %d states; V %s; top-64 var %.4f" % (len(M), tuple(V.shape),
      float((S**2).sum() / ((M-mu)**2).sum())), flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(dev)
W = lm.lm_head.weight.float().cpu()

def rank_of(target, sims):
    order = torch.argsort(sims, descending=True)
    for i in range(len(order)):
        if words[order[i]] == target: return i + 1
    return None

out = []
with torch.no_grad():
    for iid, typ, pron, n1, n2, label, prompt in ITEMS:
        ids = tok(prompt)["input_ids"]
        o = lm(torch.tensor([ids]).to(dev), output_hidden_states=True)
        h12 = o.hidden_states[12][0].float().cpu()
        h = F.normalize(h12, dim=1)                       # pool-identical recipe
        pid = tok(" " + pron, add_special_tokens=False)["input_ids"][0]
        pos = [i for i, x in enumerate(ids) if x == pid][0]
        hq = h[pos].unsqueeze(0)
        q1 = hq - (mu_he if pron == "he" else mu_it)       # 054c-C2
        q1c = q1 - mu
        q2 = q1c - (q1c @ V) @ V.T
        sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
        r1, r2, rL = rank_of(n1, sims), rank_of(n2, sims), rank_of(label, sims)
        comp = n2 if label == n1 else n1
        rC = rank_of(comp, sims)
        lean = n1 if r1 < r2 else n2
        if typ == "clean":
            verdict = "CORRECT" if (rL is not None and rL <= 100 and rL < rC) else "RISK"
        else:
            verdict = "LEAN-OK" if lean == label else "LEAN-MISMATCH"
        # behavior: forced next-token choice
        logits = o.logits[0, -1].float().cpu()
        cands = {}
        for n in (n1, n2):
            tid = tok(" " + n, add_special_tokens=False)["input_ids"]
            assert len(tid) == 1, ("multi-token noun", n, tid)
            cands[n] = float(logits[tid[0]])
        pick = max(cands, key=cands.get)
        top5 = [(tok.decode([i]), round(float(logits[i]), 2))
                for i in torch.topk(logits, 5).indices]
        # I4: model's own logits AT the pronoun (053 recipe)
        plp = h[pos] @ W.T
        t1 = tok(" " + n1, add_special_tokens=False)["input_ids"][0]
        t2 = tok(" " + n2, add_special_tokens=False)["input_ids"][0]
        pron_lean = n1 if float(plp[t1]) > float(plp[t2]) else n2
        out.append({"id": iid, "type": typ, "label": label, "n1": n1, "n2": n2,
                    "r_label": rL, "r_n1": r1, "r_n2": r2, "r_comp": rC,
                    "rtrm_lean": lean, "sme_verdict": verdict,
                    "pron_logit_lean": pron_lean,
                    "pron_logit_margin_nats": round(abs(float(plp[t1]) - float(plp[t2])), 2),
                    "model_pick": pick, "model_right": pick == label,
                    "logit_margin_nats": round(abs(cands[n1] - cands[n2]), 2),
                    "top5_next": top5})
        print("%s %-5s label=%-5s rL=%s rC=%s lean=%-5s verdict=%-12s pick=%-5s right=%s margin=%.2f" %
              (iid, typ, label, rL, rC, lean, verdict, pick, pick == label,
               abs(cands[n1] - cands[n2])), flush=True)

json.dump(out, open(D + "/results_055.json", "w"), indent=1)
print("saved", D + "/results_055.json")
