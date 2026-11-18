import json, torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

EW = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
K = 64
items = json.load(open(EW + "/results_confirmatory.json"))
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
PRONOUNS = ["it", "they", "he", "she"]
pron_mu = {}
for pron in PRONOUNS:
    mask = torch.tensor([w == pron for w in words])
    pron_mu[pron] = M[mask].mean(0)
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval()

def pick(pred):
    for it in items:
        if pred(it):
            return it
    return None

sel = {
  "miss_caught": pick(lambda it: (not it["model_right"]) and (not it["margin_flag_B1"]) and it["rtrm_flag"]),
  "false_alarm": pick(lambda it: it["model_right"] and it["rtrm_flag"]),
  "correct_quiet": pick(lambda it: it["model_right"] and not it["rtrm_flag"] and not it["margin_flag_B1"]),
}
with torch.no_grad():
    for name, it in sel.items():
        ids = tok(it["prompt"])["input_ids"]
        o = lm(torch.tensor([ids]), output_hidden_states=True)
        h = F.normalize(o.hidden_states[12][0].float(), dim=1)
        pid = tok(" " + it["pronoun"], add_special_tokens=False)["input_ids"][0]
        pos = [i for i, x in enumerate(ids) if x == pid][0]
        q1 = h[pos].unsqueeze(0) - pron_mu[it["pronoun"]]
        q2 = coderes(q1)
        sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
        order = torch.argsort(sims, descending=True)
        top20 = [words[order[i]] for i in range(20)]
        def rank_of(t):
            for i in range(len(order)):
                if words[order[i]] == t.lower():
                    return i + 1
        print(f"\n### {name}: item {it['item']}")
        print("sentence:", it["sentence"])
        print("prompt:", it["prompt"])
        print(f"label={it['label']} comp={it['competitor']} pick={it['model_pick']} right={it['model_right']} margin={it['margin_nats']:.3f} flag={it['rtrm_flag']}")
        print("top20:", top20)
        print(f"r_label recomputed={rank_of(it['label'])} saved={it['r_label']}; r_comp recomputed={rank_of(it['competitor'])} saved={it['r_comp']}")
print("\nDONE")
