"""Q3 pick-tracking extraction — LOCKED protocol (PROTOCOL_LOCKED.md,
ai-rtrm-experiments/q3-pick-tracking/, locked 2026-11-15).

Reads the 617 confirmatory items at the ANSWER position (last prompt token,
the final ':') with GPT-2, one forward pass per item:
- Neighbour-list readout at layers 6 and 12 only (banked pool_path2.pt
  stores layers 0/6/12): the run_confirmatory.py residual recipe with the
  protocol's ONE deviation — NO pronoun-mean subtraction. Per layer:
  mu = pool mean; torch.manual_seed(0) immediately before
  torch.pca_lowrank(M - mu, q=64); residual = (H-mu) - ((H-mu)@V)@V.T;
  query L2-normalised first (pool-identical), residual L2-normalised,
  cosine vs L2-normalised pool residuals; best-occurrence rank (1-based)
  of each head word; lean = run_confirmatory rule.
- Lens readout at layers 6-12: subst_rescore.py computation — manual ln_f
  ((h - mean)/sqrt(var(unbiased=False)+1e-5) * w + b), lm_head, rank of
  each head's single token " <head>"; lean = head with the lower rank.
Behaviour is NOT re-scored: sub_pick and question_pick come from the
banked records. Writes q3_results.json next to this script.
"""
import json, torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

EW = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
Q3 = "/home/ecpi-student-05/robot-mind/q3-pick-tracking"
K = 64
POOL_IDX = {6: 1, 12: 2}   # pool record hs = [layer0, layer6, layer12]
LENS_LAYERS = [6, 7, 8, 9, 10, 11, 12]

items = [json.loads(l) for l in open(EW + "/confirmatory_items.jsonl")]
res = {(r["split"], r["item"]): r for r in json.load(open(EW + "/results_confirmatory.json"))}
sub = {(s["split"], s["item"]): s for s in json.load(open(EW + "/subst_rescore_617.json"))}
print("items:", len(items), flush=True)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
def word_of(i): return tok.decode([i]).strip().lower()

# ---- pool structures per list layer (refit per layer, seed 0) ----
spool = torch.load(EW + "/pool_path2.pt", map_location="cpu", weights_only=False)
words = None
pool = {}
for layer, idx in POOL_IDX.items():
    vecs, ws = [], []
    for r in spool:
        for p, v in enumerate(r["hs"][idx]):
            vecs.append(v.float()); ws.append(word_of(r["ids"][p]))
    if words is None:
        words = ws
    assert ws == words
    M = torch.stack(vecs)
    mu = M.mean(0)
    torch.manual_seed(0)  # predeclared seed, immediately before the PCA call
    U, S, V = torch.pca_lowrank(M - mu, q=K)
    Hc = M - mu
    Rn = F.normalize(Hc - (Hc @ V) @ V.T, dim=1)
    pool[layer] = (mu, V, Rn)
    print("pool layer %d: %d states, top-64 var %.4f" %
          (layer, len(M), float((S**2).sum() / ((M - mu)**2).sum())), flush=True)
    del M, Hc, vecs
del spool

def list_ranks(layer, h_raw):
    mu, V, Rn = pool[layer]
    hq = F.normalize(h_raw, dim=0).unsqueeze(0)      # pool-identical normalisation
    Hc = hq - mu                                     # pool-mean subtraction (NO pronoun mean)
    q = F.normalize(Hc - (Hc @ V) @ V.T, dim=1)
    sims = (q @ Rn.T)[0]
    order = torch.argsort(sims, descending=True)
    return order

def ranks_from_order(order, targets):
    out = {t: None for t in targets}
    left = len(targets)
    for i in range(len(order)):
        w = words[order[i]]
        if w in out and out[w] is None:
            out[w] = i + 1
            left -= 1
            if left == 0:
                break
    return out

# ---- model + lens parameters ----
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, flush=True)
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(dev)
lw = lm.transformer.ln_f.weight.detach().float().cpu()
lb = lm.transformer.ln_f.bias.detach().float().cpu()
W = lm.lm_head.weight.detach().float().cpu()

def head_tok(name):
    ids = tok(" " + name, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, ("multi-token head", name, ids)
    return ids[0]

out = []
with torch.no_grad():
    for n, it in enumerate(items):
        pron, n1, n2, label = it["pronoun"], it["head1"], it["head2"], it["label_head"]
        comp = n2 if label == n1 else n1
        prompt = "%s What does the word %s refer to? Answer with one word:" % (it["sentence"], pron)
        ids = tok(prompt)["input_ids"]
        o = lm(torch.tensor([ids]).to(dev), output_hidden_states=True)
        t1, t2 = head_tok(n1), head_tok(n2)
        rec = {"split": it["split"], "item": it["item"], "sentence": it["sentence"],
               "prompt": prompt, "pronoun": pron, "head1": n1, "head2": n2,
               "label": label, "competitor": comp,
               "sub_pick": label if sub[(it["split"], it["item"])]["sub_right"] else comp,
               "question_pick": res[(it["split"], it["item"])]["model_pick"],
               "list": {}, "lens": {}}
        for layer in LENS_LAYERS:
            h = o.hidden_states[layer][0, -1].float().cpu()  # answer position: last token
            hn = (h - h.mean()) / (h.var(unbiased=False) + 1e-5).sqrt() * lw + lb
            logits = W @ hn
            rk1 = int((logits > logits[t1]).sum()) + 1
            rk2 = int((logits > logits[t2]).sum()) + 1
            rec["lens"][str(layer)] = {"rank_head1": rk1, "rank_head2": rk2,
                                       "lean": n1 if rk1 < rk2 else n2}
            if layer in POOL_IDX:
                order = list_ranks(layer, h)
                rr = ranks_from_order(order, {n1.lower(), n2.lower()})
                r1, r2 = rr[n1.lower()], rr[n2.lower()]
                lean = n1 if (r1 is not None and (r2 is None or r1 < r2)) else n2
                rec["list"][str(layer)] = {"r_head1": r1, "r_head2": r2, "lean": lean}
        out.append(rec)
        if (n + 1) % 50 == 0 or n == 0:
            print("extracted %d/%d" % (n + 1, len(items)), flush=True)

json.dump(out, open(Q3 + "/q3_results.json", "w"), indent=1)
print("saved", Q3 + "/q3_results.json", "records:", len(out), flush=True)
print("EXTRACTION_DONE", flush=True)
