"""Path 2 POSITIVE CONTROL — scoring run (PROTOCOL_LOCKED.md, 2026-11-15).

Scores items_control.jsonl (120 items: 60 number-agreement, 60
gender-agreement; audit PASSED, seed 0910) in BOTH locked prompt formats.

QUESTION FORMAT: recipe copied from path2-early-warning/run_confirmatory.py
WITHOUT change:
- Pool: pool_path2.pt; states = record hs[2] (L12), labels =
  tok.decode([ids[p]]).strip().lower().
- mu = pool mean; PCA-64 via torch.pca_lowrank(M - mu), torch.manual_seed(0)
  immediately before the call.
- coderes(H) = (H-mu) - ((H-mu) @ V) @ V.T; pool residuals L2-normalised;
  query residual L2-normalised before the dot product.
- Pronoun mean: mean of RAW pool states whose label equals the pronoun,
  subtracted from the raw (unit-normalised) query state FIRST; then
  pool-mean subtraction + PCA residual.
- Query: L12 state at the pronoun token in the formatted prompt
  "<sentence> What does the word <pronoun> refer to? Answer with one word:",
  pronoun token = first token equal to tok(" "+pronoun); hidden states
  L2-normalised per token exactly as the pool was stored.
- Ranks: best-occurrence rank (1-based) of each head word; None if absent.
  rank_of lowercases the target (ELIGIBILITY_RULING_2026-11-14 case fix).
- Behaviour: next-token logits after the prompt; forced choice between
  " <head1>" and " <head2>" (each asserted a single token); pick = argmax;
  margin in nats signed toward the pick.
- Primary flag (exp-056 rule): competitor rank < label rank OR label
  rank > 500 OR label absent.
- Secondary verdict (exp-055 rule): CORRECT if label rank <= 100 AND
  competitor ranked below label, else RISK.

SUBSTITUTION FORMAT (047a method, as implemented in exp-048/run_048.py,
exp-049, exp-050b — the battery's substitution method of record; the
protocol's "exp-052 style" label refers to this battery convention):
- subst(sentence, pronoun, noun): re.sub(r"\\b"+pronoun+r"\\b",
  "the "+noun, sentence, count=1, flags=IGNORECASE) — first pronoun
  occurrence replaced by "the <candidate>".
- logprob(s): TOTAL log-probability of the full substituted sentence =
  sum over all token positions of log p(token | prefix).
- Pick = candidate with the higher total log-probability.
Declared here because the protocol names the format but not the code:
this is exactly run_048.py's subst()/logprob() pair, unmodified.

Outputs (new files, control folder only): results_control.json,
results_control_top20.json.
"""
import json, re, sys, torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

EW = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
PC = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-positive-control"
ITEMS_FILE = PC + "/items_control.jsonl"
OUT_FILE = PC + "/results_control.json"
TOP20_FILE = PC + "/results_control_top20.json"
K = 64
PRONOUNS = ["he", "she", "it", "they"]

items = [json.loads(l) for l in open(ITEMS_FILE)]
assert len(items) == 120, len(items)
print("items:", len(items), "from", ITEMS_FILE, flush=True)

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

def logprob(s):
    # 047a method, verbatim from exp-048/run_048.py: total log-probability
    ids = tok(s, return_tensors="pt")["input_ids"].to(dev)
    with torch.no_grad():
        lp = torch.log_softmax(lm(ids).logits[0, :-1], dim=-1)
    tgt = ids[0, 1:]
    return float(lp[torch.arange(len(tgt)), tgt].sum())

def subst(sentence, pronoun, noun):
    # 047a method, verbatim from exp-048/run_048.py
    return re.sub(r"\b" + re.escape(pronoun) + r"\b", "the " + noun,
                  sentence, count=1, flags=re.IGNORECASE)

out = []
top20_out = []
with torch.no_grad():
    for it in items:
        pron, n1, n2, label = it["pronoun"], it["head1"], it["head2"], it["label_head"]
        prompt = "%s What does the word %s refer to? Answer with one word:" % (it["sentence"], pron)
        ids = tok(prompt)["input_ids"]
        sent_len = len(tok(it["sentence"])["input_ids"])
        o = lm(torch.tensor([ids]).to(dev), output_hidden_states=True)
        h12 = o.hidden_states[12][0].float().cpu()
        h = F.normalize(h12, dim=1)  # pool-identical recipe
        pid = tok(" " + pron, add_special_tokens=False)["input_ids"][0]
        pos = [i for i, x in enumerate(ids) if x == pid][0]
        assert pos < sent_len, ("pronoun position outside sentence", it, pos, sent_len)
        hq = h[pos].unsqueeze(0)
        q1 = hq - pron_mu[pron]              # pronoun mean first (run_055 exact)
        q2 = coderes(q1)                     # then pool-mean subtraction + PCA residual
        sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
        _order = torch.argsort(sims, descending=True)
        top20_words = [words[_order[i]] for i in range(20)]
        r1, r2, rL = rank_of(n1, sims), rank_of(n2, sims), rank_of(label, sims)
        comp = n2 if label == n1 else n1
        rC = rank_of(comp, sims)
        lean = n1 if (r1 is not None and (r2 is None or r1 < r2)) else n2
        flag = (rL is None) or (rL > 500) or (rC is not None and rC < rL)
        secondary = "CORRECT" if (rL is not None and rL <= 100 and rC is not None and rC > rL) else "RISK"
        logits = o.logits[0, -1].float().cpu()
        cands = {}
        for n in (n1, n2):
            tid = tok(" " + n, add_special_tokens=False)["input_ids"]
            assert len(tid) == 1, ("multi-token noun", n, tid)
            cands[n] = float(logits[tid[0]])
        pick = max(cands, key=cands.get)
        other = n2 if pick == n1 else n1
        margin = cands[pick] - cands[other]  # signed toward the pick
        # --- substitution format (047a method) ---
        s1 = subst(it["sentence"], pron, n1)
        s2 = subst(it["sentence"], pron, n2)
        lp1, lp2 = logprob(s1), logprob(s2)
        sub_pick = n1 if lp1 > lp2 else n2
        rec = {"split": it["split"], "item": it["item"], "sentence": it["sentence"],
               "prompt": prompt, "pronoun": pron, "head1": n1, "head2": n2,
               "label": label, "competitor": comp, "clue_class": None,
               "family": it["family"],
               "r_label": rL, "r_head1": r1, "r_head2": r2, "r_comp": rC,
               "rtrm_lean": lean, "rtrm_flag": bool(flag), "secondary_verdict": secondary,
               "logit_head1": cands[n1], "logit_head2": cands[n2],
               "model_pick": pick, "model_right": pick == label,
               "margin_nats": margin, "margin_flag_B1": bool(margin < 0.35),
               "sub_sentence_head1": s1, "sub_sentence_head2": s2,
               "sub_lp_head1": lp1, "sub_lp_head2": lp2,
               "sub_pick": sub_pick, "sub_right": sub_pick == label}
        out.append(rec)
        top20_out.append({"split": it["split"], "item": it["item"], "top20": top20_words})
        print("%s-%s [%s] label=%-10s rL=%s rC=%s lean=%-10s flag=%s pick=%-10s right=%s margin=%.4f | sub pick=%-10s right=%s" %
              (it["split"], it["item"], it["family"], label, rL, rC, lean, flag, pick,
               pick == label, margin, sub_pick, sub_pick == label), flush=True)

json.dump(out, open(OUT_FILE, "w"), indent=1)
print("saved", OUT_FILE, "records:", len(out), flush=True)
json.dump(top20_out, open(TOP20_FILE, "w"), indent=1)
print("saved", TOP20_FILE, "records:", len(top20_out), flush=True)
