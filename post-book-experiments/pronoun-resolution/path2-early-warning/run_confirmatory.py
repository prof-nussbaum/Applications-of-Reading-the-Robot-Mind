"""Path 2 CONFIRMATORY run — scores confirmatory_items.jsonl (617 items,
pilot items excluded permanently and never scored here) under the locked
protocol + Amendments 1-2 + pool-count decision + eligibility ruling
(2026-11-14). Adapted from run_pilot.py (2026-11-14) WITHOUT changing the
scoring recipe; run_pilot.py itself is untouched.

Recipe identical to run_pilot.py:
- Pool: single file pool_path2.pt; states = record hs[2] (L12), labels =
  tok.decode([ids[p]]).strip().lower() (run_055's word_of rule).
- mu = pool mean; PCA-64 via torch.pca_lowrank(M - mu) with
  torch.manual_seed(0) set immediately before the call.
- Residuals: coderes(H) = (H-mu) - ((H-mu) @ V) @ V.T; pool residuals
  L2-normalised (Rn); query residual L2-normalised before the dot product.
- Pronoun mean (run_055 exact handling): mean of the RAW pool states whose
  label equals the pronoun, subtracted from the raw (unit-normalised) query
  state FIRST; the pool-mean subtraction and PCA-residual step are applied
  to the result. (he/she/it/they, same rule as the pilot.)
- Query: L12 state at the pronoun token in the formatted prompt
  "<sentence> What does the word <pronoun> refer to? Answer with one word:",
  pronoun token = first token equal to tok(" "+pronoun); hidden states
  L2-normalised per token exactly as the pool was stored.
- Ranks: best-occurrence rank (1-based) of each head word among pool
  neighbours sorted by descending cosine to the query residual; None if absent.
- Behaviour: next-token logits after the prompt; forced choice between
  " <head1>" and " <head2>" (each asserted a single token); pick = argmax;
  margin in nats signed toward the pick; correct = pick == label head word.
- Primary flag (exp-056 rule): competitor rank < label rank OR label
  rank > 500 OR label absent.
- Secondary verdict (exp-055 stricter rule): CORRECT if label rank <= 100
  AND competitor ranked below label, else RISK.

Deviations from run_pilot.py are RECORDING-ONLY (no scoring change):
- Defaults point at confirmatory_items.jsonl / results_confirmatory.json.
- A hard guard refuses any items file whose name contains "pilot".
- Per item, the subtracted-list top-20 pool words (rank order) are also
  recorded to a sidecar JSON (<out>_top20.json), for the SME rendering of
  RTRM-caught MISS items in the predeclared analysis.
"""
# AUDIT FIX 2026-11-14: rank_of lowercases candidate heads to match lowercased pool labels
# (same case trap as materialise_draw.py; see ELIGIBILITY_RULING_2026-11-14.md). Forced-choice
# token lookups keep surface case on purpose - " Bill" is the candidate token as written.
import json, sys, torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
ITEMS_FILE = sys.argv[1] if len(sys.argv) > 1 else D + "/confirmatory_items.jsonl"
OUT_FILE = sys.argv[2] if len(sys.argv) > 2 else D + "/results_confirmatory.json"
if "pilot" in ITEMS_FILE:
    sys.exit("REFUSING to score a pilot items file in the confirmatory runner: " + ITEMS_FILE)
TOP20_FILE = OUT_FILE[:-5] + "_top20.json" if OUT_FILE.endswith(".json") else OUT_FILE + "_top20.json"
K = 64
PRONOUNS = ["he", "she", "it", "they"]

items = [json.loads(l) for l in open(ITEMS_FILE)]
print("items:", len(items), "from", ITEMS_FILE, flush=True)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
def word_of(i): return tok.decode([i]).strip().lower()

spool = torch.load(D + "/pool_path2.pt", map_location="cpu", weights_only=False)
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
    # CASE FIX 2026-11-14 (ELIGIBILITY_RULING_2026-11-14): pool labels in
    # `words` are lowercased (word_of: strip().lower(), run_055 convention),
    # while item heads carry surface case ("Bill"). Lowercase the target
    # before matching, else capitalised heads always rank absent.
    target = target.lower()
    order = torch.argsort(sims, descending=True)
    for i in range(len(order)):
        if words[order[i]] == target: return i + 1
    return None

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
        rec = {"split": it["split"], "item": it["item"], "sentence": it["sentence"],
               "prompt": prompt, "pronoun": pron, "head1": n1, "head2": n2,
               "label": label, "competitor": comp, "clue_class": it["clue_class"],
               "r_label": rL, "r_head1": r1, "r_head2": r2, "r_comp": rC,
               "rtrm_lean": lean, "rtrm_flag": bool(flag), "secondary_verdict": secondary,
               "logit_head1": cands[n1], "logit_head2": cands[n2],
               "model_pick": pick, "model_right": pick == label,
               "margin_nats": margin, "margin_flag_B1": bool(margin < 0.35)}
        out.append(rec)
        top20_out.append({"split": it["split"], "item": it["item"], "top20": top20_words})
        print("%s-%s label=%-12s rL=%s rC=%s lean=%-12s flag=%s 055=%-7s pick=%-12s right=%s margin=%.4f" %
              (it["split"], it["item"], label, rL, rC, lean, flag, secondary, pick,
               pick == label, margin), flush=True)

json.dump(out, open(OUT_FILE, "w"), indent=1)
print("saved", OUT_FILE, "records:", len(out), flush=True)
json.dump(top20_out, open(TOP20_FILE, "w"), indent=1)
print("saved", TOP20_FILE, "records:", len(top20_out), flush=True)
