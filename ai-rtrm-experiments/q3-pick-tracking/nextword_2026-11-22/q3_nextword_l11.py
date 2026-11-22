"""Q3 next-word re-analysis (R-012 s.4 / R-013 sequence; exploratory,
no verdict changes).

The banked Q3 Amendment-1 stage-2 run (l11/extract_answer_l11.py)
read the 617 confirmatory items at the ANSWER position (last prompt
token, hs[11]) against pool_L11.pt and labelled every pool neighbour
by its OWN token. This re-analysis repeats that extraction exactly
(same prompts, same pool refit, PCA-64 seed 0, NO pronoun-mean
subtraction) and labels each pool neighbour instead by the token
that FOLLOWED it in its source sentence (variant N labelling, as in
the E1b redesign). Sentence-final pool states have no next token;
they keep their rank positions and never match a head word.

Built-in gate: the current-token ranks and leans computed here must
reproduce l11/answer_l11_ranks.json on all 617 items, or the run
aborts before scoring anything.

Scoring (same footing as R-003 s.2, verified against the banked
records before this run: current-token list BA 0.6681, lens L11
0.9277, fixed rules 0.500): per-item balanced accuracy against the
question-format pick over all 617 items, positive class = the pick
is head2. Reported beside it: the same BA for the banked lens L11
leans and the fixed rules, and the native Q3 strata (stage2_common
pair_stats) for the variant-N lean.

Outputs (new folder q3-pick-tracking/nextword_2026-11-22/):
nextword_ranks.json (per item, both labellings), answer_states_l11.pt
(the raw answer-position states, banked this time so no future
relabelling needs re-extraction), nextword_scores.json.
"""
import json, os, sys
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

EW = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
Q3 = "/home/ecpi-student-05/robot-mind/q3-pick-tracking"
L11 = Q3 + "/l11"
OUT = Q3 + "/nextword_2026-11-22"
K = 64
sys.path.insert(0, L11)

items = [json.loads(l) for l in open(EW + "/confirmatory_items.jsonl")]
banked = {(r["split"], r["item"]): r for r in
          json.load(open(L11 + "/answer_l11_ranks.json"))}
assert len(items) == 617 and len(banked) == 617

tok = GPT2TokenizerFast.from_pretrained("gpt2")
def word_of(i): return tok.decode([i]).strip().lower()

spool = torch.load(L11 + "/pool_L11.pt", map_location="cpu", weights_only=False)
vecs, words, next_words = [], [], []
for r in spool:
    ids = r["ids"]
    for p, v in enumerate(r["hs"][0]):
        vecs.append(v.float())
        words.append(word_of(ids[p]))
        next_words.append(word_of(ids[p + 1]) if p + 1 < len(ids) else None)
del spool
M = torch.stack(vecs)
assert len(M) == 47499
mu = M.mean(0)
torch.manual_seed(0)
U, S, V = torch.pca_lowrank(M - mu, q=K)
def coderes(H):
    Hc = H - mu
    return Hc - (Hc @ V) @ V.T
Rn = F.normalize(coderes(M), dim=1)
top64 = float((S**2).sum() / ((M - mu)**2).sum())
print("pool %d states; top-64 var %.4f (stage 1: 0.5026)" % (len(M), top64), flush=True)
assert abs(top64 - 0.5026) < 0.001, top64
del M, vecs

def rank_of(target, order, labels):
    target = target.lower()
    for i in range(len(order)):
        if labels[order[i]] == target:
            return i + 1
    return None

def lean_of(r1, r2, n1, n2):
    return n1 if (r1 is not None and (r2 is None or r1 < r2)) else n2

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, flush=True)
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(dev)

os.makedirs(OUT, exist_ok=False)
out, states = [], {}
with torch.no_grad():
    for n, it in enumerate(items):
        pron, n1, n2 = it["pronoun"], it["head1"], it["head2"]
        prompt = "%s What does the word %s refer to? Answer with one word:" % (it["sentence"], pron)
        ids = tok(prompt)["input_ids"]
        o = lm(torch.tensor([ids]).to(dev), output_hidden_states=True)
        h_raw = o.hidden_states[11][0, -1].float().cpu()
        states[(it["split"], it["item"])] = h_raw
        hq = F.normalize(h_raw, dim=0).unsqueeze(0)
        q = F.normalize(coderes(hq), dim=1)
        sims = (q @ Rn.T)[0]
        order = torch.argsort(sims, descending=True).tolist()
        b = banked[(it["split"], it["item"])]
        r1, r2 = rank_of(n1, order, words), rank_of(n2, order, words)
        lean_cur = lean_of(r1, r2, n1, n2)
        assert r1 == b["r_head1"] and r2 == b["r_head2"] and lean_cur == b["lean_l11_answer"], \
            f"GATE FAIL on {(it['split'], it['item'])}: {(r1, r2, lean_cur)} vs banked {(b['r_head1'], b['r_head2'], b['lean_l11_answer'])}"
        nr1, nr2 = rank_of(n1, order, next_words), rank_of(n2, order, next_words)
        out.append({"split": it["split"], "item": it["item"], "sentence": it["sentence"],
                    "pronoun": pron, "head1": n1, "head2": n2, "label": b["label"],
                    "competitor": b["competitor"], "question_pick": b["question_pick"],
                    "sub_pick": b["sub_pick"],
                    "cur_r_head1": r1, "cur_r_head2": r2, "lean_current": lean_cur,
                    "next_r_head1": nr1, "next_r_head2": nr2,
                    "lean_next": lean_of(nr1, nr2, n1, n2)})
        if (n + 1) % 100 == 0:
            print("extracted %d/617" % (n + 1), flush=True)
print("GATE: current-token ranks reproduce answer_l11_ranks.json on 617/617", flush=True)

torch.save(states, OUT + "/answer_states_l11.pt")
json.dump(out, open(OUT + "/nextword_ranks.json", "w"), indent=1)

q3 = {(r["split"], r["item"]): r for r in json.load(open(Q3 + "/q3_results.json"))}
def BA(records, lean_key):
    tp = fn = tn = fp = 0
    for r in records:
        pos = (r["question_pick"] == r["head2"])
        pred = (r[lean_key] == r["head2"])
        if pos and pred: tp += 1
        elif pos: fn += 1
        elif pred: fp += 1
        else: tn += 1
    return {"ba": round(((tp / (tp + fn)) + (tn / (tn + fp))) / 2, 4),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}
merged = [dict(r, lean_lens=q3[(r["split"], r["item"])]["lens"]["11"]["lean"],
               lean_h2=r["head2"]) for r in out]
scores = {"readout": "L11 answer position, BA vs question-format pick, all 617 items",
          "variantN_list": BA(merged, "lean_next"),
          "current_token_list": BA(merged, "lean_current"),
          "lens_L11": BA(merged, "lean_lens"),
          "fixed_always_head2": BA(merged, "lean_h2")}

import stage2_common as sc
leanmap = {(r["split"], r["item"]): r["lean_next"] for r in out}
scores["variantN_strata_vs_question_pick"] = sc.pair_stats(
    lambda s, i: leanmap[(s, i)], sc.q_pick)
leanmap_cur = {(r["split"], r["item"]): r["lean_current"] for r in out}
scores["current_token_strata_vs_question_pick"] = sc.pair_stats(
    lambda s, i: leanmap_cur[(s, i)], sc.q_pick)
json.dump(scores, open(OUT + "/nextword_scores.json", "w"), indent=1)
print(json.dumps(scores, indent=1), flush=True)
print("NEXTWORD COMPLETE", flush=True)
