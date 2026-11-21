"""Diagnostic (NOT a protocol change): where does the godparent name
rank in the RTRM neighbour list if the query state is taken at other
positions of the same probe? Fresh seed-1 model and its own pool.
Construction identical to build_e1b_fresh.py (its functions imported
verbatim, including the 'is'-mean subtraction at every position --
only the position varies). Positions: final token ('is', the built
readout), the godchild's last name token, the possessive token."""
import json, sys, torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, "/home/ecpi-student-05/robot-mind/exp-201-exposure-fresh")
from build_e1b_fresh import residual_setup, coderes, word_of, rank_of

EXP = "/home/ecpi-student-05/robot-mind/exp-201-exposure-fresh"
OUT = f"{EXP}/blind-test-e1b"
SEED = 1
tok = GPT2TokenizerFast.from_pretrained("gpt2")

recs = torch.load(f"{OUT}/pool_e1b_s{SEED}.pt", map_location="cpu", weights_only=False)
vecs, words = [], []
for r in recs:
    for p, v in enumerate(r["hs11"]):
        vecs.append(v.float())
        words.append(word_of(tok, r["ids"][p]))
M = torch.stack(vecs)
mu, V = residual_setup(M)
Rn = F.normalize(coderes(M, mu, V), dim=1)
is_mu = M[torch.tensor([w == "is" for w in words])].mean(0)

model = GPT2LMHeadModel.from_pretrained(f"{EXP}/models/expD_s{SEED}").eval()
ds = json.load(open(f"{EXP}/dataset_fresh.json"))
pairs = [tuple(p) for p in ds["pairs"]]
rev_idx = set(ds["rev_idx"])
built = {x["index"]: x for x in json.load(open(f"{OUT}/lists_seed{SEED}.json"))["items"]}


def rank_at(h_row, first_name):
    q1 = h_row.unsqueeze(0) - is_mu
    q2 = coderes(q1, mu, V)
    sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
    order = torch.argsort(sims, descending=True)
    order_labels = [words[j] for j in order]
    return rank_of(words, order_labels, first_name)


ranks = defaultdict(list)
with torch.no_grad():
    for i, (g, c) in enumerate(pairs):
        probe = f"{c}'s godparent is"
        ids = tok(probe, add_special_tokens=False)["input_ids"]
        n_gc = len(tok(c, add_special_tokens=False)["input_ids"])
        hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
        h = F.normalize(hs[11][0].float(), dim=1)
        fn = g.split()[0]
        r_final = rank_at(h[-1], fn)
        assert r_final == built[i]["name_rank"], (i, r_final, built[i]["name_rank"])
        sub = "stored-trained" if i in rev_idx else "forward-only"
        ranks[("final-is", sub)].append(r_final)
        ranks[("godchild", sub)].append(rank_at(h[n_gc - 1], fn))
        ranks[("possessive", sub)].append(rank_at(h[n_gc], fn))
print("sanity: final-position ranks reproduce the built lists exactly (60/60)")
for pos in ("final-is", "godchild", "possessive"):
    for sub in ("stored-trained", "forward-only"):
        rs = sorted(r for r in ranks[(pos, sub)] if r is not None)
        med = rs[len(rs) // 2] if rs else None
        le20 = sum(1 for r in ranks[(pos, sub)] if r is not None and r <= 20)
        le100 = sum(1 for r in ranks[(pos, sub)] if r is not None and r <= 100)
        print(f"{pos:11s} {sub:15s} median {med}  <=20 {le20}/30  <=100 {le100}/30")
