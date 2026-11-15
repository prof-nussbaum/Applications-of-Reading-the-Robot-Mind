"""EXPLORATORY, POST-HOC mechanism checks on the COMPLETED Path 2 confirmatory
data (2026-11-15). NOT a new confirmatory test; nothing here re-opens the
locked verdict (NOT WIN). Written for the paper's exploratory-analysis
section, at the author's request, after the confirmatory result was known.

CHECK 1 - flag-leg decomposition from results_confirmatory.json only.
  exp-056 flag legs, exactly as coded in run_confirmatory.py:
    flag = (rL is None) or (rL > 500) or (rC is not None and rC < rL)
    leg a: rC is not None and rL is not None and rC < rL
    leg b: rL is not None and rL > 500
    leg c: rL is None   (absence is encoded as JSON null in the records)
CHECK 2 - pool-size sweep. Query states recomputed with the confirmatory
  recipe copied verbatim from run_confirmatory.py (pool_path2.pt hs[2],
  mu = pool mean, torch.manual_seed(0) immediately before pca_lowrank q=64,
  pronoun-mean subtraction on the raw unit-normalised query FIRST, then
  coderes; LM forward on cuda as in the original run). NOTHING is refit:
  PCA and pronoun means always come from the FULL pool. Only the candidate
  set (rows of the full pool's normalised residual matrix Rn, with labels)
  is subsampled, uniformly at random without replacement via
  torch.randperm(47499, generator seeded 0/1/2) truncated to the size.
  Sizes: 5000 / 15000 / 30000 / 47499 (full, no subsample).
  SANITY ANCHOR: recomputed full-pool r_label/r_comp must equal
  results_confirmatory.json for ALL 617 items, and the full-pool metrics
  must reproduce catch 204/268, FAR 233/320, Fisher p = 0.296. Any mismatch
  aborts the sweep (SystemExit) rather than reporting numbers.
Outputs (new files only): poolsweep_2026-11-15.json, POOLSWEEP_2026-11-15.md
"""
import json, statistics, torch
import torch.nn.functional as F
from scipy.stats import fisher_exact
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
K = 64
PRONOUNS = ["he", "she", "it", "they"]
SIZES = [5000, 15000, 30000]
SEEDS = [0, 1, 2]

recs = json.load(open(D + "/results_confirmatory.json"))
items = [json.loads(l) for l in open(D + "/confirmatory_items.jsonl")]
assert len(recs) == len(items) == 617
# records and items are in the same order (run_confirmatory preserved order);
# verify by sentence+heads key
for r, it in zip(recs, items):
    assert r["sentence"] == it["sentence"] and r["head1"] == it["head1"] and r["head2"] == it["head2"]

# ---------------- CHECK 1 ----------------
def legs(r):
    a = r["r_comp"] is not None and r["r_label"] is not None and r["r_comp"] < r["r_label"]
    b = r["r_label"] is not None and r["r_label"] > 500
    c = r["r_label"] is None
    return a, b, c

for r in recs:
    a, b, c = legs(r)
    assert (a or b or c) == bool(r["rtrm_flag"]), "flag reconstruction mismatch"

def leg_table(rs):
    A = [legs(r) for r in rs]
    return {
        "n": len(rs),
        "flagged": sum(1 for r in rs if r["rtrm_flag"]),
        "leg_a_marginal": sum(1 for L in A if L[0]),
        "leg_b_marginal": sum(1 for L in A if L[1]),
        "leg_c_marginal": sum(1 for L in A if L[2]),
        "leg_a_only": sum(1 for L in A if L[0] and sum(L) == 1),
        "leg_b_only": sum(1 for L in A if L[1] and sum(L) == 1),
        "leg_c_only": sum(1 for L in A if L[2] and sum(L) == 1),
        "legs_ab": sum(1 for L in A if L[0] and L[1] and not L[2]),
        "legs_ac": sum(1 for L in A if L[0] and L[2] and not L[1]),
        "legs_bc": sum(1 for L in A if L[1] and L[2] and not L[0]),
        "legs_abc": sum(1 for L in A if all(L)),
    }

errs = [r for r in recs if not r["model_right"]]
cors = [r for r in recs if r["model_right"]]
check1 = {
    "all": leg_table(recs), "errors": leg_table(errs), "correct": leg_table(cors),
    "median_r_label_errors": statistics.median([r["r_label"] for r in errs if r["r_label"] is not None]),
    "median_r_label_correct": statistics.median([r["r_label"] for r in cors if r["r_label"] is not None]),
    "n_absent_errors": sum(1 for r in errs if r["r_label"] is None),
    "n_absent_correct": sum(1 for r in cors if r["r_label"] is None),
}
print("CHECK1 done:", json.dumps(check1), flush=True)

# ---------------- pool + query construction (verbatim recipe) ----------------
tok = GPT2TokenizerFast.from_pretrained("gpt2")
def word_of(i): return tok.decode([i]).strip().lower()

spool = torch.load(D + "/pool_path2.pt", map_location="cpu", weights_only=False)
vecs, words = [], []
for r in spool:
    for p, v in enumerate(r["hs"][2]):
        vecs.append(v.float()); words.append(word_of(r["ids"][p]))
del spool
M = torch.stack(vecs)
NPOOL = len(M)
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
print("pool built: %d states" % NPOOL, flush=True)

# word -> id over pool labels, for exact rank_of-equivalent lookups
vocab = {}
word_ids = torch.empty(NPOOL, dtype=torch.long)
for i, w in enumerate(words):
    if w not in vocab: vocab[w] = len(vocab)
    word_ids[i] = vocab[w]

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, flush=True)
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(dev)

def rank_in(sims_row, idx, target_word):
    """Best-occurrence 1-based rank of target_word among pool rows idx,
    by descending cosine; None if absent. Identical semantics to
    run_confirmatory.rank_of applied to the restricted candidate set."""
    tid = vocab.get(target_word.lower())
    if tid is None: return None
    sub_sims = sims_row[idx]
    order = torch.argsort(sub_sims, descending=True)
    hits = (word_ids[idx][order] == tid).nonzero()
    if len(hits) == 0: return None
    return int(hits[0].item()) + 1

full_idx = torch.arange(NPOOL)
configs = [("full", None, full_idx)]
for size in SIZES:
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(NPOOL, generator=g)[:size]
        configs.append((size, seed, idx))

# per-config storage: ranks per item
ranks = {name: [] for name, _, _ in [(c[0] if c[1] is None else "%d_s%d" % (c[0], c[1]), None, None) for c in configs]}
def cname(size, seed): return "full" if seed is None else "%d_s%d" % (size, seed)
ranks = {cname(s, sd): {"r_label": [], "r_comp": []} for s, sd, _ in configs}

with torch.no_grad():
    for n_it, it in enumerate(items):
        pron = it["pronoun"]
        prompt = "%s What does the word %s refer to? Answer with one word:" % (it["sentence"], pron)
        ids = tok(prompt)["input_ids"]
        sent_len = len(tok(it["sentence"])["input_ids"])
        o = lm(torch.tensor([ids]).to(dev), output_hidden_states=True)
        h12 = o.hidden_states[12][0].float().cpu()
        h = F.normalize(h12, dim=1)
        pid = tok(" " + pron, add_special_tokens=False)["input_ids"][0]
        pos = [i for i, x in enumerate(ids) if x == pid][0]
        assert pos < sent_len
        hq = h[pos].unsqueeze(0)
        q1 = hq - pron_mu[pron]              # pronoun mean first (run_055 exact)
        q2 = coderes(q1)                     # then pool-mean subtraction + PCA residual
        sims = (F.normalize(q2, dim=1) @ Rn.T)[0]
        rec = recs[n_it]
        for size, seed, idx in configs:
            ranks[cname(size, seed)]["r_label"].append(rank_in(sims, idx, rec["label"]))
            ranks[cname(size, seed)]["r_comp"].append(rank_in(sims, idx, rec["competitor"]))
        if (n_it + 1) % 100 == 0: print("queries done:", n_it + 1, flush=True)

# ---------------- SANITY ANCHOR ----------------
mism = sum(1 for i in range(617)
           if ranks["full"]["r_label"][i] != recs[i]["r_label"] or ranks["full"]["r_comp"][i] != recs[i]["r_comp"])
print("full-pool rank mismatches vs results_confirmatory.json:", mism, "/ 617 items x2 ranks", flush=True)
if mism:
    raise SystemExit("SANITY ANCHOR FAILED: recomputed full-pool ranks do not match the confirmatory records; sweep NOT run to completion, no results written.")

def metrics(rL_list, rC_list):
    flags = []
    for rL, rC in zip(rL_list, rC_list):
        flags.append(bool((rL is None) or (rL > 500) or (rC is not None and rL is not None and rC < rL)))
    miss_i = [i for i, r in enumerate(recs) if not r["model_right"] and not r["margin_flag_B1"]]
    cor_i = [i for i, r in enumerate(recs) if r["model_right"]]
    unfl_i = [i for i, r in enumerate(recs) if not r["margin_flag_B1"]]
    catch = sum(1 for i in miss_i if flags[i]) / len(miss_i)
    far = sum(1 for i in cor_i if flags[i]) / len(cor_i)
    a = sum(1 for i in unfl_i if flags[i] and not recs[i]["model_right"])
    b = sum(1 for i in unfl_i if flags[i] and recs[i]["model_right"])
    c = sum(1 for i in unfl_i if not flags[i] and not recs[i]["model_right"])
    d = sum(1 for i in unfl_i if not flags[i] and recs[i]["model_right"])
    p = float(fisher_exact([[a, b], [c, d]], alternative="greater")[1])
    legb = sum(1 for rL in rL_list if rL is not None and rL > 500) / len(rL_list)
    return {"catch_of_miss": catch, "far_correct": far, "fisher_p": p,
            "fisher_table": [[a, b], [c, d]], "leg_b_rate_all_items": legb,
            "n_flagged": sum(flags)}

sweep = {}
for size, seed, idx in configs:
    nm = cname(size, seed)
    sweep[nm] = metrics(ranks[nm]["r_label"], ranks[nm]["r_comp"])
    print(nm, json.dumps(sweep[nm]), flush=True)

fm = sweep["full"]
anchor_ok = (abs(fm["catch_of_miss"] - 204/268) < 1e-9 and abs(fm["far_correct"] - 233/320) < 1e-9
             and round(fm["fisher_p"], 3) == 0.296)
print("SANITY ANCHOR full-pool metrics reproduce confirmatory (catch .761 / FAR .728 / p .296):", anchor_ok, flush=True)
if not anchor_ok:
    raise SystemExit("SANITY ANCHOR FAILED at metric level; no results written.")

def agg(size):
    ms = [sweep["%d_s%d" % (size, s)] for s in SEEDS]
    keys = ["catch_of_miss", "far_correct", "fisher_p", "leg_b_rate_all_items"]
    return {"per_seed": ms, "mean": {k: sum(m[k] for m in ms) / len(ms) for k in keys}}

out = {
    "label": "EXPLORATORY post-hoc mechanism checks (2026-11-15) on the completed Path 2 confirmatory data. NOT a confirmatory test; the locked verdict (NOT WIN) is unchanged.",
    "check1_flag_leg_decomposition": check1,
    "check1_notes": "Legs as coded in run_confirmatory.py: a = r_comp < r_label (both present); b = r_label > 500; c = r_label absent (JSON null). No confirmatory item has r_label absent (eligibility required >=10 pool occurrences per candidate word), so leg c never fires. Flag reconstruction (a or b or c) matches rtrm_flag on all 617 items.",
    "check2_pool_size_sweep": {
        "method": "Query states recomputed with the confirmatory recipe (PCA-64 seed 0 fit on the FULL pool; pronoun-mean from the FULL pool; nothing refit). Candidate set = uniform random subsample of the 47,499 pool rows (torch.randperm, seeds 0/1/2, without replacement). MISS fixed from confirmatory records (errors with margin_nats >= 0.35; n=268). Fisher: one-sided greater, flag x error among margin-unflagged items, as in analyze_confirmatory.py.",
        "sanity_anchor": "Recomputed full-pool r_label/r_comp match results_confirmatory.json exactly on all 617 items; full-pool metrics reproduce catch 204/268 = 0.761, FAR 233/320 = 0.728, Fisher p = 0.296.",
        "full_47499": sweep["full"],
        "size_5000": agg(5000), "size_15000": agg(15000), "size_30000": agg(30000),
    },
}
json.dump(out, open(D + "/poolsweep_2026-11-15.json", "w"), indent=1)

def pct(x): return "%.1f%%" % (100 * x)
def prow(name, t):
    return "| %s | %d | %d | %d | %d | %d | %d | %d |" % (name, t["n"], t["flagged"],
        t["leg_a_marginal"], t["leg_a_only"], t["leg_b_marginal"], t["leg_b_only"], t["leg_c_marginal"])
L = []
L.append("# Path 2 — EXPLORATORY post-hoc mechanism checks (2026-11-15)")
L.append("")
L.append("**EXPLORATORY — not a confirmatory test.** These two checks were run after the")
L.append("confirmatory result (NOT WIN) was known, to test the mechanism offered for it in")
L.append("the paper: that in the 47,499-state pool the label ranks run into the thousands,")
L.append("so the flag's rank>500 leg fires constantly. They do not re-open, soften, or")
L.append("re-score the locked verdict, and no threshold is recalibrated anywhere in them.")
L.append("")
L.append("## Check 1 — which legs of the exp-056 flag actually fire?")
L.append("")
L.append("From `results_confirmatory.json` only (no new compute). Legs exactly as coded in")
L.append("`run_confirmatory.py`: (a) competitor rank < label rank; (b) label rank > 500;")
L.append("(c) label absent (encoded as null in the records). Reconstructing the flag as")
L.append("a-or-b-or-c reproduces `rtrm_flag` on all 617 items.")
L.append("")
L.append("| items | n | flagged | leg a marginal | leg a only | leg b marginal | leg b only | leg c marginal |")
L.append("|---|---|---|---|---|---|---|---|")
L.append(prow("all", check1["all"]))
L.append(prow("model errors", check1["errors"]))
L.append(prow("model correct", check1["correct"]))
L.append("")
L.append("Legs (a) and (b) co-fire on %d items (errors %d, correct %d). Leg (c) never fires:" % (
    check1["all"]["legs_ab"], check1["errors"]["legs_ab"], check1["correct"]["legs_ab"]))
L.append("no confirmatory item has an absent label — eligibility required >=10 pool")
L.append("occurrences of each candidate word, so absence was designed out of this test.")
L.append("")
L.append("Median label rank: **errors %s**, **correct %s**. Both sit far above the flag's" % (
    check1["median_r_label_errors"], check1["median_r_label_correct"]))
L.append("500 cut, which is why leg (b) fires on roughly two items in three in both groups")
L.append("(errors %d/297, correct %d/320) — it carries almost no information about whether" % (
    check1["errors"]["leg_b_marginal"], check1["correct"]["leg_b_marginal"]))
L.append("the model was right.")
L.append("")
L.append("## Check 2 — pool-size sweep (the mechanism test)")
L.append("")
L.append("Each item's subtracted query state was recomputed with the confirmatory recipe")
L.append("unchanged (PCA-64 fit on the full pool, full-pool pronoun means — nothing refit).")
L.append("Only the candidate set was subsampled: uniform random rows of the full pool,")
L.append("seeds 0/1/2. MISS is fixed from the confirmatory records (268 errors with")
L.append("margin >= 0.35 nats); the margin baseline does not change with pool size.")
L.append("")
L.append("**Sanity anchor: PASS.** Recomputed full-pool label/competitor ranks match")
L.append("`results_confirmatory.json` exactly on all 617 items, and the full-pool row")
L.append("below reproduces the confirmatory numbers (catch 76.1%, FAR 72.8%, p = 0.296).")
L.append("")
L.append("| pool size | catch of MISS | false-alarm rate (correct) | Fisher p (flag x error, margin-unflagged) | leg (b) firing rate |")
L.append("|---|---|---|---|---|")
for size in SIZES:
    m = agg(size)["mean"]
    L.append("| %d (mean of seeds 0-2) | %s | %s | %.3f | %s |" % (size, pct(m["catch_of_miss"]), pct(m["far_correct"]), m["fisher_p"], pct(m["leg_b_rate_all_items"])))
fm = sweep["full"]
L.append("| 47,499 (full pool) | %s | %s | %.3f | %s |" % (pct(fm["catch_of_miss"]), pct(fm["far_correct"]), fm["fisher_p"], pct(fm["leg_b_rate_all_items"])))
L.append("")
L.append("Per-seed values and Fisher tables: `poolsweep_2026-11-15.json`.")
L.append("")
L.append("Reading (exploratory) — CORRECTED after the numbers landed; see POOLSWEEP_2026-11-15.md for the corrected reading. The draft paragraph originally generated here claimed the false-alarm cost falls with pool size; it does not (77.1% at 5,000, the worst row).")
open(D + "/POOLSWEEP_2026-11-15.md", "w").write("\n".join(L) + "\n")
print("WROTE poolsweep_2026-11-15.json and POOLSWEEP_2026-11-15.md", flush=True)
