"""Render SUBTRACTED-list prompt pages for the 6 layperson-restatement prompts,
from the FROZEN exp-054e (seeded) results -- no new findings.

C2 recipe reproduced EXACTLY from pronoun-battery-v2/exp-054e/run_054e.py:
  pool side : global-center, PCA-64 (torch.pca_lowrank, torch.manual_seed(0)),
              residual; true cosine on both sides.
  query side: q = ((h - mu_pron) - mu) - V V^T ((h - mu_pron) - mu);
              cosine against normalized pool residuals.
One page per prompt at the sentence pronoun token: word cloud of the top-40
subtracted similarities + ranked table (top-15 + referent + distractor rows),
full prompt verbatim. Panel layout mirrors exp-052/render_052.py prompt pages.

Hard gate: every rendered rank must reproduce the recorded exp-054e C2
numbers from results_054e.json. Any mismatch -> exit(1), no pages.
"""
import os, json, random
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import GPT2TokenizerFast

D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/exp-054e"
E = "/home/ecpi-student-05/robot-mind/experiments"
B = "/home/ecpi-student-05/robot-mind/pronoun-battery"
BV = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2"
K = 64
os.makedirs(D, exist_ok=True)

# (pdf basename, pool id, pronoun, referent, distractor, expected verbatim prompt)
ITEMS = [
    ("king_subtracted", "ceil2", "he", "king", "farmer",
     "The king ruled for years because he was wise. What does the word he refer to? Answer with one word:"),
    ("dog_subtracted", "ceil1", "it", "dog", "cat",
     "The dog barked because it was excited. What does the word it refer to? Answer with one word:"),
    ("car_subtracted", "ceil3", "it", "car", "truck",
     "The car broke down because it needed repairs. What does the word it refer to? Answer with one word:"),
    ("baby_subtracted", "ceil4", "it", "baby", "child",
     "The baby cried because it was hungry. What does the word it refer to? Answer with one word:"),
    ("pair3a_car_tree_subtracted", "pair3a", "it", "car", "tree",
     "The car hit the tree because it was speeding. What does the word it refer to? Answer with one word:"),
    ("pair3b_car_tree_subtracted", "pair3b", "it", "tree", "car",
     "The car hit the tree because it had deep roots. What does the word it refer to? Answer with one word:"),
]

# expected ranks: from the frozen exp-054e results (the numbers we may cite)
rec054e = json.load(open(D + "/results_054e.json"))
by_row = {r["id"]: r for r in rec054e["rows"]}
assert rec054e["seed"] == 0 and rec054e["K"] == 64

tok = GPT2TokenizerFast.from_pretrained("gpt2")


def word_of(i):
    return tok.decode([i]).strip().lower()


# ---- pool (identical to run_054e.py) ----
spool = (torch.load(E + "/exp-032d-all-pools/pool_all.pt", map_location="cpu", weights_only=False)
         + torch.load(E + "/exp-037-pronoun-binding/pool_037.pt", map_location="cpu", weights_only=False)
         + torch.load(B + "/exp-047/pool_047.pt", map_location="cpu", weights_only=False))
vecs, words = [], []
for r in spool:
    for p, v in enumerate(r["hs"][2]):
        vecs.append(v.float())
        words.append(word_of(r["ids"][p]))
M = torch.stack(vecs)
print("pool", M.shape[0], flush=True)
assert M.shape[0] == 30099, "pool size changed!"

mu = M.mean(dim=0)
Mc = M - mu
torch.manual_seed(0)
U, S, V = torch.pca_lowrank(Mc, q=K)


def coderes(H):
    Hc = H - mu
    return Hc - (Hc @ V) @ V.T


R = coderes(M)
Rn = F.normalize(R, dim=1)
print("pool residual energy fraction: %.4f" % float((R ** 2).sum() / (Mc ** 2).sum()), flush=True)

uniq = sorted(set(words))
w2i = {w: i for i, w in enumerate(uniq)}
wi = torch.tensor([w2i[w] for w in words])
counts = torch.bincount(wi)
mu_w = torch.zeros(len(uniq), M.shape[1]).index_add_(0, wi, M) / counts.unsqueeze(1)

tpool = torch.load(BV + "/exp-052/pool_052.pt", map_location="cpu", weights_only=False)
by_id = {r["id"]: r for r in tpool}


def rank_of(sims, target):
    order = torch.argsort(sims, descending=True)
    return next(int(i) + 1 for i in range(len(order)) if words[order[i]] == target)


# ---- compute + VERIFY (hard gate against results_054e.json) ----
verified = []
for base, iid, pron, ref, dist, exp_prompt in ITEMS:
    row = by_row[iid]
    expected = {ref: row["C2_rank"], dist: row["C2_wrong"]}
    rec = by_id[iid]
    ids = rec["ids"]
    prompt = tok.decode(ids)
    assert prompt == exp_prompt, "prompt text mismatch for %s:\n%r\n%r" % (iid, prompt, exp_prompt)
    pos = ids.index(tok(" " + pron, add_special_tokens=False)["input_ids"][0])
    h = rec["hs"][2][pos].float().unsqueeze(0)
    mu_q = mu_w[w2i[pron]].unsqueeze(0)
    q1 = h - mu_q
    q1c = q1 - mu
    q2 = q1c - (q1c @ V) @ V.T
    s2 = (F.normalize(q2, dim=1) @ Rn.T)[0]
    order = torch.argsort(s2, descending=True)
    ranked = [(words[i], float(s2[i])) for i in order]
    got = {w: rank_of(s2, w) for w in expected}
    ok = all(got[w] == e for w, e in expected.items())
    print("%s prompt-ok pos=%d expected=%s got=%s -> %s"
          % (iid, pos, expected, got, "OK" if ok else "MISMATCH"), flush=True)
    print("   top20:", [(w, round(s, 3)) for w, s in ranked[:20]], flush=True)
    if not ok:
        raise SystemExit("RANK MISMATCH on %s: expected %s, got %s -- refusing to render" % (iid, expected, got))
    verified.append((base, iid, pron, ref, dist, pos, len(ids), prompt, ranked))

print("ALL %d RANKS VERIFIED against results_054e.json -- rendering" % len(verified), flush=True)


# ---- render (panel layout mirrors exp-052/render_052.py prompt pages) ----
def draw_cloud(ax, fig, words_sims, title, tfs=19):
    W, H = 1600, 1000
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_title(title, fontsize=tfs, pad=12)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    sims = [s for _, s in words_sims]
    smin, smax = min(sims), max(sims)
    placed = []
    for w, s in words_sims:
        fs = 22 + 78 * (s - smin) / (smax - smin + 1e-9)
        for _ in range(400):
            x = random.uniform(80, W - 80)
            y = random.uniform(80, H - 80)
            t = ax.text(x, y, w, fontsize=fs,
                        ha="center", va="center", color="#1a1a1a",
                        fontweight="bold" if fs > 70 else "normal")
            bb = t.get_window_extent(
                renderer=renderer).transformed(ax.transData.inverted())
            if all(bb.x1 < px0 or px1 < bb.x0 or
                   bb.y1 < py0 or py1 < bb.y0
                   for px0, py0, px1, py1 in placed):
                placed.append((bb.x0, bb.y0, bb.x1, bb.y1))
                break
            t.remove()


def draw_table(ax, rows, subtitle, hi_words=()):
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, len(rows) + 1.5)
    ax.text(6, len(rows) + 1.0, subtitle, fontsize=15, ha="center")
    for j, (rk, w, s) in enumerate(rows):
        y = len(rows) - j
        hi = w in hi_words
        ax.text(1.2, y, str(rk), fontsize=13, ha="right", va="center",
                fontweight="bold" if hi else "normal",
                color="#b00000" if hi else "#1a1a1a")
        ax.text(2.0, y, w, fontsize=19, ha="left", va="center",
                fontweight="bold", color="#b00000" if hi else "#1a1a1a")
        ax.text(10.8, y, "%.3f" % s, fontsize=13, ha="right",
                va="center", color="#b00000" if hi else "#555555")
        if hi_words and j == 14:
            ax.plot([0.5, 11.5], [y - 0.55, y - 0.55],
                    color="#999999", lw=1.5)


random.seed(7)
for base, iid, pron, ref, dist, pos, ntok, prompt, ranked in verified:
    top40 = ranked[:40]
    rows = [(k + 1, w, s) for k, (w, s) in enumerate(ranked[:15])]
    seen = {w for _, w, _ in rows}
    for cand in (ref, dist):
        if cand in seen:
            continue
        rk = next(k + 1 for k, (w, _) in enumerate(ranked) if w == cand)
        sc = next(s for w, s in ranked if w == cand)
        rows.append((rk, cand, sc))
        seen.add(cand)
    hi = (ref, dist)
    rec = by_id[iid]
    frag = tok.decode(rec["ids"][:pos + 1])
    title = ("Analyzed fragment: \"%s\"\n"
             "Pronoun token \u2018%s\u2019 (token %d of %d) \u2014 40 closest tokens, SUBTRACTED list "
             "(PCA-64 residual, pronoun-mean removed). Bigger = closer.\n"
             "Full prompt: \"%s\"" % (frag, pron, pos + 1, ntok, prompt))
    fig = plt.figure(figsize=(13, 16), dpi=100)
    gs = gridspec.GridSpec(2, 1, height_ratios=[11, 9], hspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    draw_cloud(ax1, fig, top40, title)
    ax2 = fig.add_subplot(gs[1])
    draw_table(ax2, rows, "Closest tokens to \u2018%s\u2019 after subtraction" % pron,
               hi_words=hi)
    rrk = next(k + 1 for k, (w, _) in enumerate(ranked) if w == ref)
    drk = next(k + 1 for k, (w, _) in enumerate(ranked) if w == dist)
    fig.text(0.5, 0.030,
             "expected referent \u2018%s\u2019 #%d   |   distractor \u2018%s\u2019 #%d"
             % (ref, rrk, dist, drk),
             ha="center", fontsize=11, color="#333333")
    fig.text(0.5, 0.012, "exp-054e C2 recipe, torch.manual_seed(0) (frozen results)",
             ha="center", fontsize=10, color="#888888")
    pdf = D + "/%s.pdf" % base
    png = D + "/%s.png" % base
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf, png, flush=True)

print("DONE", flush=True)
