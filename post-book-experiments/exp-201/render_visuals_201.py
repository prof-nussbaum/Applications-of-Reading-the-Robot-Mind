"""exp-201 SME visual layer: big-type PNG tables + showcase pages.
Words headline; statistics live in RESULTS_201.txt (appendix role).
Selection rules are fixed in code (see pick_showcase) -- no hand-picking.
L11 top-10 for showcase pages is recomputed with the published probe's
own convention (hs[11], final LayerNorm + LM head) and ASSERTED equal
to the published probe's rank for that item before any page renders.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

EXP = "/home/ecpi-student-05/robot-mind/exp-201"
ARMS = ["mix_201a", "mix_201b", "mix_201c"]
PRIMARY = ["mix_201b", "mix_201c"]
ARM_LABEL = {
    "mix_201a": "ARM 201a -- BRIDGE (forwards x6 + bridges)",
    "mix_201b": "ARM 201b -- NO BRIDGE (forwards x6)",
    "mix_201c": "ARM 201c -- FACTS x1 (each forward once)",
}
VCOLOR = {"STORED": "#cfe8cf", "WEAK": "#f7e6bd", "MISSING": "#f3c6c6"}

tab = json.load(open(f"{EXP}/tabulation_201.json"))
data = json.load(open(f"{EXP}/dataset_201.json"))
items = data["probe_items"]
rendered = []

# ---------------- results tables (one PNG per arm) ----------------
for arm in ARMS:
    rows = tab[arm]
    fig, ax = plt.subplots(figsize=(13, 23))
    ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 23)
    tall = {v: sum(1 for r in rows if r["verdict"] == v)
            for v in ("STORED", "WEAK", "MISSING")}
    npass = sum(r["pass_8tok"] for r in rows)
    ax.text(0.2, 22.55, f"exp-201  {ARM_LABEL[arm]}", fontsize=21, weight="bold")
    ax.text(0.2, 22.05,
            f"Lens verdicts: STORED {tall['STORED']}   WEAK {tall['WEAK']}   "
            f"MISSING {tall['MISSING']}      Behavioral test: {npass}/60 PASS",
            fontsize=15)
    ax.text(0.2, 21.6, "Every item: the probe asks \"<godparent>'s godchild is\"; "
                       "PASS = the model generates the godchild's full name.",
            fontsize=12.5, style="italic")
    y = 21.15
    rh = 0.335
    ax.text(0.2, y, "#", fontsize=14, weight="bold")
    ax.text(0.75, y, "GODPARENT -> GODCHILD", fontsize=14, weight="bold")
    ax.text(7.6, y, "LENS SAYS", fontsize=14, weight="bold")
    ax.text(9.9, y, "L11 RANK", fontsize=14, weight="bold")
    ax.text(11.2, y, "BEHAVIOR", fontsize=14, weight="bold")
    y -= rh
    for r in rows:
        ax.add_patch(Rectangle((7.45, y - 0.045), 2.3, rh,
                               facecolor=VCOLOR[r["verdict"]], edgecolor="none"))
        ax.text(0.2, y, str(r["index"]), fontsize=13.5)
        ax.text(0.75, y, f"{r['godparent']} -> {r['godchild']}", fontsize=13.5)
        ax.text(7.6, y, r["verdict"], fontsize=13.5, weight="bold")
        ax.text(9.9, y, str(r["rank"]), fontsize=13.5)
        ax.text(11.2, y, "PASS" if r["pass_8tok"] else "FAIL",
                fontsize=13.5, weight="bold",
                color="#1a7a1a" if r["pass_8tok"] else "#b00020")
        y -= rh
    path = f"{EXP}/results_table_{arm}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    rendered.append(path)
    print("table:", path)

# ---------------- showcase selection (fixed rules) ----------------
def pick(kind):
    pools = [(arm, tab[arm]) for arm in PRIMARY] + [("mix_201a", tab["mix_201a"])]
    if kind == "stored_pass":
        cand = [(arm, r) for arm, rows in pools for r in rows
                if r["verdict"] == "STORED" and r["pass_8tok"]]
        cand.sort(key=lambda t: t[1]["rank"])
        return cand[0] if cand else None
    if kind == "missing_fail":
        cand = [(arm, r) for arm, rows in pools for r in rows
                if r["verdict"] == "MISSING" and not r["pass_8tok"]]
        cand.sort(key=lambda t: -t[1]["rank"])
        return cand[0] if cand else None
    if kind == "disagreement":
        cand = [(arm, r) for arm, rows in pools for r in rows
                if (r["verdict"] == "STORED" and not r["pass_8tok"]) or
                   (r["verdict"] == "MISSING" and r["pass_8tok"])]
        return cand[0] if cand else None
    if kind == "weak":
        cand = [(arm, r) for arm, rows in pools for r in rows
                if r["verdict"] == "WEAK"]
        cand.sort(key=lambda t: abs(t[1]["rank"] - 50))
        return cand[0] if cand else None
    raise ValueError(kind)

# ---------------- L11 top-10 (published convention, asserted) ----------------
tok = GPT2Tokenizer.from_pretrained("gpt2")
_cache = {}

def lens_top10(arm, row):
    if arm not in _cache:
        m = GPT2LMHeadModel.from_pretrained(f"{EXP}/models/{arm}")
        m.eval()
        _cache[arm] = m
    model = _cache[arm]
    it = items[row["index"]]
    ids = tok(it["reverse_prompt"], return_tensors="pt").input_ids
    with torch.no_grad():
        hs = model.transformer(ids, output_hidden_states=True).hidden_states
        h = model.transformer.ln_f(hs[11][0, -1])
        logits = model.lm_head(h)
    target = tok.encode(" " + it["name"])[0]
    rank = int((logits > logits[target]).sum()) + 1
    assert rank == row["rank"], \
        f"rank mismatch {arm} item {row['index']}: lens {rank} vs probe {row['rank']}"
    top = torch.topk(logits, 10)
    return [(int(i), tok.decode([int(i)])) for i in top.indices], target

# ---------------- showcase pages ----------------
import textwrap

def showcase(kind, title_word):
    got = pick(kind)
    if got is None:
        print(f"showcase {kind}: NO qualifying item -- skipped (declared, not hidden)")
        return
    arm, r = got
    it = items[r["index"]]
    tops, target = lens_top10(arm, r)
    fig, ax = plt.subplots(figsize=(12.5, 15))
    ax.axis("off"); ax.set_xlim(0, 12.5); ax.set_ylim(0, 15)
    ax.add_patch(Rectangle((0, 13.55), 12.5, 1.45, facecolor=VCOLOR[r["verdict"]],
                           edgecolor="none"))
    ax.text(0.3, 13.95, f"VERDICT: {r['verdict']}", fontsize=40, weight="bold")
    ax.text(8.1, 14.12, f"{ARM_LABEL[arm].split(' -- ')[0]}  item {r['index']}",
            fontsize=15, weight="bold")
    ax.text(8.1, 13.72, f"behavior: {'PASS' if r['pass_8tok'] else 'FAIL'}   "
                        f"L11 rank: {r['rank']}", fontsize=15)
    y = 13.15
    def block(label, text, size=16, dy=0.62, mono=False):
        nonlocal y
        ax.text(0.3, y, label, fontsize=13, weight="bold", color="#444444")
        y -= 0.34
        for ln in textwrap.wrap(text, 74) or [""]:
            ax.text(0.3, y, ln, fontsize=size,
                    family="monospace" if mono else "sans-serif")
            y -= dy
        y -= 0.28
    fwd = [x for x in data["forwards"]
           if it["godparent"] in x and it["godchild"] in x][0]
    block("FORWARD TEXT (trained):", f'"{fwd}"')
    if arm == "mix_201a":
        brg = [x for x in data["bridges"]
               if it["godparent"] in x and it["godchild"] in x][0]
        block("BRIDGE TEXT (trained, this arm only):", f'"{brg}"')
    block("PROBE PROMPT (the test input, never trained):",
          f'"{it["reverse_prompt"]}"')
    pieces = [tok.decode([t]) for t in tok.encode(" " + it["name"])]
    ax.text(0.3, y, "EXPECTED ANSWER AND ITS TOKENS:", fontsize=13,
            weight="bold", color="#444444"); y -= 0.42
    ax.text(0.3, y, f"{it['name']}    tokens: " +
            " | ".join(repr(p) for p in pieces), fontsize=16); y -= 0.4
    ax.text(0.3, y, f"the lens scores the FIRST piece: {pieces[0]!r}  "
                    f"(marked with >> below)", fontsize=13.5, style="italic")
    y -= 0.62
    ax.text(0.3, y, "WHAT LAYER 11 EXPECTS NEXT -- TOP 10 TOKEN PIECES:",
            fontsize=13, weight="bold", color="#444444"); y -= 0.42
    for k, (tid, piece) in enumerate(tops, 1):
        mark = "  >> EXPECTED PIECE" if tid == target else ""
        ax.text(0.5, y, f"{k:>2}. {piece!r}{mark}", fontsize=16,
                weight="bold" if tid == target else "normal",
                color="#b00020" if tid == target else "black")
        y -= 0.4
    if r["rank"] > 10:
        ax.text(0.5, y, f"the expected piece {pieces[0]!r} is ranked "
                        f"#{r['rank']} -- outside the top 10",
                fontsize=15, weight="bold", color="#b00020"); y -= 0.55
    y -= 0.15
    block("THE MODEL'S ACTUAL ANSWER (greedy, first 8 tokens):",
          f'"{r["gen_8tok"]}"')
    ax.text(0.3, y, f"contains the full name \"{it['name']}\": "
                    f"{'YES' if r['pass_8tok'] else 'NO'}  ->  behavioral "
                    f"{'PASS' if r['pass_8tok'] else 'FAIL'}",
            fontsize=17, weight="bold"); y -= 0.6
    ax.text(0.3, 0.55, "exp-201 - godparent/godchild domain - published RTRM "
                       "logit-lens probe (layer 11)", fontsize=11.5, color="#555555")
    path = f"{EXP}/showcase_{kind}_{arm}_item{r['index']}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    rendered.append(path)
    print(f"showcase {kind}: {arm} item {r['index']} -> {path}")

showcase("stored_pass", "STORED")
showcase("missing_fail", "MISSING")
showcase("disagreement", "DISAGREEMENT")
showcase("weak", "WEAK")
print(json.dumps({"rendered": rendered}, indent=1))
