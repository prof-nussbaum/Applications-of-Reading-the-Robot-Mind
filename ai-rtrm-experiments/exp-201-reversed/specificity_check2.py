import json, torch, statistics as st
from transformers import AutoModelForCausalLM, AutoTokenizer
base = "/home/ecpi-student-05/robot-mind/"
d = json.load(open(base + "exp-201/dataset_201.json"))
pairs = d["pairs"]; items = d["probe_items"]
verd = json.load(open(base + "exp-201-reversed/verdicts_reversed.json"))["arms"]
tok = AutoTokenizer.from_pretrained("gpt2")
def first_tok(name):
    return tok(" " + name, add_special_tokens=False)["input_ids"][0]
G_first = [first_tok(G) for G, C in pairs]
out = {}
for arm in ["mix_201a", "mix_201b", "mix_201c"]:
    model = AutoModelForCausalLM.from_pretrained(base + "exp-201/models/" + arm).eval()
    lw = model.transformer.ln_f.weight.detach().float()
    lb = model.transformer.ln_f.bias.detach().float()
    W = model.lm_head.weight.detach().float()
    rows = []; mismatches = 0
    for it, (G, C) in zip(items, pairs):
        probe = it["godchild"] + "'s godparent is"
        ids = torch.tensor([tok(probe, add_special_tokens=False)["input_ids"]])
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        h = hs[11][0, -1].float()
        hn = (h - h.mean()) / (h.var(unbiased=False) + 1e-5).sqrt() * lw + lb
        logits = W @ hn
        def rank(t):
            return int((logits > logits[t]).sum()) + 1
        rc = rank(first_tok(G))
        if rc != verd[arm][len(rows)]["lens_rank_hs11"]:
            mismatches += 1
        others = [rank(t) for t in set(G_first) if t != first_tok(G)]
        others_sorted = sorted(others)
        pct = sum(1 for r in others if rc < r) / max(len(others), 1)
        rows.append({"item": it["godchild"], "correct_rank": rc,
                     "median_other": others_sorted[len(others_sorted) // 2],
                     "best_other": others_sorted[0],
                     "frac_others_beaten": round(pct, 3)})
    out[arm] = {"validation_mismatches_vs_verdicts": mismatches,
                "median_correct_rank": st.median(r["correct_rank"] for r in rows),
                "median_of_median_other": st.median(r["median_other"] for r in rows),
                "mean_frac_others_beaten": round(st.mean(r["frac_others_beaten"] for r in rows), 3),
                "n_correct_is_best_of_set": sum(1 for r in rows if r["correct_rank"] <= r["best_other"]),
                "n_correct_in_top100": sum(1 for r in rows if r["correct_rank"] <= 100)}
    print(arm, json.dumps(out[arm]), flush=True)
    json.dump(rows, open(base + "exp-201-reversed/specificity2_" + arm + ".json", "w"), indent=1)
    del model
json.dump(out, open(base + "exp-201-reversed/specificity2_summary.json", "w"), indent=1)
print("DONE")
