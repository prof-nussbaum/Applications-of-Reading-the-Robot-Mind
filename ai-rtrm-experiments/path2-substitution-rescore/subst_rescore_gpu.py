import json, re, torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# GPU variant of subst_rescore.py (Paul's instruction, 2026-11-15):
# identical scoring, model and inputs on cuda. Original CPU script untouched.
EW = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
PC = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-positive-control"
DEV = "cuda"
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEV)

def logprob(s):
    ids = tok(s, return_tensors="pt")["input_ids"].to(DEV)
    with torch.no_grad():
        lp = torch.log_softmax(lm(ids).logits[0, :-1], dim=-1)
    tgt = ids[0, 1:]
    return float(lp[torch.arange(len(tgt), device=DEV), tgt].sum())

def subst(sentence, pronoun, noun):
    return re.sub(r"\b" + re.escape(pronoun) + r"\b", "the " + noun,
                  sentence, count=1, flags=re.IGNORECASE)

# ---- Part 1: substitution rescoring of the 617 DPR items (exploratory) ----
items = json.load(open(EW + "/results_confirmatory.json"))
out = []
for i, it in enumerate(items):
    if i % 100 == 0:
        print(f"part1 {i}/{len(items)}", flush=True)
    s_lab = subst(it["sentence"], it["pronoun"], it["label"])
    s_comp = subst(it["sentence"], it["pronoun"], it["competitor"])
    lp_lab, lp_comp = logprob(s_lab), logprob(s_comp)
    sub_right = lp_lab > lp_comp
    out.append({"item": it["item"], "split": it["split"],
                "question_right": bool(it["model_right"]),
                "sub_right": bool(sub_right),
                "sub_margin": lp_lab - lp_comp})
n = len(out)
q_acc = sum(o["question_right"] for o in out) / n
s_acc = sum(o["sub_right"] for o in out) / n
both = sum(1 for o in out if o["question_right"] and o["sub_right"])
q_only = sum(1 for o in out if o["question_right"] and not o["sub_right"])
s_only = sum(1 for o in out if not o["question_right"] and o["sub_right"])
neither = sum(1 for o in out if not o["question_right"] and not o["sub_right"])
print(f"DPR 617: question-format accuracy {q_acc:.3f}, substitution accuracy {s_acc:.3f}")
print(f"both right {both}, question-only {q_only}, substitution-only {s_only}, neither {neither}")
print(f"of {n - sum(o['question_right'] for o in out)} question-format errors, "
      f"{s_only} are correct by substitution "
      f"({100*s_only/max(1,(n - sum(o['question_right'] for o in out))):.1f}%)")
json.dump(out, open(EW + "/subst_rescore_617.json", "w"), indent=1)

# ---- Part 2: logit lens at the pronoun, control number items (exploratory) ----
ctrl = json.load(open(PC + "/results_control.json"))
num = [c for c in ctrl if c["family"] == "number"]
lw = lm.transformer.ln_f.weight.detach().float()
lb = lm.transformer.ln_f.bias.detach().float()
W = lm.lm_head.weight.detach().float()
def first_tok_id(name):
    return tok(" " + name, add_special_tokens=False)["input_ids"][0]
lens_rows = []
with torch.no_grad():
    for c in num:
        ids = tok(c["prompt"])["input_ids"]
        hs = lm(torch.tensor([ids], device=DEV), output_hidden_states=True).hidden_states
        pid = tok(" " + c["pronoun"], add_special_tokens=False)["input_ids"][0]
        pos = [i for i, x in enumerate(ids) if x == pid][0]
        res = {}
        for layer in (11, 12):
            h = hs[layer][0, pos].float()
            hn = (h - h.mean()) / (h.var(unbiased=False) + 1e-5).sqrt() * lw + lb
            logits = W @ hn
            def rank(t):
                return int((logits > logits[t]).sum()) + 1
            res[layer] = (rank(first_tok_id(c["label"])), rank(first_tok_id(c["competitor"])))
        lens_rows.append({"item": c["item"],
                          "lens11_label": res[11][0], "lens11_comp": res[11][1],
                          "lens12_label": res[12][0], "lens12_comp": res[12][1],
                          "cos_label": c["r_label"], "cos_comp": c["r_comp"]})
for layer, kl, kc in ((11, "lens11_label", "lens11_comp"), (12, "lens12_label", "lens12_comp")):
    lean = sum(1 for r in lens_rows if r[kl] < r[kc]) / len(lens_rows)
    print(f"lens hs[{layer}] on 60 number control items: label outranks competitor in {lean:.1%} "
          f"(cosine list: 65.0%)")
json.dump(lens_rows, open(PC + "/lens_number_items.json", "w"), indent=1)
print("DONE")
