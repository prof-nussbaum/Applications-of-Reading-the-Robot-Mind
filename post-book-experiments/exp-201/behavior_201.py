"""exp-201 behavioral test (published bundle's definition).
Greedy generation from the probe prompt; PASS iff the generation
contains the full expected godchild name. Primary window 8 tokens;
50-token variant recorded too. One arm per invocation.
Usage: behavior_201.py <mix_201a|mix_201b|mix_201c>
"""
import json, sys, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

EXP = "/home/ecpi-student-05/robot-mind/exp-201"
arm = sys.argv[1]
assert arm in ("mix_201a", "mix_201b", "mix_201c"), arm
dev = "cuda" if torch.cuda.is_available() else "cpu"

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(f"{EXP}/models/{arm}").to(dev)
model.eval()

items = json.load(open(f"{EXP}/dataset_201.json"))["probe_items"]
out = []
with torch.no_grad():
    for i, it in enumerate(items):
        ids = tok(it["reverse_prompt"], return_tensors="pt").input_ids.to(dev)
        gen = model.generate(ids, max_new_tokens=50, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        new = gen[0][ids.shape[1]:].tolist()
        gen8 = tok.decode(new[:8])
        gen50 = tok.decode(new)
        out.append({
            "index": i,
            "godparent": it["godparent"],
            "godchild": it["godchild"],
            "reverse_prompt": it["reverse_prompt"],
            "expected_name": it["name"],
            "gen_8tok": gen8,
            "gen_50tok": gen50,
            "pass_8tok": it["name"] in gen8,
            "pass_50tok": it["name"] in gen50,
        })
path = f"{EXP}/behavior_{arm}.json"
json.dump(out, open(path, "w"), indent=1)
n8 = sum(r["pass_8tok"] for r in out)
n50 = sum(r["pass_50tok"] for r in out)
print(f"[{arm}] behavioral 8tok: {n8}/60 pass; 50tok: {n50}/60 pass -> {path}")
