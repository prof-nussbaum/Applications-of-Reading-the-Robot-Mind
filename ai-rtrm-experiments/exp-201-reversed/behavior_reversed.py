"""exp-201-reversed STEP 3: BEHAVIOUR. Mirrors behavior_201.py exactly,
except the probe is the reversed probe and the expected name is the godparent.
Greedy 50 tokens generated; 8-token window is primary.
"""
import json, sys, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import os
ROBOT_MIND = os.environ.get("ROBOT_MIND", "/home/ecpi-student-05/robot-mind")  # Jetson working tree (data lives outside the repo); override with $ROBOT_MIND

EXP = ROBOT_MIND + "/exp-201"
OUT = ROBOT_MIND + "/exp-201-reversed"
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
        probe = f"{it['godchild']}'s godparent is"
        ids = tok(probe, return_tensors="pt").input_ids.to(dev)
        gen = model.generate(ids, max_new_tokens=50, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        new = gen[0][ids.shape[1]:].tolist()
        gen8 = tok.decode(new[:8])
        gen50 = tok.decode(new)
        out.append({
            "index": i,
            "godparent": it["godparent"],
            "godchild": it["godchild"],
            "probe": probe,
            "expected_name": it["godparent"],
            "gen_8tok": gen8,
            "gen_50tok": gen50,
            "pass_8tok": it["godparent"] in gen8,
            "pass_50tok": it["godparent"] in gen50,
        })
path = f"{OUT}/behavior_{arm}.json"
json.dump(out, open(path, "w"), indent=1)
n8 = sum(r["pass_8tok"] for r in out)
n50 = sum(r["pass_50tok"] for r in out)
print(f"[{arm}] behavioral 8tok: {n8}/60 pass; 50tok: {n50}/60 pass -> {path}")
