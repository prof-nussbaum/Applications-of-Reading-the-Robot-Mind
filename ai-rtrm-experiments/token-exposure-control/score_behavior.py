"""exp-201-exposure STEP: BEHAVIOUR (runs only after verdicts saved).
Mirrors exp-201/behavior_201.py and exp-201-reversed/behavior_reversed.py:
greedy, 50 tokens generated, pass = full expected name in first 8 tokens
(primary) or 50 tokens. Both probe directions, all 12 models.
"""
import json, torch
from datetime import datetime, timezone
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import os
ROBOT_MIND = os.environ.get("ROBOT_MIND", "/home/ecpi-student-05/robot-mind")  # Jetson working tree (data lives outside the repo); override with $ROBOT_MIND

EXP = ROBOT_MIND + "/exp-201-exposure"
SRC = ROBOT_MIND + "/exp-201/dataset_201.json"
items = json.load(open(SRC))["probe_items"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = GPT2Tokenizer.from_pretrained("gpt2")

out = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "models": {}}
for arm in ("expA", "expB", "expC", "expD"):
    for seed in (0, 1, 2):
        key = f"{arm}_s{seed}"
        model = GPT2LMHeadModel.from_pretrained(f"{EXP}/models/{key}").to(dev)
        model.eval()
        recs = {"reversed": [], "forward": []}
        with torch.no_grad():
            for i, it in enumerate(items):
                for direction, probe, expected in (
                    ("reversed", f"{it['godchild']}'s godparent is", it["godparent"]),
                    ("forward", it["reverse_prompt"], it["name"]),
                ):
                    ids = tok(probe, return_tensors="pt").input_ids.to(dev)
                    gen = model.generate(ids, max_new_tokens=50, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                    new = gen[0][ids.shape[1]:].tolist()
                    gen8 = tok.decode(new[:8]); gen50 = tok.decode(new)
                    recs[direction].append({
                        "index": i, "godparent": it["godparent"],
                        "godchild": it["godchild"], "probe": probe,
                        "expected_name": expected, "gen_8tok": gen8,
                        "gen_50tok": gen50,
                        "pass_8tok": expected in gen8,
                        "pass_50tok": expected in gen50,
                    })
        out["models"][key] = recs
        r8 = sum(r["pass_8tok"] for r in recs["reversed"])
        f8 = sum(r["pass_8tok"] for r in recs["forward"])
        print(f"[{key}] reversed 8tok {r8}/60 | forward 8tok {f8}/60", flush=True)
        del model
json.dump(out, open(f"{EXP}/behavior_exposure.json", "w"), indent=1)
print("BEHAVIOR_SAVED", flush=True)
