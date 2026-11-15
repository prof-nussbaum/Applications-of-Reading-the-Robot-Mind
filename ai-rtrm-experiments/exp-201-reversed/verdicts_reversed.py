"""exp-201-reversed STEP 2: BLIND VERDICTS (before any behavioural scoring).
Mirrors the published probe exactly (imports probe_layers from rtrm_probe.py).
Primary: hs[11] lens rank. Secondary: hs[12] lens rank. Baseline: final-output rank.
"""
import json, sys, torch
from datetime import datetime, timezone
sys.path.insert(0, "/home/ecpi-student-05/robot-mind/github-published/post-book-experiments/reversal-curse-detector")
from rtrm_probe import probe_layers, verdict as band
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP = "/home/ecpi-student-05/robot-mind/exp-201"
OUT = "/home/ecpi-student-05/robot-mind/exp-201-reversed"
d = json.load(open(f"{EXP}/dataset_201.json"))
items = d["probe_items"]
tok = AutoTokenizer.from_pretrained("gpt2")

ts_start = datetime.now(timezone.utc).isoformat()
all_out = {"timestamp_utc": ts_start, "note": "Blind verdicts computed and saved BEFORE any behavioural generation was run.",
           "arms": {}}
for arm in ("mix_201a", "mix_201b", "mix_201c"):
    model = AutoModelForCausalLM.from_pretrained(f"{EXP}/models/{arm}").eval()
    lw = model.transformer.ln_f.weight.detach().float()
    lb = model.transformer.ln_f.bias.detach().float()
    W = model.lm_head.weight.detach().float()
    recs = []
    for i, it in enumerate(items):
        probe = f"{it['godchild']}'s godparent is"
        answer = f" {it['godparent']}"
        per_layer, piece = probe_layers(model, tok, lw, lb, W, probe, answer, [11, 12])
        r11, r12 = per_layer[11][0], per_layer[12][0]
        ids = torch.tensor([tok(probe, add_special_tokens=False)["input_ids"]])
        tgt = tok(answer, add_special_tokens=False)["input_ids"][0]
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        frank = int((logits > logits[tgt]).sum()) + 1
        recs.append({
            "index": i, "godparent": it["godparent"], "godchild": it["godchild"],
            "probe": probe, "expected_name": it["godparent"], "first_piece": piece,
            "lens_rank_hs11": r11, "verdict_hs11": band(r11),
            "lens_rank_hs12": r12, "verdict_hs12": band(r12),
            "final_rank": frank, "verdict_final": band(frank),
            "top_hs11": per_layer[11][1], "top_hs12": per_layer[12][1],
        })
        print(f"[{arm}] {i:2d} hs11 {band(r11):7s} {r11:>6} | hs12 {band(r12):7s} {r12:>6} | final {band(frank):7s} {frank:>6} :: {probe!r}")
    all_out["arms"][arm] = recs
    del model
all_out["timestamp_saved_utc"] = datetime.now(timezone.utc).isoformat()
json.dump(all_out, open(f"{OUT}/verdicts_reversed.json", "w"), indent=1)
print("VERDICTS_SAVED_UTC", all_out["timestamp_saved_utc"])
