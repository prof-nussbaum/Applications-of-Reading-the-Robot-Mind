"""exp-201-exposure STEP: BLIND LENS VERDICTS + SPECIFICITY.
Must run and be saved BEFORE any behavioural generation (protocol).
Lens identical to exp-201-reversed/verdicts_reversed.py: imports
probe_layers from rtrm_probe.py (ln_f applied before lm_head).
hs[11] primary; hs[12] and final-output rank recorded.
Specificity at hs[11]: correct godparent's first token ranked against
the 60 godparent first tokens (same lens logit vector).
"""
import json, sys, torch
from datetime import datetime, timezone
sys.path.insert(0, "/home/ecpi-student-05/robot-mind/github-published/post-book-experiments/reversal-curse-detector")
from rtrm_probe import probe_layers, verdict as band
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP = "/home/ecpi-student-05/robot-mind/exp-201-exposure"
SRC = "/home/ecpi-student-05/robot-mind/exp-201/dataset_201.json"
d = json.load(open(SRC))
items = d["probe_items"]; pairs = d["pairs"]
tok = AutoTokenizer.from_pretrained("gpt2")

def tids(s):
    return tok(s, add_special_tokens=False)["input_ids"]

G_first = [tids(" " + G)[0] for G, C in pairs]
assert len(set(G_first)) == 60, "godparent first tokens not distinct"

def lens_logits(model, lw, lb, W, prompt):
    ids = torch.tensor([tids(prompt)])
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    out = {}
    for L in (11, 12):
        h = hs[L][0].float()
        mu = h.mean(-1, keepdim=True)
        var = ((h - mu) ** 2).mean(-1, keepdim=True)
        hn = (h - mu) / torch.sqrt(var + 1e-5) * lw + lb
        out[L] = (hn @ W.T)[-1]
    return out

def rank_of(v, t):
    return int((v > v[t]).sum()) + 1

ts_start = datetime.now(timezone.utc).isoformat()
all_out = {"timestamp_utc": ts_start,
           "note": "Blind verdicts + specificity computed and saved BEFORE any behavioural generation.",
           "models": {}}
for arm in ("expA", "expB", "expC", "expD"):
    for seed in (0, 1, 2):
        key = f"{arm}_s{seed}"
        model = AutoModelForCausalLM.from_pretrained(f"{EXP}/models/{key}").eval()
        lw = model.transformer.ln_f.weight.detach().float()
        lb = model.transformer.ln_f.bias.detach().float()
        W = model.lm_head.weight.detach().float()
        recs = {"reversed": [], "forward": []}
        for i, it in enumerate(items):
            # REVERSED probe: "{C}'s godparent is" -> " {G}"
            probe = f"{it['godchild']}'s godparent is"
            answer = f" {it['godparent']}"
            per_layer, piece = probe_layers(model, tok, lw, lb, W, probe, answer, [11, 12])
            v11 = lens_logits(model, lw, lb, W, probe)[11]
            tgt = tids(answer)[0]
            ranks60 = [rank_of(v11, t) for t in G_first]
            rc = ranks60[i]
            others = [r for j, r in enumerate(ranks60) if j != i]
            ids = torch.tensor([tids(probe)])
            with torch.no_grad():
                logits = model(ids).logits[0, -1].float()
            frank = rank_of(logits, tgt)
            recs["reversed"].append({
                "index": i, "godparent": it["godparent"], "godchild": it["godchild"],
                "probe": probe, "expected_name": it["godparent"], "first_piece": piece,
                "lens_rank_hs11": per_layer[11][0], "verdict_hs11": band(per_layer[11][0]),
                "lens_rank_hs12": per_layer[12][0], "verdict_hs12": band(per_layer[12][0]),
                "final_rank": frank, "verdict_final": band(frank),
                "top_hs11": per_layer[11][1], "top_hs12": per_layer[12][1],
                "spec_best_of_60": rc == min(ranks60),
                "spec_frac_others_beaten": sum(1 for r in others if rc < r) / 59.0,
                "spec_correct_rank_among_60": rc,
                "set_median_rank_60": sorted(ranks60)[30],
                "ranks60_hs11": ranks60,
            })
            # FORWARD probe: "{G}'s godchild is" -> " {C}"
            fprobe = it["reverse_prompt"]
            fanswer = f" {it['name']}"
            fpl, fpiece = probe_layers(model, tok, lw, lb, W, fprobe, fanswer, [11, 12])
            recs["forward"].append({
                "index": i, "godparent": it["godparent"], "godchild": it["godchild"],
                "probe": fprobe, "expected_name": it["name"], "first_piece": fpiece,
                "lens_rank_hs11": fpl[11][0], "verdict_hs11": band(fpl[11][0]),
                "lens_rank_hs12": fpl[12][0], "verdict_hs12": band(fpl[12][0]),
            })
        all_out["models"][key] = recs
        n_stored = sum(r["verdict_hs11"] == "STORED" for r in recs["reversed"])
        n_spec = sum(r["spec_best_of_60"] for r in recs["reversed"])
        print(f"[{key}] reversed hs11 STORED={n_stored}/60 specificity_best={n_spec}/60", flush=True)
        del model
all_out["timestamp_saved_utc"] = datetime.now(timezone.utc).isoformat()
json.dump(all_out, open(f"{EXP}/verdicts_exposure.json", "w"), indent=1)
print("VERDICTS_SAVED_UTC", all_out["timestamp_saved_utc"], flush=True)
