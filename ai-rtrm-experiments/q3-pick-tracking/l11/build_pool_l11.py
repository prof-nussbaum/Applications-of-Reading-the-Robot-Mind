"""Q3 Amendment 1, stage 1: build the LAYER-11 pool mirroring the banked
pool_path2.pt construction EXACTLY, plus an L12 rebuild for validation.

Adapted copy of ~/robot-mind/pronoun-battery-v2/path2-early-warning/
build_pool_path2.py (the original is untouched). Same input
(pool_build_selection.jsonl, same order), same tokenisation, same
per-token unit-normalisation, same fp32 storage, same GPT2LMHeadModel
on CPU. The ONLY change: the forward pass records hidden_states[11]
(the new pool) and hidden_states[12] (a rebuild slice used solely to
validate this machinery against the banked pool before any L11 readout
is computed). Two output files, each in the banked record format
{"id", "ids", "hs": [<one layer>]}:
  pool_L11.pt         hs[0] = hidden_states[11]  (the stage-1 pool)
  pool_L12_rebuild.pt hs[0] = hidden_states[12]  (validation only)
No test items are touched: input is pool_build_selection.jsonl only.
"""
import json, time, torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

EW = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
OUT = "/home/ecpi-student-05/robot-mind/q3-pick-tracking/l11"
rows = [json.loads(l) for l in open(EW + "/pool_build_selection.jsonl")]
print("sentences to build: %d" % len(rows), flush=True)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to("cpu")
recs11, recs12 = [], []
raw_n = 0; raw_sum = 0.0; raw_min = 1e9; raw_max = 0.0
t0 = time.time()
with torch.no_grad():
    for n, r in enumerate(rows):
        ids = tok(r["sentence"])["input_ids"]
        assert len(ids) == r["gpt2_tokens"], ("token count mismatch", r["corpus_index"], len(ids), r["gpt2_tokens"])
        hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
        raw = hs[12][0].float().norm(dim=1)
        raw_n += len(ids); raw_sum += float(raw.sum())
        raw_min = min(raw_min, float(raw.min())); raw_max = max(raw_max, float(raw.max()))
        recs11.append({"id": "wt103-%d" % r["corpus_index"], "ids": ids,
                       "hs": [F.normalize(hs[11][0].float(), dim=1)]})
        recs12.append({"id": "wt103-%d" % r["corpus_index"], "ids": ids,
                       "hs": [F.normalize(hs[12][0].float(), dim=1)]})
        if (n + 1) % 100 == 0:
            print("done %d/%d (%.0fs)" % (n + 1, len(rows), time.time() - t0), flush=True)
torch.save(recs11, OUT + "/pool_L11.pt")
torch.save(recs12, OUT + "/pool_L12_rebuild.pt")
stats = {"records": len(recs11), "total_states": sum(len(r["ids"]) for r in recs11),
         "raw_L12_norm_mean": raw_sum / raw_n, "raw_L12_norm_min": raw_min,
         "raw_L12_norm_max": raw_max, "build_seconds": round(time.time() - t0, 1)}
json.dump(stats, open(OUT + "/build_stats_l11.json", "w"), indent=1)
print(json.dumps(stats), flush=True)
print("wrote pool_L11.pt and pool_L12_rebuild.pt", flush=True)
