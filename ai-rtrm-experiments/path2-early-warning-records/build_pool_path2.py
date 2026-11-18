"""Path 2 Amendment 1+2: build GPT-2 states for the subsampled WikiText-103 pool.

Convention, copied from the honest-pool builders of record:
  - exp-032d build_all_pools.py: GPT2LMHeadModel on CPU; per text,
    ids = tok(text)["input_ids"]; hs = model(..., output_hidden_states=True)
    .hidden_states; record {"id", "ids", "hs": [F.normalize(hs[li][0].float(),
    dim=1) for li in [0, 6, 12]]} -- so record hs[2] IS hidden_states[12],
    the battery"s "L12", one unit-normalised state per token position,
    position p labelled by tok.decode([ids[p]]).strip().lower() (run_055.py).
  - exp-037 build_supp_pool.py stores the same normalised states in fp32
    (its header: "match the 032d pool convention: every stored state is
    unit-normalized"); pool_all.pt itself is fp16, pool_037/pool_047 fp32.
    This build stores fp32, matching 037/047 and run_055"s fp32
    pool-identical query path (F.normalize with no half cast).
No test items are touched: input is pool_build_selection.jsonl only.
"""
import json, time, torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
W = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
LAYERS = [0, 6, 12]
rows = [json.loads(l) for l in open(W + "/pool_build_selection.jsonl")]
print("sentences to build: %d" % len(rows), flush=True)
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to("cpu")
recs = []
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
        recs.append({"id": "wt103-%d" % r["corpus_index"], "ids": ids,
                     "hs": [F.normalize(hs[li][0].float(), dim=1) for li in LAYERS]})
        if (n + 1) % 100 == 0:
            print("done %d/%d (%.0fs)" % (n + 1, len(rows), time.time() - t0), flush=True)
torch.save(recs, W + "/pool_path2.pt")
stats = {"records": len(recs), "total_states": sum(len(r["ids"]) for r in recs),
         "raw_L12_norm_mean": raw_sum / raw_n, "raw_L12_norm_min": raw_min,
         "raw_L12_norm_max": raw_max, "build_seconds": round(time.time() - t0, 1)}
json.dump(stats, open(W + "/build_stats.json", "w"), indent=1)
print(json.dumps(stats), flush=True)
print("wrote pool_path2.pt", flush=True)
