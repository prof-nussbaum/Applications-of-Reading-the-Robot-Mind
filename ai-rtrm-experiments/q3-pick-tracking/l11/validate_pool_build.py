"""Q3 Amendment 1, stage 1, gate A: prove the adapted pool build matches
the banked pool_path2.pt construction. Checks, against the banked file:
  - same record count, same record ids in the same order;
  - same token ids in every record (sentence selection/order/tokenisation);
  - L12 states: max abs difference between banked hs[2] and the rebuild
    hs[0] over ALL records (must be at float-recompute tolerance);
  - raw L12 norm stats vs the banked build_stats.json.
Any failure exits nonzero and stops stage 1. Reads banked files only.
"""
import json, torch

import os
ROBOT_MIND = os.environ.get("ROBOT_MIND", "/home/ecpi-student-05/robot-mind")  # Jetson working tree (data lives outside the repo); override with $ROBOT_MIND

EW = ROBOT_MIND + "/pronoun-battery-v2/path2-early-warning"
L11 = ROBOT_MIND + "/q3-pick-tracking/l11"

banked = torch.load(EW + "/pool_path2.pt", map_location="cpu", weights_only=False)
reb12 = torch.load(L11 + "/pool_L12_rebuild.pt", map_location="cpu", weights_only=False)
reb11 = torch.load(L11 + "/pool_L11.pt", map_location="cpu", weights_only=False)
bstats = json.load(open(EW + "/build_stats.json"))
mstats = json.load(open(L11 + "/build_stats_l11.json"))

fails = []
def check(name, ok, detail=""):
    print("%-52s %s %s" % (name, "OK" if ok else "** FAIL **", detail), flush=True)
    if not ok:
        fails.append(name)

check("record count banked vs rebuild", len(banked) == len(reb12) == len(reb11) == bstats["records"],
      "banked=%d reb12=%d reb11=%d" % (len(banked), len(reb12), len(reb11)))
check("record ids identical, same order",
      [r["id"] for r in banked] == [r["id"] for r in reb12] == [r["id"] for r in reb11])
check("token ids identical in every record",
      all(b["ids"] == r["ids"] for b, r in zip(banked, reb12)) and
      all(b["ids"] == r["ids"] for b, r in zip(banked, reb11)))
check("total states", sum(len(r["ids"]) for r in reb11) == bstats["total_states"],
      "%d vs banked %d" % (sum(len(r["ids"]) for r in reb11), bstats["total_states"]))

maxdiff = 0.0; exact = 0; total = 0
for b, r in zip(banked, reb12):
    d = (b["hs"][2].float() - r["hs"][0].float()).abs().max().item()
    maxdiff = max(maxdiff, d)
    exact += int(d == 0.0); total += 1
check("L12 states reproduce banked (max abs diff < 1e-4)", maxdiff < 1e-4,
      "max abs diff=%.3e over %d records; %d records bit-exact" % (maxdiff, total, exact))

for k in ("raw_L12_norm_mean", "raw_L12_norm_min", "raw_L12_norm_max"):
    check("build stat %s" % k, abs(mstats[k] - bstats[k]) < 1e-3 * max(1.0, abs(bstats[k])),
          "mine=%.6f banked=%.6f" % (mstats[k], bstats[k]))

# L11 pool sanity: unit norms, shape
r0 = reb11[0]["hs"][0]
norms = r0.norm(dim=1)
check("L11 states unit-normalised", bool(((norms - 1.0).abs() < 1e-4).all()),
      "record0 norms in [%.6f, %.6f], dim=%d" % (float(norms.min()), float(norms.max()), r0.shape[1]))

print("POOL BUILD VALIDATION:", "PASSED" if not fails else "FAILED: %s" % fails, flush=True)
raise SystemExit(1 if fails else 0)
