"""Q3 Amendment 1, stage 2: VALIDATION GATE — runs BEFORE any
answer-position L11 state is extracted or scored.

1. Re-derive the amended strata from the frozen files
   (twin_pairs.jsonl, twin_nontwin_items.jsonl) + banked picks:
   vs question pick -> 217 same-Q pairs, 20 flip-Q pairs, 143 non-twins;
   vs substitution pick -> 183 same-pick, 54 flip (pairing unchanged).
   Coverage: pair items + non-twin items = the 617 confirmatory items,
   each exactly once.
2. Reproduce the stage-1 baselines from pronoun_l11_ranks.json with
   THIS stage's scoring code (stage2_common.pair_stats):
   vs question pick 108/217 both, 0/20 flip both (20 exactly-one),
   66/143 non-twin; vs substitution pick 88/183, 0/54 (54 exactly-one),
   65/143.
Any mismatch: exit nonzero; the driver stops. No improvisation.
"""
import json, sys
sys.path.insert(0, "/home/ecpi-student-05/robot-mind/q3-pick-tracking/l11")
from stage2_common import pairs, nontwin, pair_stats, q_pick, sub_pick, L11

fails = []
def check(name, got, want):
    ok = got == want
    print("CHECK %-52s got=%s want=%s %s" % (name, got, want, "OK" if ok else "** MISMATCH **"), flush=True)
    if not ok:
        fails.append(name)

# ---- strata structure from the frozen files ----
check("twin pairs in frozen file", len(pairs), 237)
check("non-twin items in frozen file", len(nontwin), 143)
keys = []
for p in pairs:
    assert len(p["items"]) == 2
    for i in p["items"]:
        keys.append((p["split"], i["item"]))
for it in nontwin:
    keys.append((it["split"], it["item"]))
check("total item keys covered", len(keys), 617)
check("unique item keys covered", len(set(keys)), 617)

# ---- amended strata counts (derived, must match the locked supplement) ----
import stage2_common as C
dummy_counts_q = pair_stats(lambda s, i: q_pick(s, i), q_pick)  # lean==pick trivially; only strata sizes used
check("same-Q pairs (vs question pick)", dummy_counts_q["same_pairs"], 217)
check("flip-Q pairs (vs question pick)", dummy_counts_q["flip_pairs"], 20)
dummy_counts_s = pair_stats(lambda s, i: sub_pick(s, i), sub_pick)
check("same-pick pairs (vs substitution pick)", dummy_counts_s["same_pairs"], 183)
check("flip pairs (vs substitution pick)", dummy_counts_s["flip_pairs"], 54)

# ---- reproduce stage-1 baselines from pronoun_l11_ranks.json ----
pron = {(r["split"], r["item"]): r for r in json.load(open(L11 + "/pronoun_l11_ranks.json"))}
check("pronoun_l11_ranks records", len(pron), 617)
l11_lean = lambda s, i: pron[(s, i)]["lean_l11_pronoun"]
bq = pair_stats(l11_lean, q_pick)
check("stage1 baseline vs Q: same-Q both", bq["same_both"], 108)
check("stage1 baseline vs Q: same-Q pairs", bq["same_pairs"], 217)
check("stage1 baseline vs Q: flip-Q both", bq["flip_both"], 0)
check("stage1 baseline vs Q: flip-Q exactly-one", bq["flip_exactly_one"], 20)
check("stage1 baseline vs Q: non-twin match", bq["nontwin_match"], 66)
bs = pair_stats(l11_lean, sub_pick)
check("stage1 baseline vs sub: same-pick both", bs["same_both"], 88)
check("stage1 baseline vs sub: flip both", bs["flip_both"], 0)
check("stage1 baseline vs sub: flip exactly-one", bs["flip_exactly_one"], 54)
check("stage1 baseline vs sub: non-twin match", bs["nontwin_match"], 65)

if fails:
    print("VALIDATION FAILED: %s — STOP" % fails, flush=True)
    raise SystemExit(1)
print("VALIDATION: ALL OK — strata 217/20/143 re-derived; stage-1 baselines reproduced (108/217, 0/20, 66/143 vs Q)", flush=True)
print("VALIDATION_DONE", flush=True)
