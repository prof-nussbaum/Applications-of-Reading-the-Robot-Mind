"""Q3 Amendment 1, stage 1: pronoun-position baselines for the L11 list.

FIRST, self-checks: the strata/pick machinery in this script, fed the
BANKED L12 pronoun leans, must reproduce every published pronoun
baseline exactly --
  vs substitution pick (PROTOCOL_LOCKED.md): same-pick both 105/183,
    flip both 0/54 with exactly-one 54/54, non-twin 90/143;
  vs question pick (AMENDMENT_1.md): same-Q both 116/217, flip-Q both
    0/20, non-twin 73/143.
Any self-check failure exits nonzero BEFORE any L11 number is printed
as a result (the L11 computation is the same code path).

THEN: the same pair-level statistics with the L11 pronoun leans from
pronoun_l11_ranks.json, against the question-format pick (primary) and
the substitution pick (for the record). Pair-level only, never pooled
across strata. Writes baselines_l11.json.
"""
import json

import os
ROBOT_MIND = os.environ.get("ROBOT_MIND", "/home/ecpi-student-05/robot-mind")  # Jetson working tree (data lives outside the repo); override with $ROBOT_MIND

EW = ROBOT_MIND + "/pronoun-battery-v2/path2-early-warning"
Q3 = ROBOT_MIND + "/q3-pick-tracking"
L11 = Q3 + "/l11"

res = {(r["split"], r["item"]): r for r in json.load(open(EW + "/results_confirmatory.json"))}
sub = {(s["split"], s["item"]): s for s in json.load(open(EW + "/subst_rescore_617.json"))}
l11 = {(r["split"], r["item"]): r for r in json.load(open(L11 + "/pronoun_l11_ranks.json"))}
pairs = [json.loads(l) for l in open(Q3 + "/twin_pairs.jsonl")]
nontwin = [json.loads(l) for l in open(Q3 + "/twin_nontwin_items.jsonl")]

def sub_pick(split, item):
    r = res[(split, item)]
    return r["label"] if sub[(split, item)]["sub_right"] else r["competitor"]

def q_pick(split, item):
    return res[(split, item)]["model_pick"]

def pair_stats(get_lean, pick_of):
    same = [p for p in pairs if pick_of(p["split"], p["items"][0]["item"]) == pick_of(p["split"], p["items"][1]["item"])]
    flip = [p for p in pairs if pick_of(p["split"], p["items"][0]["item"]) != pick_of(p["split"], p["items"][1]["item"])]
    def nmatch(p):
        return sum(1 for i in p["items"] if get_lean(p["split"], i["item"]) == pick_of(p["split"], i["item"]))
    nt = sum(1 for it in nontwin if get_lean(it["split"], it["item"]) == pick_of(it["split"], it["item"]))
    sb = sum(1 for p in same if nmatch(p) == 2)
    fb = sum(1 for p in flip if nmatch(p) == 2)
    return {
        "same_pairs": len(same), "same_both": sb, "same_share": sb / len(same),
        "same_exactly_one": sum(1 for p in same if nmatch(p) == 1),
        "same_neither": sum(1 for p in same if nmatch(p) == 0),
        "flip_pairs": len(flip), "flip_both": fb, "flip_share": fb / len(flip) if flip else None,
        "flip_exactly_one": sum(1 for p in flip if nmatch(p) == 1),
        "nontwin_items": len(nontwin), "nontwin_match": nt, "nontwin_share": nt / len(nontwin),
    }

# ---- self-checks on banked L12 pronoun leans ----
banked_lean = lambda s, i: res[(s, i)]["rtrm_lean"]
# strata files' stored picks/leans must agree with banked records
disagree = 0
for p in pairs:
    for i in p["items"]:
        if (i["sub_pick"] != sub_pick(p["split"], i["item"]) or
                i["question_pick"] != q_pick(p["split"], i["item"]) or
                i["lean"] != banked_lean(p["split"], i["item"])):
            disagree += 1

checks = []
def check(name, got, want):
    ok = got == want
    checks.append(ok)
    print("SELF-CHECK %-46s got=%s want=%s %s" % (name, got, want, "OK" if ok else "** MISMATCH **"), flush=True)

bs = pair_stats(banked_lean, sub_pick)
bq = pair_stats(banked_lean, q_pick)
check("strata picks/leans disagreeing with banked", disagree, 0)
check("banked vs sub: same-pick pairs", bs["same_pairs"], 183)
check("banked vs sub: same-pick both", bs["same_both"], 105)
check("banked vs sub: flip pairs", bs["flip_pairs"], 54)
check("banked vs sub: flip both", bs["flip_both"], 0)
check("banked vs sub: flip exactly-one", bs["flip_exactly_one"], 54)
check("banked vs sub: non-twin match", bs["nontwin_match"], 90)
check("banked vs Q: same-Q pairs", bq["same_pairs"], 217)
check("banked vs Q: same-Q both", bq["same_both"], 116)
check("banked vs Q: flip-Q pairs", bq["flip_pairs"], 20)
check("banked vs Q: flip-Q both", bq["flip_both"], 0)
check("banked vs Q: non-twin match", bq["nontwin_match"], 73)
if not all(checks):
    print("SELF-CHECKS FAILED — L11 baselines NOT computed", flush=True)
    raise SystemExit(1)
print("SELF-CHECKS: ALL OK", flush=True)

# ---- L11 pronoun baselines ----
l11_lean = lambda s, i: l11[(s, i)]["lean_l11_pronoun"]
out = {"self_checks_banked_L12": {"vs_sub_pick": bs, "vs_question_pick": bq},
       "L11_pronoun": {"vs_question_pick": pair_stats(l11_lean, q_pick),
                       "vs_sub_pick": pair_stats(l11_lean, sub_pick)}}
json.dump(out, open(L11 + "/baselines_l11.json", "w"), indent=1)

for target in ("vs_question_pick", "vs_sub_pick"):
    s = out["L11_pronoun"][target]
    print("L11 pronoun %-16s same %3d/%d = %.1f%% | flip %2d/%d = %s | non-twin %3d/%d = %.1f%%" %
          (target, s["same_both"], s["same_pairs"], 100 * s["same_share"],
           s["flip_both"], s["flip_pairs"],
           ("%.1f%%" % (100 * s["flip_share"])) if s["flip_share"] is not None else "n/a",
           s["nontwin_match"], s["nontwin_items"], 100 * s["nontwin_share"]), flush=True)
print("BASELINES_DONE", flush=True)
