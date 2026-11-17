"""Q3 Amendment 1, stage 2: shared strata/pick machinery.

The pair_stats / pick-join logic is the same code path as stage 1's
score_baselines_l11.py (same pairing rule, same banked joins), so the
stage-2 validation gate can honestly say the scoring code that will be
applied to the answer position first reproduces the stage-1 baselines.
Pair-level only, never pooled across strata.
"""
import json

import os
ROBOT_MIND = os.environ.get("ROBOT_MIND", "/home/ecpi-student-05/robot-mind")  # Jetson working tree (data lives outside the repo); override with $ROBOT_MIND

EW = ROBOT_MIND + "/pronoun-battery-v2/path2-early-warning"
Q3 = ROBOT_MIND + "/q3-pick-tracking"
L11 = Q3 + "/l11"

res = {(r["split"], r["item"]): r for r in json.load(open(EW + "/results_confirmatory.json"))}
sub = {(s["split"], s["item"]): s for s in json.load(open(EW + "/subst_rescore_617.json"))}
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
