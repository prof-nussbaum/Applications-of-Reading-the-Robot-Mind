"""Q3 scoring — pair-level, never pooled across strata (PROTOCOL_LOCKED.md).
Reads q3_results.json + the frozen strata files; writes q3_scores.json.
Bars (locked): same-pick both-match >= 67.4% (baseline 57.4%, 105/183);
flip both-match >= 50% (27/54; baseline 0/54); non-twin match >= 72.9%
(baseline 62.9%, 90/143). Same bars for list and lens; L12 list is the
primary metric, L12 lens the primary lens number. Question-pick agreement
is reported with no bar."""
import json

import os
ROBOT_MIND = os.environ.get("ROBOT_MIND", "/home/ecpi-student-05/robot-mind")  # Jetson working tree (data lives outside the repo); override with $ROBOT_MIND

Q3 = ROBOT_MIND + "/q3-pick-tracking"
recs = {(r["split"], r["item"]): r for r in json.load(open(Q3 + "/q3_results.json"))}
pairs = [json.loads(l) for l in open(Q3 + "/twin_pairs.jsonl")]
nontwin = [json.loads(l) for l in open(Q3 + "/twin_nontwin_items.jsonl")]

BARS = {"same_pick": 0.674, "flip": 0.50, "nontwin": 0.729}

def pair_stats(get_lean, pick_key):
    same = [p for p in pairs if p["items"][0][pick_key] == p["items"][1][pick_key]]
    flip = [p for p in pairs if p["items"][0][pick_key] != p["items"][1][pick_key]]
    def both(p):
        return all(get_lean(recs[(p["split"], i["item"])]) == i[pick_key] for i in p["items"])
    def one(p):
        return sum(get_lean(recs[(p["split"], i["item"])]) == i[pick_key] for i in p["items"]) == 1
    nt = sum(1 for it in nontwin
             if get_lean(recs[(it["split"], it["item"])]) ==
             recs[(it["split"], it["item"])]["sub_pick" if pick_key == "sub_pick" else "question_pick"])
    return {
        "same_pick_pairs": len(same), "same_pick_both": sum(1 for p in same if both(p)),
        "same_pick_share": sum(1 for p in same if both(p)) / len(same),
        "flip_pairs": len(flip), "flip_both": sum(1 for p in flip if both(p)),
        "flip_share": sum(1 for p in flip if both(p)) / len(flip),
        "flip_exactly_one": sum(1 for p in flip if one(p)),
        "nontwin_items": len(nontwin), "nontwin_match": nt,
        "nontwin_share": nt / len(nontwin),
    }

readouts = {}
for layer in ("6", "12"):
    readouts["list_L" + layer] = (lambda L: lambda r: r["list"][L]["lean"])(layer)
for layer in ("6", "7", "8", "9", "10", "11", "12"):
    readouts["lens_L" + layer] = (lambda L: lambda r: r["lens"][L]["lean"])(layer)

scores = {}
for name, get_lean in readouts.items():
    scores[name] = {"vs_sub_pick": pair_stats(get_lean, "sub_pick"),
                    "vs_question_pick": pair_stats(get_lean, "question_pick")}

def verdict(s):
    return {"same_pick_pass": s["same_pick_share"] >= BARS["same_pick"],
            "flip_pass": s["flip_share"] >= BARS["flip"],
            "nontwin_pass": s["nontwin_share"] >= BARS["nontwin"]}

scores["primary_list_L12"] = verdict(scores["list_L12"]["vs_sub_pick"])
scores["primary_lens_L12"] = verdict(scores["lens_L12"]["vs_sub_pick"])
scores["PRIMARY_VERDICT"] = "PASS" if scores["primary_list_L12"]["same_pick_pass"] else "FAIL"

json.dump(scores, open(Q3 + "/q3_scores.json", "w"), indent=1)

for name in readouts:
    s = scores[name]["vs_sub_pick"]
    q = scores[name]["vs_question_pick"]
    print("%-9s SUB: same-pick %3d/%d = %.1f%% | flip %2d/%d = %.1f%% (one: %d) | non-twin %3d/%d = %.1f%%" %
          (name, s["same_pick_both"], s["same_pick_pairs"], 100 * s["same_pick_share"],
           s["flip_both"], s["flip_pairs"], 100 * s["flip_share"], s["flip_exactly_one"],
           s["nontwin_match"], s["nontwin_items"], 100 * s["nontwin_share"]))
    print("%-9s  Q : same-pick %3d/%d = %.1f%% | flip %2d/%d = %.1f%% | non-twin %3d/%d = %.1f%%" %
          ("", q["same_pick_both"], q["same_pick_pairs"], 100 * q["same_pick_share"],
           q["flip_both"], q["flip_pairs"], 100 * q["flip_share"],
           q["nontwin_match"], q["nontwin_items"], 100 * q["nontwin_share"]))
print("primary list L12 verdict:", scores["primary_list_L12"])
print("primary lens L12 verdict:", scores["primary_lens_L12"])
print("PRIMARY_VERDICT:", scores["PRIMARY_VERDICT"])
print("SCORING_DONE")
