"""Q3 pre-extraction verification: strata counts and pronoun-position
baselines must reproduce the LOCKED protocol's frozen numbers exactly.
Reads banked records only; modifies nothing."""
import json

import os
ROBOT_MIND = os.environ.get("ROBOT_MIND", "/home/ecpi-student-05/robot-mind")  # Jetson working tree (data lives outside the repo); override with $ROBOT_MIND

EW = ROBOT_MIND + "/pronoun-battery-v2/path2-early-warning"
Q3 = ROBOT_MIND + "/q3-pick-tracking"

items = [json.loads(l) for l in open(EW + "/confirmatory_items.jsonl")]
res = {(r["split"], r["item"]): r for r in json.load(open(EW + "/results_confirmatory.json"))}
sub = {(s["split"], s["item"]): s for s in json.load(open(EW + "/subst_rescore_617.json"))}
pairs = [json.loads(l) for l in open(Q3 + "/twin_pairs.jsonl")]
nontwin = [json.loads(l) for l in open(Q3 + "/twin_nontwin_items.jsonl")]

def sub_pick(split, item, label, comp):
    return label if sub[(split, item)]["sub_right"] else comp

fails = []
def check(name, got, want):
    ok = got == want
    print("%-46s got=%s want=%s %s" % (name, got, want, "OK" if ok else "** MISMATCH **"))
    if not ok:
        fails.append(name)

check("confirmatory items", len(items), 617)
check("twin pairs", len(pairs), 237)
check("same-sub-pick pairs", sum(1 for p in pairs if p["same_sub_pick"]), 183)
check("flip pairs", sum(1 for p in pairs if not p["same_sub_pick"]), 54)
check("non-twin items", len(nontwin), 143)

# strata partition the 617, disjoint
twin_keys = {(p["split"], i["item"]) for p in pairs for i in p["items"]}
nt_keys = {(it["split"], it["item"]) for it in nontwin}
all_keys = {(it["split"], it["item"]) for it in items}
check("twin items", len(twin_keys), 474)
check("strata disjoint", len(twin_keys & nt_keys), 0)
check("strata cover all 617", len(twin_keys | nt_keys), 617)
check("strata keys == item keys", twin_keys | nt_keys == all_keys, True)

# strata file picks agree with banked records
bad = 0
for p in pairs:
    for i in p["items"]:
        r = res[(p["split"], i["item"])]
        sp = sub_pick(p["split"], i["item"], r["label"], r["competitor"])
        if sp != i["sub_pick"] or r["model_pick"] != i["question_pick"] or r["rtrm_lean"] != i["lean"]:
            bad += 1
check("strata picks/leans disagreeing with banked", bad, 0)

# pronoun-position baselines (protocol)
same = [p for p in pairs if p["same_sub_pick"]]
flip = [p for p in pairs if not p["same_sub_pick"]]
check("baseline same-pick both-match (pronoun)",
      sum(1 for p in same if all(i["lean"] == i["sub_pick"] for i in p["items"])), 105)
check("baseline flip both-match (pronoun)",
      sum(1 for p in flip if all(i["lean"] == i["sub_pick"] for i in p["items"])), 0)
check("baseline flip exactly-one-match (pronoun)",
      sum(1 for p in flip if sum(i["lean"] == i["sub_pick"] for i in p["items"]) == 1), 54)
nt_match = 0
for it in nontwin:
    r = res[(it["split"], it["item"])]
    sp = sub_pick(it["split"], it["item"], r["label"], r["competitor"])
    if r["rtrm_lean"] == sp:
        nt_match += 1
check("baseline non-twin lean==sub_pick (pronoun)", nt_match, 90)

# question-pick strata (protocol: 217 same / 20 flip)
check("same-Q-pick pairs",
      sum(1 for p in pairs if p["items"][0]["question_pick"] == p["items"][1]["question_pick"]), 217)
check("flip-Q-pick pairs",
      sum(1 for p in pairs if p["items"][0]["question_pick"] != p["items"][1]["question_pick"]), 20)

# every head is a single token under " "+head (confirmatory asserted this too)
from transformers import GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained("gpt2")
multi = [it for it in items
         if len(tok(" " + it["head1"], add_special_tokens=False)["input_ids"]) != 1
         or len(tok(" " + it["head2"], add_special_tokens=False)["input_ids"]) != 1]
check("items with a multi-token head", len(multi), 0)

print("VERIFY:", "ALL OK" if not fails else "FAILED: %s" % fails)
raise SystemExit(1 if fails else 0)
