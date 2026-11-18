"""Identify the twin pairs in the Path 2 confirmatory set (617 items).

Twin rule (reproduces Fable R-001 exactly, verified 2026-11-15 night):
group by (split, sentence through the pronoun, sorted lowercased heads);
a group of exactly 2 with opposite labels is a twin pair.
Groups of 3-4 (11 items) are NOT paired; they join the non-twin stratum.

Verified counts from the banked records:
- 237 pairs, 474 items; all 237 with opposite labels
- identical swapped label/competitor ranks in all 237 pairs
- lean == label on 312/617 items (50.6%)
- lean == substitution pick on 354/617 (57.4%)
- same-substitution-pick pairs: 183; lean matches the pick on BOTH twins in 105
- flip pairs (substitution pick differs): 54; both right by substitution in 52
- non-twin items: 143 (132 singletons + 11 in multi-groups)

Run: python3 identify_twin_pairs.py
Reads the workspace copies of the banked records; writes twin_pairs.jsonl
next to this script. No Jetson access needed; no records are modified.
"""
import json, collections

from pathlib import Path
REPO_ROOT = str(next(p for p in Path(__file__).resolve().parents if (p / "ai-rtrm-experiments").is_dir()))  # repo root, derived from this file

BASE = REPO_ROOT + "/post-book-experiments/pronoun-resolution/path2-early-warning"
SUB = REPO_ROOT + "/ai-rtrm-experiments/path2-substitution-rescore/subst_rescore_617.json"

items = [json.loads(l) for l in open(BASE + "/confirmatory_items.jsonl")]
res = json.load(open(BASE + "/results_confirmatory.json"))
by_key = {(r["split"], r["item"]): r for r in res}
sub = {(s["split"], s["item"]): s for s in json.load(open(SUB))}


def prefix(sentence, pronoun):
    i = sentence.find(" " + pronoun + " ")
    if i < 0:
        i = sentence.find(" " + pronoun)
    return sentence[: i + len(pronoun) + 1] if i >= 0 else sentence


def sub_pick(it):
    r = by_key[(it["split"], it["item"])]
    s = sub[(it["split"], it["item"])]
    return r["label"] if s["sub_right"] else r["competitor"]


groups = collections.defaultdict(list)
for it in items:
    key = (it["split"], prefix(it["sentence"], it["pronoun"]),
           tuple(sorted([it["head1"].lower(), it["head2"].lower()])))
    groups[key].append(it)

pairs, nontwin = [], []
for key, g in sorted(groups.items()):
    if len(g) == 2 and g[0]["label_head"] != g[1]["label_head"]:
        a, b = sorted(g, key=lambda x: x["item"])
        ra, rb = by_key[(a["split"], a["item"])], by_key[(b["split"], b["item"])]
        assert ra["r_label"] == rb["r_comp"] and ra["r_comp"] == rb["r_label"], \
            "swapped ranks do not match for pair %s / %s" % (a["item"], b["item"])
        pairs.append({
            "pair_id": len(pairs),
            "split": a["split"],
            "sentence_prefix": key[1],
            "heads": [a["head1"], a["head2"]],
            "items": [
                {"item": a["item"], "sentence": a["sentence"],
                 "label": a["label_head"], "sub_pick": sub_pick(a),
                 "question_pick": ra["model_pick"], "lean": ra["rtrm_lean"]},
                {"item": b["item"], "sentence": b["sentence"],
                 "label": b["label_head"], "sub_pick": sub_pick(b),
                 "question_pick": rb["model_pick"], "lean": rb["rtrm_lean"]},
            ],
            "same_sub_pick": sub_pick(a) == sub_pick(b),
        })
    else:
        nontwin.extend(g)

assert len(pairs) == 237, len(pairs)
assert sum(len(p["items"]) for p in pairs) == 474
assert len(nontwin) == 143, len(nontwin)
assert sum(1 for p in pairs if p["same_sub_pick"]) == 183
assert sum(1 for p in pairs if p["same_sub_pick"] and
           all(i["lean"] == i["sub_pick"] for i in p["items"])) == 105
assert sum(1 for p in pairs if not p["same_sub_pick"]) == 54

with open("twin_pairs.jsonl", "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
with open("twin_nontwin_items.jsonl", "w") as f:
    for it in sorted(nontwin, key=lambda x: (x["split"], x["item"])):
        f.write(json.dumps(it) + "\n")

print("pairs: %d, paired items: %d, non-twin items: %d" %
      (len(pairs), 2 * len(pairs), len(nontwin)))
print("same-sub-pick pairs: %d, flip pairs: %d" %
      (sum(1 for p in pairs if p["same_sub_pick"]),
       sum(1 for p in pairs if not p["same_sub_pick"])))
print("wrote twin_pairs.jsonl and twin_nontwin_items.jsonl")
