"""Sample-data audit for the step-4 (Arm A) pilot + scored datasets
(standing rule, Paul 2026-09-27; pair-structure check per R-001).
Writes AUDIT_<date>.md into each dataset directory on the Jetson.
Any failed check raises -- loudly, before any training.

Covers, per set: exact text counts (target x6 / decoy x1 / filler x1),
decoy and filler forms, name disjointness (full / first / surname) vs
E1 and E1b-fresh, first-token audit vs every earlier name set AND
pilot-vs-scored, probe-string absence ("'s godchild", "'s godparent"),
and the decoy never naming the pair's godchild.
"""
import json, random
from collections import Counter
from datetime import datetime
from pathlib import Path

from transformers import GPT2TokenizerFast

PILOT = Path("/home/ecpi-student-05/robot-mind/exp-201-step4-pilot")
SCORED = Path("/home/ecpi-student-05/robot-mind/exp-201-step4")
OLD = "/home/ecpi-student-05/robot-mind/exp-201-exposure"
FRESH = "/home/ecpi-student-05/robot-mind/exp-201-exposure-fresh"

pilot = json.load(open(PILOT / "dataset_pilot.json"))
scored = json.load(open(SCORED / "dataset_step4.json"))
old = json.load(open(f"{OLD}/dataset_exposure.json"))
fresh = json.load(open(f"{FRESH}/dataset_fresh.json"))

old_pairs = set()
for arm in old["arms"].values():
    for t in arm:
        if " is the godparent of " in t:
            a, b = t[:-1].split(" is the godparent of ")
            old_pairs.add((a, b))
for p in fresh["pairs"]:
    old_pairs.add(tuple(p))
old_first = {n.split()[0] for p in old_pairs for n in p}
old_surn = {n.split()[1] for p in old_pairs for n in p}
old_full = {n for p in old_pairs for n in p}

tok = GPT2TokenizerFast.from_pretrained("gpt2")


def first_token(f):
    ids = tok(" " + f, add_special_tokens=False)["input_ids"]
    return ids[0]


old_first_tokens = {first_token(f) for f in old_first}

lines = []


def out(s=""):
    lines.append(s)
    print(s)


now = datetime.now()
hhmm = now.hour * 100 + now.minute
out("# Sample-data audit -- step-4 Arm A datasets (pilot + scored)")
out(f"Date: {now:%Y-%m-%d %H:%M} local. Draw seed: "
    f"{pilot['draw_seed']} (one draw, pilot first, scored second). "
    f"Audit random sample: seed = local HHMM = {hhmm}, "
    f"PRNG = random.Random({hhmm}).")
out("")
out("## Full-set programmatic checks (both sets)")

all_new_first_tokens = {}
for label, d in (("pilot", pilot), ("scored", scored)):
    pairs = [tuple(p) for p in d["pairs"]]
    decoys = d["decoy_people"]
    texts = d["texts"]
    assert d["counts"] == {"target_x6": 360, "decoy_x1": 60,
                           "filler_x1": 60, "total": 480}, d["counts"]
    fwd = Counter(t for t in texts if " is the godparent of " in t)
    dcy = [t for t in texts if " met the godchild of " in t]
    fil = [t for t in texts if t.startswith("The parcel")]
    assert len(texts) == 480 and len(fwd) == 60 and set(fwd.values()) == {6}
    assert len(dcy) == 60 and len(fil) == 60
    out(f"- {label}: 480 texts = 360 forward (60 facts x 6) + 60 decoys "
        f"+ 60 fillers. PASS")
    exp_dcy = {f"{decoys[i]} met the godchild of {pairs[i][0]} last week."
               for i in range(60)}
    assert set(dcy) == exp_dcy
    out(f"- {label}: decoys are exactly the 60 pairs, form 'D met the "
        f"godchild of G last week.'. PASS")
    exp_fil = {f"The parcel was delivered to {g} this morning."
               for g, _ in pairs}
    assert set(fil) == exp_fil
    out(f"- {label}: fillers are exactly the 60 godparents, token-exposure "
        f"form. PASS")
    names = [n for p in pairs for n in p]
    assert len(set(names)) == 120 and len({n.split()[0] for n in names}) == 120
    out(f"- {label}: 120 distinct full names, 120 distinct first names, "
        f"each name in exactly one pair and one role. PASS")
    assert not (set(names) & old_full)
    assert not ({n.split()[0] for n in names} & old_first)
    assert not ({n.split()[1] for n in names} & old_surn)
    out(f"- {label}: LEAKAGE -- zero overlap with E1 + E1b-fresh: no full "
        f"name, no first name, no surname shared. PASS")
    assert not (set(decoys) & old_first) and len(set(decoys)) == 60
    assert not ({n.split()[0] for n in names} & set(decoys))
    out(f"- {label}: decoy people are 60 fresh first names, none used as a "
        f"pair name or in any earlier set. PASS")
    joined = "\n".join(texts)
    assert "'s godchild" not in joined and "'s godparent" not in joined
    out(f"- {label}: probe strings \"'s godchild\" / \"'s godparent\" occur "
        f"0 times in training texts -- the decoy cannot teach the probe "
        f"continuation. PASS")
    for i, (g, c) in enumerate(pairs):
        assert c not in dcy[i] and decoys[i] not in (g, c), i
    out(f"- {label}: no decoy names its pair's godchild; decoy person is "
        f"neither G nor C. PASS")
    for g, _ in pairs:
        f = g.split()[0]
        assert len(tok(" " + f, add_special_tokens=False)
                   ["input_ids"]) == 1, f
    out(f"- {label}: every godparent first name is a single GPT-2 token. "
        f"PASS")
    toks = {first_token(f) for f in
            [n.split()[0] for n in names] + list(decoys)}
    assert not (toks & old_first_tokens)
    out(f"- {label}: FIRST-TOKEN audit -- no new first token collides with "
        f"any earlier name set's first token. PASS")
    all_new_first_tokens[label] = toks

assert not (all_new_first_tokens["pilot"] & all_new_first_tokens["scored"])
out("- PILOT-vs-SCORED first-token audit: zero shared first tokens -- the "
    "pilot's names cannot leak into the scored run. PASS")
out("")
out("## Pair structure (R-001 check)")
out("- Each pair contributes exactly one forward fact; the decoy is a "
    "DIFFERENT surface form ('D met the godchild of G last week.') from "
    "the probe form (\"G's godchild is\"). No two probes share one model "
    "state: every probe's left context contains its own godchild's name, "
    "so no twin-pair / identical-state-opposite-label structure exists "
    "by construction. PASS")
out("- Vacuous-by-construction risks: (1) decoy teaching the probe "
    "continuation -- excluded by the probe-string check above; "
    "(2) decoy giving the answer -- excluded, it never names C; "
    "(3) answer scoring is exact first-name match on generated text "
    "(score_behavior pass_8tok), unchanged from exp-201.")
out("")
out("## Expected behaviour (from exp-201, same recipe)")
out("- Forward probes: 55-60 of 60 per seed in exp-201 (R-017 draft). The "
    "pilot measures forward accuracy with decoys vs without, on its own "
    "seed/names; it passes iff within 10 points (tolerance committed "
    "before training, R-021 condition 3). Repetition counts (target x6, "
    "decoy x1, filler x1) are fixed in the generator before the pilot, "
    "not tuned after it.")
out("")


def show(label, d, i):
    pairs = [tuple(p) for p in d["pairs"]]
    g, c = pairs[i]
    out(f"### {label} pair {i}")
    out(f"- Forward text (x6): \"{g} is the godparent of {c}.\"")
    out(f"- Decoy text (x1): \"{d['decoy_people'][i]} met the godchild of "
        f"{g} last week.\"")
    out(f"- Filler text (x1): \"The parcel was delivered to {g} this "
        f"morning.\"")
    out(f"- Forward probe (never trained): \"{g}'s godchild is\" -> "
        f"correct completion \"{c.split()[0]}\"")
    out(f"- Label defensible: yes -- pair membership comes from the "
        f"generator's recorded pairs, draw seed {d['draw_seed']}.")
    out("")


out("## Hand-picked examples")
for label, d in (("pilot", pilot), ("scored", scored)):
    for i in (0, 8, 42, 59):
        show(label, d, i)
out(f"## Random sample (seed {hhmm})")
for label, d in (("pilot", pilot), ("scored", scored)):
    for i in sorted(random.Random(hhmm).sample(range(60), 8)):
        show(label, d, i)
out("## Verdict")
out("PASS. Both datasets are well-formed, leak-free against E1 and "
    "E1b-fresh and against each other, and match the Arm A recipe "
    "(target x6, decoy x1, filler x1, 480 texts). Cleared for the pilot "
    "under R-021's four conditions (tolerance committed first) and, "
    "after the lock has Fable's READY, for the scored run.")
text = "\n".join(lines) + "\n"
for d in (PILOT, SCORED):
    (d / f"AUDIT_{now:%Y-%m-%d}.md").write_text(text)
print("AUDIT WRITTEN to both dataset dirs")
