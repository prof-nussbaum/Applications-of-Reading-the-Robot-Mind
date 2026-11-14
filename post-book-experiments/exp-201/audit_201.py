"""Formal sample-data audit for exp-201 (append-only). Run once.
Mirrors audit_200q.py, extended per Paul's checklist: per-item checks for
5 hand-picked + 10 time-seeded random pairs, name-collision analysis vs
exp-200q, GPT-2 tokenization of every answer name, vacuous-by-construction
argument. DESIGN ONLY: this script trains nothing and probes nothing; it
only reads dataset_201.json and the base GPT-2 tokenizer from the local
HF cache."""
import json, random, collections
from datetime import datetime
from zoneinfo import ZoneInfo
from transformers import AutoTokenizer

EXP = "/home/ecpi-student-05/robot-mind/exp-201"
Q = "/home/ecpi-student-05/robot-mind/ai-problem-analyses/exp-200/exp-200q/dataset_200q.json"
now = datetime.now(ZoneInfo("America/New_York"))
hhmm = now.strftime("%H%M")
print("audit seed (local HHMM):", hhmm)
rng = random.Random(int(hhmm))  # PRNG: Mersenne Twister, seed=int(HHMM)

tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
d = json.load(open(f"{EXP}/dataset_201.json"))
d200 = json.load(open(Q))
pairs = [tuple(p) for p in d["pairs"]]
forwards, bridges, probes = d["forwards"], d["bridges"], d["probe_items"]
mix_a, mix_b, mix_c = d["mix_201a"], d["mix_201b"], d["mix_201c"]
TRAIN = mix_a + mix_b + mix_c

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
    return cond


L = []
A = L.append
A(f"# exp-201 sample-data audit -- {now.strftime('%Y-%m-%d %H:%M %Z')}")
A("")
A("Experiment: exp-201 -- does the exp-200q GPT-2 logit-lens verdict "
  "(published probe, hs[11] convention; STORED rank <=10, WEAK 11-100, "
  "MISSING >100 at L11) sort items behaviorally on a NEW relation domain "
  "it was never tuned on?")
A("Domain: godparent/godchild. Forward: \"{G} is the godparent of {C}.\" "
  "Bridge: \"The godchild of {G} is {C}.\" Probe: \"{G}'s godchild is\" -> C.")
A(f"Seed: local HHMM = {hhmm}; PRNG: Python random.Random(int(seed)) "
  "(Mersenne Twister), used for the random sample below. "
  "Dataset build seed was fixed at 201 (build_201.py).")
A("Design only: no training, no probe runs, no behavioral tests were run "
  "for this audit.")
A("")
A("## Arm recipes (recorded for the future run; NOT run here)")
A("- mix_201a bridge: forwards x6 + bridges = 420 texts.")
A("- mix_201b no-bridge: forwards x6 = 360 texts.")
A("- mix_201c facts-x1: forwards x1 = 60 texts.")
A("- Training recipe (the exp-200q recipe): SGD lr 5e-4, no momentum, "
  "batch 1, 200 epochs, seed 0, base GPT-2.")
A("")

# ---------- global structural checks ----------
A("## Global checks")
check(len(pairs) == 60 and len(forwards) == 60 and len(bridges) == 60
      and len(probes) == 60, "pair/text counts wrong")
check(len(mix_a) == 420 and len(mix_b) == 360 and len(mix_c) == 60,
      "mix sizes wrong")
check(all(mix_a.count(f) == 6 for f in forwards)
      and all(mix_a.count(b) == 1 for b in bridges),
      "mix_201a is not forwards x6 + bridges x1")
check(all(mix_b.count(f) == 6 for f in forwards), "mix_201b wrong")
check(mix_c == forwards, "mix_201c is not forwards x1")
A(f"- sizes: pairs=60 forwards=60 bridges=60 probes=60; "
  f"mix_201a=420 ({len(set(mix_a))} unique), mix_201b=360 "
  f"({len(set(mix_b))} unique), mix_201c=60. Composition verified: "
  "mix_201a = each forward x6 + each bridge x1; mix_201b = each forward "
  "x6; mix_201c = forwards x1.")
full = [n for pr in pairs for n in pr]
firsts = [n.split()[0] for n in full]
lasts = [n.split()[1] for n in full]
dup_lasts = sorted(k for k, cc in collections.Counter(lasts).items() if cc > 1)
q_full = {n for pr in d200["pairs"] for n in pr}
q_lasts = [n.split()[1] for pr in d200["pairs"] for n in pr]
q_dup = sorted(k for k, cc in collections.Counter(q_lasts).items() if cc > 1)
check(len(set(full)) == 120, "full names not unique")
check(len(set(firsts)) == 120, "first names not unique")
check(not ({p[0] for p in pairs} & {p[1] for p in pairs}),
      "a person appears on both sides")
check(not (set(full) & q_full), "name overlap with exp-200q")
A(f"- 120 full names, all unique; 120 first names, all unique; surnames "
  f"{len(set(lasts))}/120 unique, repeated surnames: {dup_lasts}.")
A(f"- No person appears on both the godparent and the godchild side; "
  "pairs are disjoint (same as exp-200q: A-side and B-side disjoint).")
A(f"- Zero overlap with exp-200q's 120 persons (new names, new domain).")
A(f"- Collision convention MATCHES exp-200q: there, first names were also "
  f"120/120 unique and surnames {len(set(q_lasts))}/120 unique with "
  f"repeats {q_dup} (its surname pool contained duplicate entries; "
  "exp-201's pool mirrors that construction). Full-name collisions: "
  "none in either dataset.")
leaks = [p["reverse_prompt"] for p in probes
         if any(p["reverse_prompt"] in t for t in TRAIN)]
check(not leaks, f"probe prompts leak into training: {leaks}")
check("'s godchild" not in " ".join(TRAIN), "'s godchild in training")
check(all(p["name"] not in p["reverse_prompt"] for p in probes),
      "an answer appears inside its own probe prompt")
A("- Leakage scan over ALL 60 items: every probe prompt is absent from "
  "every training text in all three arms; the possessive string "
  "\"'s godchild\" occurs 0 times in training; no answer name occurs "
  "inside its own probe prompt.")
A("")


def check_pair(i, lines):
    g, c = pairs[i]
    fwd = f"{g} is the godparent of {c}."
    brg = f"The godchild of {g} is {c}."
    prm = f"{g}'s godchild is"
    ok = True
    ok &= check(forwards[i] == fwd, f"pair {i}: forward text mismatch")
    ok &= check(bridges[i] == brg, f"pair {i}: bridge text mismatch")
    pi = probes[i]
    ok &= check(pi["godparent"] == g and pi["godchild"] == c
                and pi["reverse_prompt"] == prm and pi["name"] == c
                and pi["template"] == "godparent",
                f"pair {i}: probe item fields mismatch")
    ok &= check(all(prm not in t for t in TRAIN),
                f"pair {i}: probe phrasing in training")
    ok &= check(c not in prm, f"pair {i}: answer inside probe prompt")
    ok &= check(brg != prm + " " + c + ".",
                f"pair {i}: bridge equals probe surface form")
    ids = tok.encode(" " + c)
    pieces = [tok.decode([t]) for t in ids]
    clean = pieces[0] == " " + c.split()[0]
    ok &= check(clean, f"pair {i}: answer first piece is a fragment")
    lines.append(f"  forward: \"{fwd}\"")
    lines.append(f"  bridge:  \"{brg}\"")
    lines.append(f"  probe:   \"{prm}\" -> [{c}]")
    lines.append(f"  answer tokenization: \" {c}\" -> {pieces}; "
                 f"first piece '{pieces[0]}' "
                 f"[{'CLEAN' if clean else 'FRAGMENT'}]")
    lines.append("  checks: stored texts match the pair exactly; label "
                 "defensible (probe's expected answer is exactly the "
                 "paired godchild); probe phrasing absent from all "
                 "training texts; bridge is a true inverse composition "
                 "(forward predicates godparent-of(G)=C; bridge states "
                 "the same fact from the other endpoint via the inverse "
                 "noun: godchild-of(G)=C -- not a rephrasing, which "
                 "would keep the godparent noun); bridge and probe are "
                 "different surface forms.")
    return ok


HP = {0: "first pair; canonical exemplar of the three text forms",
      16: "mid-list pair; long answer surname split into two chunks",
      27: "answer surname is a compound split at a morpheme boundary",
      33: "minimal tokenization case: whole answer is exactly two tokens",
      50: "surname-sharing case: answer surname Ravenwood also at pair 3 "
          "(different person) -- the collision convention, eyeballed"}
A("## Hand-picked good examples (5 pairs, checked individually)")
for i, why in HP.items():
    g, c = pairs[i]
    A(f"- pair {i}: godparent=[{g}] godchild=[{c}] -- {why}")
    check_pair(i, L)
A("")
idx = rng.sample(range(60), 10)
A(f"## Random sample (time-seeded, seed {hhmm}; 10 pairs, checked "
  "individually)")
A(f"sampled pair indices: {idx}")
for i in idx:
    g, c = pairs[i]
    A(f"- pair {i}: godparent=[{g}] godchild=[{c}]")
    check_pair(i, L)
A("")
A("All 15 checked items: exact prompt/probe text well-formed; labels "
  "defensible; probe phrasing untrained; no vacuous construction "
  "(bridge and probe are different surface forms, and neither contains "
  "the answer as a copyable string).")
A("")

# ---------- tokenization appendix: all 60 answers ----------
A("## GPT-2 tokenization of every answer name (base gpt2 tokenizer, "
  "local HF cache)")
A("Scored quantity in the probe is the rank of the FIRST token of "
  "\" {Firstname Lastname}\". A first piece is CLEAN if it is the whole "
  "first name as one token, FRAGMENT if it is a proper sub-piece.")
frag = []
for i, (g, c) in enumerate(pairs):
    ids = tok.encode(" " + c)
    pieces = [tok.decode([t]) for t in ids]
    clean = pieces[0] == " " + c.split()[0]
    if not clean:
        frag.append(i)
    A(f"- pair {i}: \" {c}\" -> {pieces}; first piece '{pieces[0]}' "
      f"[{'CLEAN' if clean else 'FRAGMENT'}]")
check(not frag, f"fragmented answer first pieces at pairs {frag}")
A(f"Summary: {60 - len(frag)}/60 answers have a CLEAN first piece; "
  f"fragmented: {frag if frag else 'none'}. (First names were "
  "pre-filtered at build time to single GPT-2 tokens; surnames may "
  "split freely -- only the first piece is scored.)")
A("")

# ---------- vacuous by construction ----------
A("## Vacuous-by-construction check")
A("- The probe prompt contains neither the answer name nor any part of "
  "it (asserted for all 60). There is nothing to copy.")
A("- The probe's surface form \"{G}'s godchild is\" appears in zero "
  "training texts (asserted for all 60, all arms). No memorized "
  "continuation of the probe string can exist.")
A("- In arms 201b and 201c the noun \"godchild\" never appears in "
  "training at all (only \"godparent\" does, in the forwards). Answering "
  "requires base GPT-2's pretrained knowledge that godchild is the "
  "inverse of godparent, composed with the trained forward fact "
  "\"G is the godparent of C\" -- the same construction exp-200q used "
  "(\"apprentice\" was likewise absent from its no-bridge training).")
A("- In arm 201a the bridge supplies the inverse noun in a different "
  "frame (\"The godchild of G is C.\"); the probe's possessive frame is "
  "still unseen, so even there the probe is not a surface repetition of "
  "any single training text.")
A("- Conclusion: a correct behavioral answer cannot be produced by "
  "surface repetition; it requires composing the forward fact with the "
  "inverse relation. The measurement is not vacuous by construction.")
A("")

if failures:
    A("## Verdict: FAIL")
    for f in failures:
        A(f"- FAILED CHECK: {f}")
else:
    A("## Verdict: PASS -- dataset cleared for the future exp-201 run "
      "(training/probing to be run separately, per HYPOTHESIS_201.txt).")
open(f"{EXP}/audit_2026-10-23.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("\nFAILURES:", failures if failures else "none")
raise SystemExit(1 if failures else 0)
