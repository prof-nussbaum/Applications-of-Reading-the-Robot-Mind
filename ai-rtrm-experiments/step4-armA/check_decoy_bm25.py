"""Step-4 check 4: does BM25 prefer the decoy on the exact strings?

Question (R-020 s.2): the Arm-A decoy is built to mislead a word search --
it holds the probe's rarest word ("godchild") and never gives the answer.
If BM25 does NOT rank the decoy above the target, the decoy is too weak
and must be rewritten before anything is trained.

Method: the EXACT BM25 from build_controls_seed0.py (Okapi k1 = 1.5,
b = 0.75, idf = ln(1 + (N - df + 0.5) / (df + 0.5)), word tokens
lowercased, split on non-alphanumerics). Corpus = the step-4 shape:
1,756 Wikipedia sentences + 60 target sentences ("G is the godparent
of C.") + 60 decoy sentences ("D met the godchild of G last week.")
+ the 60 E1b-fresh fillers. NO reversed sentences -- Arm A does not
train them. Names are the E1b-fresh pairs' real names (the scored run
will use fresh ones; the preference is driven by token overlap and
IDF, not by which names). Decoy people D are drawn from a candidate
list filtered against every name in the fresh dataset.

Query per pair: the probe "G's godchild is".
Report per pair: rank of decoy vs rank of target; count of
decoy-preferred / tied / target-preferred.
"""
import json, math, re
from collections import Counter

EXP = "/home/ecpi-student-05/robot-mind/exp-201-exposure-fresh"
P2 = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"

dataset = json.load(open(f"{EXP}/dataset_fresh.json"))
pairs = dataset["pairs"]
rows = [json.loads(l) for l in open(P2 + "/pool_build_selection.jsonl")]
wiki_texts = [r["sentence"] for r in rows]

# Decoy people: candidate first names, filtered against every name in
# the fresh dataset (godparents, godchildren, and any name in fillers).
used = set()
for gp, gc in pairs:
    used.add(gp.lower())
    used.add(gc.lower())
for s in dataset["fillers"]:
    for w in re.findall(r"[A-Z][a-z]+", s):
        used.add(w.lower())
candidates = ["Quentin", "Xenia", "Yusuf", "Zelda", "Omar", "Priya",
              "Rosa", "Felix", "Nadia", "Hugo", "Iris", "Karl",
              "Lena", "Milo", "Nina", "Oscar", "Tara", "Umar",
              "Vera", "Wade", "Yara", "Zane", "Amos", "Beth",
              "Cora", "Drew", "Elsa", "Finn", "Gwen", "Hank",
              "Ivy", "Jude", "Kara", "Liam", "Maya", "Nate",
              "Opal", "Paul", "Ruth", "Seth", "Tess", "Vince",
              "Wren", "Yuki", "Zola", "Abe", "Cara", "Duke",
              "Emma", "Fred", "Gail", "Hana", "Ivan", "June",
              "Kyle", "Lola", "Max", "Nora", "Owen", "Pete",
              "Quinn", "Ruby", "Aldo", "Bree", "Cleo", "Dana",
              "Ezra", "Farah", "Gus", "Holly"]
decoys = [c for c in candidates if c.lower() not in used]
assert len(decoys) >= 60, f"only {len(decoys)} fresh decoy names"
decoys = decoys[:60]

docs = [(t, "wiki", None) for t in wiki_texts]
for i, (gp, gc) in enumerate(pairs):
    docs.append((f"{gp} is the godparent of {gc}.", "target", i))
    docs.append((f"{decoys[i]} met the godchild of {gp} last week.",
                 "decoy", i))
for s in dataset["fillers"]:
    hits = [i for i, (gp, gc) in enumerate(pairs) if gp in s]
    assert len(hits) == 1
    docs.append((s, "filler", hits[0]))

def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())

doc_tokens = [tokens(t) for t, _, _ in docs]
N = len(docs)
df = Counter()
for dt in doc_tokens:
    for w in set(dt):
        df[w] += 1
idf = {w: math.log(1 + (N - c + 0.5) / (c + 0.5)) for w, c in df.items()}
avgdl = sum(len(dt) for dt in doc_tokens) / N
k1, b = 1.5, 0.75
doc_tf = [Counter(dt) for dt in doc_tokens]

def bm25_scores(query):
    q = tokens(query)
    scores = []
    for d in range(N):
        dl = len(doc_tokens[d])
        s = 0.0
        for w in q:
            f = doc_tf[d].get(w, 0)
            if f:
                s += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores

print(f"corpus N={N} (wiki {len(wiki_texts)}, targets 60, decoys 60, "
      f"fillers {len(dataset['fillers'])}), avgdl={avgdl:.2f}")
print(f"idf: godchild={idf.get('godchild', 0):.3f} "
      f"is={idf.get('is', 0):.3f} s={idf.get('s', 0):.3f} "
      f"godparent={idf.get('godparent', 0):.3f}")
print(f"df: godchild={df.get('godchild', 0)} is={df.get('is', 0)}")

target_doc = {}
decoy_doc = {}
for d, (t, kind, i) in enumerate(docs):
    if kind == "target":
        target_doc[i] = d
    elif kind == "decoy":
        decoy_doc[i] = d

decoy_pref = tie = target_pref = 0
decoy_ranks, target_ranks = [], []
example = None
for i, (gp, gc) in enumerate(pairs):
    probe = f"{gp}'s godchild is"
    scores = bm25_scores(probe)
    order = sorted(range(N), key=lambda d: -scores[d])
    pos = {d: r + 1 for r, d in enumerate(order)}
    rd, rt = pos[decoy_doc[i]], pos[target_doc[i]]
    decoy_ranks.append(rd)
    target_ranks.append(rt)
    if rd < rt:
        decoy_pref += 1
    elif rd == rt:
        tie += 1
    else:
        target_pref += 1
    if example is None and rd < rt:
        example = (i, gp, gc, probe, scores[decoy_doc[i]],
                   scores[target_doc[i]], order[0], docs[order[0]][0])

print(f"pairs: {len(pairs)}")
print(f"decoy outranks target: {decoy_pref}")
print(f"tied: {tie}")
print(f"target outranks decoy: {target_pref}")
print(f"mean decoy rank: {sum(decoy_ranks)/len(decoy_ranks):.1f}, "
      f"mean target rank: {sum(target_ranks)/len(target_ranks):.1f}")
print(f"decoy at rank 1: {sum(1 for r in decoy_ranks if r == 1)}, "
      f"target at rank 1: {sum(1 for r in target_ranks if r == 1)}")
if example:
    i, gp, gc, probe, sd, st, top, toptext = example
    print(f"example pair {i} ({gp}/{gc}), probe '{probe}': "
          f"decoy score {sd:.3f}, target score {st:.3f}; "
          f"top-1 doc is '{toptext}'")
print("VERDICT:", "DECOY PREFERRED" if decoy_pref == len(pairs)
      else ("DECOY MOSTLY PREFERRED" if decoy_pref > len(pairs) / 2
            else "DECOY NOT PREFERRED - REWRITE NEEDED"))
