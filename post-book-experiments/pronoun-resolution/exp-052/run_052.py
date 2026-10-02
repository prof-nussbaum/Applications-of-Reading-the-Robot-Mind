"""exp-052: positive control -- can RTRM cosine similarity SEE the referent
when the model is not confused?

Paul (2026-09-29): the spider/trophy pictures show the referent far away
(#19,456 etc.), but that only means something if the pictures CAN show
the referent when resolution is unambiguous. This experiment is the
positive control: items where a functioning model is guaranteed to get
it right, so any failure to view the referent is a failure of the
RTRM cosine-similarity method, not of the model.

Items (051 template: <sentence> + ' What does the word {pronoun} refer
to? Answer with one word:'):
  6 flipped pairs -- same frame, verb selects the referent unambiguously
    pair1a dog/cat barking   -> dog
    pair1b dog/cat meowing   -> cat
    pair2a man/boy strong    -> man
    pair2b man/boy light     -> boy
    pair3a car/tree speeding -> car
    pair3b car/tree roots    -> tree
  4 ceiling items -- exactly one noun; nothing to resolve
    ceil1 dog barked/excited -> dog
    ceil2 king ruled/wise -> king
    ceil3 car broke/repairs  -> car
    ceil4 baby cried/hungry  -> baby

PREDICTED: behavioral 10/10 (or near); RTRM L12 neighborhood at the
  pronoun ranks the true referent <= 500 of 30,099 on all 10 items,
  and clearly on the 4 ceiling items.
KILL (predeclared, Paul's framing -- the point is viewability, not the
  model's score): on the 4 ceiling items the referent ranks > 500 at
  the pronoun position -> RTRM cosine similarity cannot view the
  referent even with zero ambiguity; the spider/trophy pictures are
  uninformative and the method needs revision (autoencoder / linear
  recreation of the input -- Paul's suggested next step, held off).
Metric: per-item right/wrong (051 decision rule); referent rank at the
  sentence pronoun position and at the final token, same 30,099-state
  L12 pool as exp-050/051b.
"""
import json
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

V = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2"
D = V + "/exp-052"

tok = GPT2TokenizerFast.from_pretrained("gpt2")
dev = "cuda" if torch.cuda.is_available() else "cpu"
lm = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()

items = [
    # flipped pairs: verb selects the referent
    {"id": "pair1a", "sentence": "The dog chased the cat because it was barking.",
     "pronoun": "it", "correct": "dog", "wrong": "cat"},
    {"id": "pair1b", "sentence": "The dog chased the cat because it was meowing.",
     "pronoun": "it", "correct": "cat", "wrong": "dog"},
    {"id": "pair2a", "sentence": "The man lifted the boy because he was strong.",
     "pronoun": "he", "correct": "man", "wrong": "boy"},
    {"id": "pair2b", "sentence": "The man lifted the boy because he was light.",
     "pronoun": "he", "correct": "boy", "wrong": "man"},
    {"id": "pair3a", "sentence": "The car hit the tree because it was speeding.",
     "pronoun": "it", "correct": "car", "wrong": "tree"},
    {"id": "pair3b", "sentence": "The car hit the tree because it had deep roots.",
     "pronoun": "it", "correct": "tree", "wrong": "car"},
    # ceiling: exactly one noun
    {"id": "ceil1", "sentence": "The dog barked because it was excited.",
     "pronoun": "it", "correct": "dog", "wrong": "cat"},
    {"id": "ceil2", "sentence": "The king ruled for years because he was wise.",
     "pronoun": "he", "correct": "king", "wrong": "farmer"},
    {"id": "ceil3", "sentence": "The car broke down because it needed repairs.",
     "pronoun": "it", "correct": "car", "wrong": "truck"},
    {"id": "ceil4", "sentence": "The baby cried because it was hungry.",
     "pronoun": "it", "correct": "baby", "wrong": "child"},
]

def qa_prompt(it):
    return (it["sentence"] + " What does the word %s refer to? "
            "Answer with one word:" % it["pronoun"].lower())

for it in items:
    for w in (it["correct"], it["wrong"]):
        n = len(tok(" " + w, add_special_tokens=False)["input_ids"])
        assert n == 1, (it["id"], w, n)

def score(it):
    p = qa_prompt(it)
    ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
    with torch.no_grad():
        lp = torch.log_softmax(lm(ids).logits[0, -1], dim=-1)
    tc = tok(" " + it["correct"], add_special_tokens=False)["input_ids"][0]
    tw = tok(" " + it["wrong"], add_special_tokens=False)["input_ids"][0]
    lpc, lpw = float(lp[tc]), float(lp[tw])
    pick = it["correct"] if lpc > lpw else it["wrong"]
    return {"id": it["id"], "prompt": p,
            "lp_correct": round(lpc, 2), "lp_wrong": round(lpw, 2),
            "pick": pick, "right": pick == it["correct"]}

rows = []
with torch.no_grad():
    for it in items:
        r = score(it)
        rows.append(r)
        print(r["id"], r["lp_correct"], r["lp_wrong"], r["pick"],
              "RIGHT" if r["right"] else "WRONG", flush=True)

n = len(rows)
k = sum(r["right"] for r in rows)
kceil = sum(r["right"] for r in rows if r["id"].startswith("ceil"))
kpair = sum(r["right"] for r in rows if r["id"].startswith("pair"))
print("total %d/%d  pairs %d/6  ceiling %d/4" % (k, n, kpair, kceil),
      flush=True)
json.dump({"rows": rows,
           "summary": {"n": n, "k": k, "pairs": kpair, "ceiling": kceil}},
          open(D + "/results_052.json", "w"), indent=1)
