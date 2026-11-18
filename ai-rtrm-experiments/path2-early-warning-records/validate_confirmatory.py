"""Path 2 confirmatory validation: completeness of results_confirmatory.json
against confirmatory_items.jsonl (and disjointness from pilot items), internal
consistency recomputation of every derived field, top-20 sidecar cross-check,
and the determinism comparison against the 5-item end-to-end re-run.
Writes validation_confirmatory.json. Read-only on all run outputs."""
import json, math

D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"

items = [json.loads(l) for l in open(D + "/confirmatory_items.jsonl")]
pilot = [json.loads(l) for l in open(D + "/pilot_items.jsonl")]
res = json.load(open(D + "/results_confirmatory.json"))
top20 = json.load(open(D + "/results_confirmatory_top20.json"))

problems = []

def key(s, i): return (s, i)

item_keys = [key(it["split"], it["item"]) for it in items]
res_keys = [key(r["split"], r["item"]) for r in res]
if len(res) != 617: problems.append("record count %d != 617" % len(res))
if len(set(res_keys)) != len(res_keys): problems.append("duplicate (split,item) keys in results")
if set(res_keys) != set(item_keys): problems.append("results keys != confirmatory item keys")
pilot_keys = set(key(p["split"], p["item"]) for p in pilot)
if set(res_keys) & pilot_keys: problems.append("PILOT OVERLAP: %d keys" % len(set(res_keys) & pilot_keys))

REQ = ["split", "item", "sentence", "prompt", "pronoun", "head1", "head2", "label",
       "competitor", "clue_class", "r_label", "r_head1", "r_head2", "r_comp",
       "rtrm_lean", "rtrm_flag", "secondary_verdict", "logit_head1", "logit_head2",
       "model_pick", "model_right", "margin_nats", "margin_flag_B1"]
item_by_key = {key(it["split"], it["item"]): it for it in items}

for r in res:
    tag = "%s-%s" % (r["split"], r["item"])
    for f in REQ:
        if f not in r: problems.append(tag + " missing field " + f)
    if problems and problems[-1].startswith(tag + " missing"): continue
    it = item_by_key[key(r["split"], r["item"])]
    if r["sentence"] != it["sentence"]: problems.append(tag + " sentence mismatch vs items file")
    exp_prompt = "%s What does the word %s refer to? Answer with one word:" % (it["sentence"], it["pronoun"])
    if r["prompt"] != exp_prompt: problems.append(tag + " prompt format mismatch")
    if r["label"] != it["label_head"] or r["head1"] != it["head1"] or r["head2"] != it["head2"]:
        problems.append(tag + " heads/label mismatch vs items file")
    if r["clue_class"] != it["clue_class"]: problems.append(tag + " clue class mismatch")
    if not (math.isfinite(r["logit_head1"]) and math.isfinite(r["logit_head2"]) and math.isfinite(r["margin_nats"])):
        problems.append(tag + " non-finite logit/margin")
    if r["model_pick"] not in (r["head1"], r["head2"]): problems.append(tag + " pick not a head")
    other = r["head2"] if r["model_pick"] == r["head1"] else r["head1"]
    lp = r["logit_head1"] if r["model_pick"] == r["head1"] else r["logit_head2"]
    lo = r["logit_head1"] if other == r["head1"] else r["logit_head2"]
    if abs(r["margin_nats"] - (lp - lo)) > 1e-5: problems.append(tag + " margin != logit_pick - logit_other")
    if r["margin_nats"] < -1e-9: problems.append(tag + " negative margin")
    if r["margin_flag_B1"] != (r["margin_nats"] < 0.35): problems.append(tag + " B1 flag mismatch")
    if r["model_right"] != (r["model_pick"] == r["label"]): problems.append(tag + " model_right mismatch")
    rL, rC = r["r_label"], r["r_comp"]
    exp_flag = (rL is None) or (rL > 500) or (rC is not None and rC < rL)
    if r["rtrm_flag"] != exp_flag: problems.append(tag + " RTRM flag recompute mismatch")
    exp_sec = "CORRECT" if (rL is not None and rL <= 100 and rC is not None and rC > rL) else "RISK"
    if r["secondary_verdict"] != exp_sec: problems.append(tag + " secondary verdict recompute mismatch")
    exp_rL = r["r_head1"] if r["label"] == r["head1"] else r["r_head2"]
    exp_rC = r["r_head2"] if r["label"] == r["head1"] else r["r_head1"]
    if rL != exp_rL or rC != exp_rC: problems.append(tag + " r_label/r_comp inconsistent with r_head1/r_head2")
    r1, r2 = r["r_head1"], r["r_head2"]
    exp_lean = r["head1"] if (r1 is not None and (r2 is None or r1 < r2)) else r["head2"]
    if r["rtrm_lean"] != exp_lean: problems.append(tag + " lean recompute mismatch")
    exp_comp = r["head2"] if r["label"] == r["head1"] else r["head1"]
    if r["competitor"] != exp_comp: problems.append(tag + " competitor mismatch")

# top-20 sidecar cross-check
t20 = {key(t["split"], t["item"]): t["top20"] for t in top20}
if len(top20) != 617: problems.append("top20 sidecar count %d != 617" % len(top20))
for r in res:
    lst = t20.get(key(r["split"], r["item"]))
    if lst is None or len(lst) != 20:
        problems.append("%s-%s top20 missing/wrong length" % (r["split"], r["item"])); continue
    for rank, word in ((r["r_label"], r["label"]), (r["r_comp"], r["competitor"]),
                       (r["r_head1"], r["head1"]), (r["r_head2"], r["head2"])):
        if rank is not None and rank <= 20 and lst[rank - 1] != word.lower():
            problems.append("%s-%s top20[%d]=%s != %s" % (r["split"], r["item"], rank - 1, lst[rank - 1], word.lower()))

# determinism: first-5 re-run vs full-run records
rerun = json.load(open(D + "/results_confirmatory_first5_rerun.json"))
rerun20 = json.load(open(D + "/results_confirmatory_first5_rerun_top20.json"))
full_by_key = {key(r["split"], r["item"]): r for r in res}
rank_mismatch, field_mismatch = [], []
max_margin_diff = 0.0; max_logit_diff = 0.0
for rr in rerun:
    fr = full_by_key[key(rr["split"], rr["item"])]
    for f in ("r_label", "r_head1", "r_head2", "r_comp"):
        if rr[f] != fr[f]: rank_mismatch.append("%s-%s %s %s vs %s" % (rr["split"], rr["item"], f, rr[f], fr[f]))
    for f in ("rtrm_flag", "secondary_verdict", "rtrm_lean", "model_pick", "model_right", "margin_flag_B1"):
        if rr[f] != fr[f]: field_mismatch.append("%s-%s %s" % (rr["split"], rr["item"], f))
    max_margin_diff = max(max_margin_diff, abs(rr["margin_nats"] - fr["margin_nats"]))
    max_logit_diff = max(max_logit_diff, abs(rr["logit_head1"] - fr["logit_head1"]), abs(rr["logit_head2"] - fr["logit_head2"]))
r20 = {key(t["split"], t["item"]): t["top20"] for t in rerun20}
top20_mismatch = [("%s-%s" % k) for k in r20 if r20[k] != t20[k]]

out = {"n_records": len(res),
       "completeness_problems": problems,
       "rerun_n": len(rerun),
       "rank_mismatches": rank_mismatch,
       "field_mismatches": field_mismatch,
       "top20_mismatches": top20_mismatch,
       "max_margin_diff": max_margin_diff,
       "max_logit_diff": max_logit_diff}
json.dump(out, open(D + "/validation_confirmatory.json", "w"), indent=1)
print(json.dumps(out, indent=1))
