"""exp-201 tabulation: join published-probe verdicts + behavioral results
per item; compute the primary pooled conditional rates (arms 201b+201c)
and apply the decision bar locked in HYPOTHESIS_201.txt.
Writes RESULTS_201.txt and tabulation_201.json.
"""
import json

EXP = "/home/ecpi-student-05/robot-mind/exp-201"
ARMS = ["mix_201a", "mix_201b", "mix_201c"]
PRIMARY = ["mix_201b", "mix_201c"]

items = json.load(open(f"{EXP}/dataset_201.json"))["probe_items"]

joined = {}   # arm -> list of per-item dicts
for arm in ARMS:
    probe = json.load(open(f"{EXP}/probe_{arm}_results.json"))
    beh = {r["index"]: r for r in json.load(open(f"{EXP}/behavior_{arm}.json"))}
    assert len(probe) == 60, (arm, len(probe))
    rows = []
    for i, pr in enumerate(probe):
        it = items[i]
        assert pr["prompt"] == it["reverse_prompt"], (arm, i, pr["prompt"])
        assert pr["answer"] == " " + it["name"], (arm, i)
        b = beh[i]
        rows.append({
            "index": i, "godparent": it["godparent"], "godchild": it["godchild"],
            "rank": pr["rank"], "verdict": pr["verdict"],
            "pass_8tok": b["pass_8tok"], "pass_50tok": b["pass_50tok"],
            "gen_8tok": b["gen_8tok"], "gen_50tok": b["gen_50tok"],
        })
    joined[arm] = rows
json.dump(joined, open(f"{EXP}/tabulation_201.json", "w"), indent=1)

def cond(rows, verdict, want_pass):
    sel = [r for r in rows if r["verdict"] == verdict]
    if not sel:
        return 0, 0, None
    hits = sum(1 for r in sel if r["pass_8tok"] == want_pass)
    return len(sel), hits, hits / len(sel)

def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"

lines = []
def w(s=""):
    lines.append(s)

# ---- primary pooled rates ----
prows = [r for arm in PRIMARY for r in joined[arm]]
mn, mf, mrate = cond(prows, "MISSING", False)   # fail = not pass
sn, sp, srate = cond(prows, "STORED", True)
m_inc = mn < 10
s_inc = sn < 10
if m_inc or s_inc:
    verdict = "INCONCLUSIVE"
elif mrate >= 0.80 and srate >= 0.80:
    verdict = "SUPPORTED"
else:
    verdict = "REFUTED"

w("exp-201 RESULTS -- does the published RTRM logit-lens verdict flag")
w("reversal failures on NEW data (godparent/godchild domain)?")
w(f"VERDICT: {verdict}  (per the decision bar locked in HYPOTHESIS_201.txt)")
w("")
w("PREDICTED (locked before any run):")
w("  Items verdicted MISSING fail the behavioral test at a rate >= 80%.")
w("  Items verdicted STORED pass the behavioral test at a rate >= 80%.")
w("  (Pooled over arms 201b + 201c, 120 items; WEAK = no claim;")
w("   each class needs >= 10 items or that half is INCONCLUSIVE.)")
w("")
w("ACTUAL (primary metric):")
w(f"  P(FAIL | MISSING) = {mf}/{mn} = {pct(mrate)}"
  + ("  [class < 10 items -> INCONCLUSIVE half]" if m_inc else ""))
w(f"  P(PASS | STORED)  = {sp}/{sn} = {pct(srate)}"
  + ("  [class < 10 items -> INCONCLUSIVE half]" if s_inc else ""))
w("")
w("INSTRUMENT: published rtrm_probe.py, hs[11] convention, bands")
w("STORED <= 10 / WEAK 11-100 / MISSING > 100. Behavioral: greedy")
w("8-token generation contains the full expected name (50-token")
w("variant recorded in the per-arm files). Verdict files were saved")
w("and timestamped before any behavioral scoring ran.")
w("")
w("PER-ARM DETAIL")
for arm in ARMS:
    rows = joined[arm]
    tall = {v: sum(1 for r in rows if r["verdict"] == v)
            for v in ("STORED", "WEAK", "MISSING")}
    npass = sum(r["pass_8tok"] for r in rows)
    npass50 = sum(r["pass_50tok"] for r in rows)
    an, af, ar = cond(rows, "MISSING", False)
    bn, bp, br = cond(rows, "STORED", True)
    w(f"  {arm}: verdicts STORED {tall['STORED']} / WEAK {tall['WEAK']} / "
      f"MISSING {tall['MISSING']}; behavioral pass 8tok {npass}/60, "
      f"50tok {npass50}/60")
    w(f"      P(FAIL|MISSING) = {af}/{an} = {pct(ar)};  "
      f"P(PASS|STORED) = {bp}/{bn} = {pct(br)}")
w("")
w("DISAGREEMENTS (verdict vs behavior, all arms)")
dis = []
for arm in ARMS:
    for r in joined[arm]:
        if r["verdict"] == "STORED" and not r["pass_8tok"]:
            dis.append(f"  {arm} item {r['index']}: STORED (rank {r['rank']}) "
                       f"but FAILED -- {r['godparent']} -> {r['godchild']}")
        if r["verdict"] == "MISSING" and r["pass_8tok"]:
            dis.append(f"  {arm} item {r['index']}: MISSING (rank {r['rank']}) "
                       f"but PASSED -- {r['godparent']} -> {r['godchild']}")
if dis:
    lines.extend(dis)
else:
    w("  none: no STORED item failed and no MISSING item passed.")
w("")
w("SECONDARY EXPECTATIONS (not part of the primary metric)")
a = joined["mix_201a"]
c = joined["mix_201c"]
w(f"  201a (bridge) like 200q q3 (60/60, mostly STORED): actual pass "
  f"{sum(r['pass_8tok'] for r in a)}/60, STORED "
  f"{sum(1 for r in a if r['verdict'] == 'STORED')}/60.")
w(f"  201c (facts-x1) like 200q q5 (19/60, mostly MISSING): actual pass "
  f"{sum(r['pass_8tok'] for r in c)}/60, MISSING "
  f"{sum(1 for r in c if r['verdict'] == 'MISSING')}/60.")
w("")
w("Per-item data: tabulation_201.json; probe files probe_mix_201*_results.json;")
w("behavior files behavior_mix_201*.json (this folder).")

open(f"{EXP}/RESULTS_201.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
