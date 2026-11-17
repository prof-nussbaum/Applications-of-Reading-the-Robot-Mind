"""exp-201-exposure STEP: ANALYSIS vs locked bars.
Writes analysis_exposure.json, RESULTS_EXPOSURE.md, SME_EXPOSURE.md.
"""
import json, statistics as st
from datetime import datetime, timezone

import os
ROBOT_MIND = os.environ.get("ROBOT_MIND", "/home/ecpi-student-05/robot-mind")  # Jetson working tree (data lives outside the repo); override with $ROBOT_MIND

EXP = ROBOT_MIND + "/exp-201-exposure"
ds = json.load(open(f"{EXP}/dataset_exposure.json"))
rev_set = set(ds["rev_idx"])
V = json.load(open(f"{EXP}/verdicts_exposure.json"))["models"]
B = json.load(open(f"{EXP}/behavior_exposure.json"))["models"]
ARMS = ("expA", "expB", "expC", "expD"); SEEDS = (0, 1, 2)

def bands(recs):
    return {b: sum(r["verdict_hs11"] == b for r in recs)
            for b in ("STORED", "WEAK", "MISSING")}

A = {}
for arm in ARMS:
    A[arm] = {}
    for seed in SEEDS:
        k = f"{arm}_s{seed}"
        vr = V[k]["reversed"]; br = B[k]["reversed"]; bf = B[k]["forward"]
        m = {
            "rev_behav_8": sum(r["pass_8tok"] for r in br),
            "rev_behav_50": sum(r["pass_50tok"] for r in br),
            "fwd_behav_8": sum(r["pass_8tok"] for r in bf),
            "fwd_behav_50": sum(r["pass_50tok"] for r in bf),
            "bands_hs11_rev": bands(vr),
            "spec_best_frac": sum(r["spec_best_of_60"] for r in vr) / 60.0,
            "spec_best_n": sum(r["spec_best_of_60"] for r in vr),
            "spec_mean_beaten": st.mean(r["spec_frac_others_beaten"] for r in vr),
            "set_median_rank": st.median(r["set_median_rank_60"] for r in vr),
            "correct_median_rank": st.median(r["lens_rank_hs11"] for r in vr),
        }
        if arm == "expD":
            for name, sel in (("revtrained", lambda r: r["index"] in rev_set),
                              ("fwdonly", lambda r: r["index"] not in rev_set)):
                sub_v = [r for r in vr if sel(r)]; sub_b = [r for r in br if sel(r)]
                m[name] = {
                    "rev_behav_8": sum(r["pass_8tok"] for r in sub_b),
                    "n": len(sub_b),
                    "bands_hs11_rev": bands(sub_v),
                    "spec_best_frac": sum(r["spec_best_of_60"] for r in sub_v) / len(sub_v),
                    "spec_best_n": sum(r["spec_best_of_60"] for r in sub_v),
                    "spec_mean_beaten": st.mean(r["spec_frac_others_beaten"] for r in sub_v),
                }
        A[arm][f"s{seed}"] = m

# Pooled specificity expA/B/C
pool_spec = [A[a][f"s{s}"]["spec_best_n"] for a in ("expA", "expB", "expC") for s in SEEDS]
pool_frac = sum(pool_spec) / (9 * 60)
abc_le10 = sum(1 for a in ("expA", "expB", "expC") for s in SEEDS
               if A[a][f"s{s}"]["spec_best_frac"] <= 0.10)
d_rev_ge50 = sum(1 for s in SEEDS if A["expD"][f"s{s}"]["revtrained"]["spec_best_frac"] >= 0.50)
d_fwd_le15 = sum(1 for s in SEEDS if A["expD"][f"s{s}"]["fwdonly"]["spec_best_frac"] <= 0.15)

# INVALID check: forward behaviour vs exp-201 (A 60, B 55, C 29) +-15 points
expected_fwd = {"expA": 60, "expB": 55, "expC": 29}
invalid = []
for arm, expv in expected_fwd.items():
    for s in SEEDS:
        got = A[arm][f"s{s}"]["fwd_behav_8"]
        if abs(got - expv) > 15:
            invalid.append(f"{arm}_s{s}: forward {got}/60 vs expected ~{expv}/60")

if invalid:
    verdict = "INVALID — retrain departs from exp-201 forward pattern: " + "; ".join(invalid)
elif abc_le10 >= 8 and d_rev_ge50 >= 2 and d_fwd_le15 >= 2:
    verdict = "FAMILIARITY CONFIRMED"
elif pool_frac >= 0.25:
    verdict = "FAMILIARITY REFUTED"
else:
    verdict = "MIXED"

summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "verdict": verdict, "per_arm_seed": A,
           "abc_armseeds_spec_le10": abc_le10,
           "abc_pooled_spec_frac": pool_frac,
           "expD_revtrained_seeds_spec_ge50": d_rev_ge50,
           "expD_fwdonly_seeds_spec_le15": d_fwd_le15,
           "invalid_flags": invalid}
json.dump(summary, open(f"{EXP}/analysis_exposure.json", "w"), indent=1)

L = []
L.append("# exp-201 Token-Exposure Control — RESULTS")
L.append(f"Date: 2026-11-15 (analysis {summary['timestamp_utc']} UTC)")
L.append("")
L.append(f"## VERDICT: {verdict}")
L.append("")
L.append("Decision bars (locked): FAMILIARITY CONFIRMED if expA/B/C specificity "
         "≤10% in ≥8 of 9 arm-seeds AND expD reversed-trained specificity ≥50% in "
         "≥2 of 3 seeds AND expD forward-only specificity ≤15% in ≥2 of 3 seeds. "
         "REFUTED if expA/B/C pooled specificity ≥25%. INVALID if forward "
         "behaviour departs from exp-201 (A~60, B~55, C~29) by >15 in any arm.")
L.append(f"- expA/B/C arm-seeds with specificity ≤10%: {abc_le10}/9")
L.append(f"- expA/B/C pooled specificity: {pool_frac:.3f} (chance 0.017)")
L.append(f"- expD reversed-trained seeds with specificity ≥50%: {d_rev_ge50}/3")
L.append(f"- expD forward-only seeds with specificity ≤15%: {d_fwd_le15}/3")
L.append(f"- INVALID flags: {invalid if invalid else 'none'}")
L.append("")
L.append("## PREDICTED vs ACTUAL")
L.append("PREDICTED (referee, adopted): expA/B/C reversed behaviour ≤3/60 per seed; "
         "MISSING share collapses (pooled <50%); specificity ≤10%; godparent set "
         "median rank in WEAK zone for expA/B. expD reversed-trained: behaviour "
         "≥24/30, specificity ≥50%, STORED common; expD forward-only like expB.")
L.append("")
for arm in ARMS:
    L.append(f"## {arm}")
    L.append("| seed | rev behav 8tok | fwd behav 8tok | bands S/W/M (rev, hs11) | spec best | mean beaten | set med rank |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        m = A[arm][f"s{s}"]; b = m["bands_hs11_rev"]
        L.append(f"| s{s} | {m['rev_behav_8']}/60 | {m['fwd_behav_8']}/60 | "
                 f"{b['STORED']}/{b['WEAK']}/{b['MISSING']} | "
                 f"{m['spec_best_n']}/60 ({m['spec_best_frac']:.2f}) | "
                 f"{m['spec_mean_beaten']:.2f} | {m['set_median_rank']:.0f} |")
    if arm == "expD":
        L.append("")
        L.append("Arm D split (30 reversed-trained / 30 forward-only pairs):")
        L.append("| seed | subset | rev behav 8tok | bands S/W/M | spec best | mean beaten |")
        L.append("|---|---|---|---|---|---|")
        for s in SEEDS:
            for sub in ("revtrained", "fwdonly"):
                m = A[arm][f"s{s}"][sub]; b = m["bands_hs11_rev"]
                L.append(f"| s{s} | {sub} | {m['rev_behav_8']}/{m['n']} | "
                         f"{b['STORED']}/{b['WEAK']}/{b['MISSING']} | "
                         f"{m['spec_best_n']}/{m['n']} ({m['spec_best_frac']:.2f}) | "
                         f"{m['spec_mean_beaten']:.2f} |")
    L.append("")
open(f"{EXP}/RESULTS_EXPOSURE.md", "w").write("\n".join(L) + "\n")

# SME page: seed-0 models; one rev-trained D pair, one forward-only D pair, one expB item
def sme_item(armkey, idx):
    vr = V[armkey]["reversed"][idx]; br = B[armkey]["reversed"][idx]
    pairs = json.load(open(ROBOT_MIND + "/exp-201/dataset_201.json"))["pairs"]
    ranked = sorted(range(60), key=lambda j: vr["ranks60_hs11"][j])
    top = ", ".join(f"{pairs[j][0]} (rank {vr['ranks60_hs11'][j]})" for j in ranked[:5])
    return (f"### {armkey}, pair {idx}: {vr['godparent']} / {vr['godchild']}\n"
            f"- Probe (verbatim): \"{vr['probe']}\"\n"
            f"- Correct godparent: {vr['godparent']} — hs[11] lens rank "
            f"{vr['lens_rank_hs11']} (verdict {vr['verdict_hs11']}); rank among the 60 "
            f"godparent names: {vr['spec_correct_rank_among_60']}; best of 60: "
            f"{vr['spec_best_of_60']}; fraction of other names beaten: "
            f"{vr['spec_frac_others_beaten']:.2f}\n"
            f"- Top-5 godparent names by lens rank: {top}\n"
            f"- Model's generation (first 8 tokens): \"{br['gen_8tok']}\" — "
            f"pass: {br['pass_8tok']}\n")
S = ["# exp-201 Token-Exposure Control — SME items (verbatim)", ""]
rt = sorted(rev_set)[0]
fo = [i for i in range(60) if i not in rev_set][0]
S.append("## One reversed-trained pair (arm expD, seed 0)")
S.append(sme_item("expD_s0", rt))
S.append("## One forward-only pair (arm expD, seed 0)")
S.append(sme_item("expD_s0", fo))
S.append("## One expB item (seed 0)")
S.append(sme_item("expB_s0", 0))
open(f"{EXP}/SME_EXPOSURE.md", "w").write("\n".join(S) + "\n")
print("ANALYSIS_DONE verdict:", verdict, flush=True)
