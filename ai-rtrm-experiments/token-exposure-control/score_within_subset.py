#!/usr/bin/env python3
"""Within-subset specificity for arm D of the token-exposure retrain
(referee request R-003 section 1c, 2026-11-16).

The published specificity ranks the correct godparent name against all
59 others. R-003 1c asks: for arm D's reversed probes, rank the correct
name only against the other 29 names of its OWN subset -- the 30
reverse-trained pairs vs the 30 forward-only pairs (arm D trained the
reversed bridge for 30 of the 60 pairs only, so the two subsets differ
in exposure; ranking across subsets mixes that lift in).

Records-only: verdicts_exposure.json carries, per reversed item, the
full hs[11] lens ranks of all 60 godparent first-name tokens
(ranks60_hs11, pair-index order; see score_verdicts.py lines 62-80).
No model is loaded or run.

VALIDATION GATE (runs first; any failure aborts before any
within-subset number is produced):
  1. Per item: ranks60_hs11[index] == lens_rank_hs11 ==
     spec_correct_rank_among_60, and the stored spec_best_of_60 /
     spec_frac_others_beaten recompute exactly from ranks60_hs11.
  2. The recomputed all-59 aggregates reproduce RESULTS_EXPOSURE.md
     (arm D table): rev-trained spec best 15/30, 16/30, 12/30 and
     mean beaten 0.94 for s0/s1/s2; forward-only spec best 0/30 and
     mean beaten 0.35 for s0/s1/s2.

Output: within_subset_specificity.json beside this script.
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Arm D subsets: reversed bridges trained for the first 30 pairs of
# random.Random(201).shuffle(range(60)) -- same construction as
# build_audit.py / dataset_exposure.json.
_r = list(range(60)); random.Random(201).shuffle(_r)
REV = sorted(_r[:30])
FWD = sorted(_r[30:])
SUBSETS = {"revtrained": REV, "fwdonly": FWD}

# Published arm D figures (RESULTS_EXPOSURE.md), the validation target.
PUBLISHED = {  # seed -> subset -> (spec best count of 30, mean beaten rounded to 2dp)
    "expD_s0": {"revtrained": (15, 0.94), "fwdonly": (0, 0.35)},
    "expD_s1": {"revtrained": (16, 0.94), "fwdonly": (0, 0.35)},
    "expD_s2": {"revtrained": (12, 0.94), "fwdonly": (0, 0.35)},
}


def fail(msg):
    print("VALIDATION FAILED — STOP:", msg)
    sys.exit(1)


def main():
    d = json.load(open(os.path.join(HERE, "verdicts_exposure.json")))
    models = d["models"]

    # ---- validation + recompute all-59 from ranks60 -------------------
    per_item = {}  # (seed, index) -> dict of recomputed quantities
    for seed in ("expD_s0", "expD_s1", "expD_s2"):
        recs = models[seed]["reversed"]
        if len(recs) != 60:
            fail(f"{seed}: expected 60 reversed items, got {len(recs)}")
        seen = set()
        for it in recs:
            i = it["index"]
            seen.add(i)
            r60 = it["ranks60_hs11"]
            if len(r60) != 60:
                fail(f"{seed} item {i}: ranks60_hs11 has {len(r60)} entries")
            rc = r60[i]
            if rc != it["lens_rank_hs11"] or rc != it["spec_correct_rank_among_60"]:
                fail(f"{seed} item {i}: ranks60[index]={rc} != stored ranks "
                     f"({it['lens_rank_hs11']}, {it['spec_correct_rank_among_60']})")
            best59 = rc == min(r60)
            frac59 = sum(1 for j, r in enumerate(r60) if j != i and rc < r) / 59.0
            if best59 != it["spec_best_of_60"]:
                fail(f"{seed} item {i}: recomputed spec_best {best59} != stored {it['spec_best_of_60']}")
            if abs(frac59 - it["spec_frac_others_beaten"]) > 1e-12:
                fail(f"{seed} item {i}: recomputed frac beaten {frac59} != stored "
                     f"{it['spec_frac_others_beaten']}")
            per_item[(seed, i)] = {"r60": r60, "rc": rc, "best59": best59, "frac59": frac59}
        if seen != set(range(60)):
            fail(f"{seed}: item indices are not exactly 0..59")

    for seed, subs in PUBLISHED.items():
        for sub, (pub_best, pub_mean) in subs.items():
            idxs = SUBSETS[sub]
            best = sum(per_item[(seed, i)]["best59"] for i in idxs)
            mean = sum(per_item[(seed, i)]["frac59"] for i in idxs) / len(idxs)
            if best != pub_best:
                fail(f"{seed} {sub}: recomputed all-59 spec best {best}/30 != published {pub_best}/30")
            if round(mean, 2) != pub_mean:
                fail(f"{seed} {sub}: recomputed all-59 mean beaten {mean:.4f} "
                     f"rounds to {round(mean, 2)} != published {pub_mean}")
    print("VALIDATION PASSED: per-item spec fields recompute exactly from "
          "ranks60_hs11, and the all-59 aggregates reproduce RESULTS_EXPOSURE.md "
          "(rev-trained best 15/16/12 of 30, mean beaten 0.94; fwd-only best 0, "
          "mean beaten 0.35, per seed).")

    # ---- within-subset specificity ------------------------------------
    results = {"note": "Within-subset specificity for arm D reversed probes at hs[11] "
                       "(R-003 section 1c). Correct name ranked against the other 29 "
                       "names of its own subset only. Computed from ranks60_hs11 in "
                       "verdicts_exposure.json; no model runs.",
               "subsets": {"revtrained_indices": REV, "fwdonly_indices": FWD},
               "validation": "passed (see script docstring)",
               "per_seed": {}, "pooled": {}}
    pooled_items = {"revtrained": [], "fwdonly": []}
    for seed in ("expD_s0", "expD_s1", "expD_s2"):
        results["per_seed"][seed] = {}
        for sub, idxs in SUBSETS.items():
            rows = []
            for i in idxs:
                r60, rc = per_item[(seed, i)]["r60"], per_item[(seed, i)]["rc"]
                others = [r60[j] for j in idxs if j != i]
                rows.append({
                    "index": i,
                    "best_within_subset": rc < min(others),
                    "frac_same_subset_beaten": sum(1 for r in others if rc < r) / 29.0,
                    "best_of_60": per_item[(seed, i)]["best59"],
                    "frac_all59_beaten": per_item[(seed, i)]["frac59"],
                })
            pooled_items[sub].extend(rows)
            results["per_seed"][seed][sub] = {
                "n": len(rows),
                "best_within_subset_count": sum(r["best_within_subset"] for r in rows),
                "best_within_subset_share": sum(r["best_within_subset"] for r in rows) / len(rows),
                "mean_frac_same_subset_beaten": sum(r["frac_same_subset_beaten"] for r in rows) / len(rows),
                "mean_frac_all59_beaten": sum(r["frac_all59_beaten"] for r in rows) / len(rows),
                "items": rows,
            }
    for sub, rows in pooled_items.items():
        results["pooled"][sub] = {
            "n": len(rows),
            "best_within_subset_count": sum(r["best_within_subset"] for r in rows),
            "best_within_subset_share": sum(r["best_within_subset"] for r in rows) / len(rows),
            "mean_frac_same_subset_beaten": sum(r["frac_same_subset_beaten"] for r in rows) / len(rows),
            "mean_frac_all59_beaten": sum(r["frac_all59_beaten"] for r in rows) / len(rows),
        }

    out = os.path.join(HERE, "within_subset_specificity.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out}")
    for seed in ("expD_s0", "expD_s1", "expD_s2"):
        for sub in ("revtrained", "fwdonly"):
            r = results["per_seed"][seed][sub]
            print(f"{seed} {sub:11s} best-within {r['best_within_subset_count']:2d}/30 "
                  f"= {r['best_within_subset_share']:.3f}   mean beaten within "
                  f"{r['mean_frac_same_subset_beaten']:.3f}   (all-59: {r['mean_frac_all59_beaten']:.3f})")
    for sub in ("revtrained", "fwdonly"):
        r = results["pooled"][sub]
        print(f"POOLED  {sub:11s} best-within {r['best_within_subset_count']:2d}/90 "
              f"= {r['best_within_subset_share']:.3f}   mean beaten within "
              f"{r['mean_frac_same_subset_beaten']:.3f}   (all-59: {r['mean_frac_all59_beaten']:.3f})")


if __name__ == "__main__":
    main()
