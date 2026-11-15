"""Path 2 CONFIRMATORY analysis — exactly the predeclared quantities of
path2_early_warning_PROTOCOL_2026-11-13.md (as amended), nothing else as
headline. Reads results_confirmatory.json (validated complete) and the
top-20 sidecar; writes analysis_confirmatory.json, caught_miss_lists.json,
RESULTS_confirmatory.md. Definitions:
  ERR = pick != label; B1 flag = margin < 0.35 nats.
  MISS = ERR and NOT B1-flagged. RTRM catch = share of MISS that are RTRM-flagged.
  Combined flag = RTRM OR B1. Sensitivity = share of ERR flagged;
  false-alarm rate = share of correct items flagged.
Locked verdict: INCONCLUSIVE if MISS < 30. WIN requires ALL of
  (i) RTRM catch of MISS >= 30%; (ii) combined sens - B1 sens >= 10 pp;
  (iii) combined FAR - B1 FAR <= 20 pp; (iv) one-sided Fisher exact
  (RTRM flag x ERR among B1-unflagged items), p < 0.05.
  REFUTED if MISS >= 30 and (i) or (ii) fails.
Secondary regardless: exp-056 H1 2x2 (flag x error, all items) + Fisher p.
Appendix only: AUC of margin as an error score; (sens+spec)/2 for the RTRM flag.
"""
import json
from scipy.stats import fisher_exact

D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/path2-early-warning"
res = json.load(open(D + "/results_confirmatory.json"))
top20 = { (t["split"], t["item"]): t["top20"] for t in json.load(open(D + "/results_confirmatory_top20.json")) }

def block(recs):
    n = len(recs)
    err = [r for r in recs if not r["model_right"]]
    cor = [r for r in recs if r["model_right"]]
    ne, nc = len(err), len(cor)
    def frac(flagged, tot): return (len(flagged) / tot) if tot else None
    b1_e = [r for r in err if r["margin_flag_B1"]]; b1_c = [r for r in cor if r["margin_flag_B1"]]
    rt_e = [r for r in err if r["rtrm_flag"]];     rt_c = [r for r in cor if r["rtrm_flag"]]
    cb_e = [r for r in err if r["rtrm_flag"] or r["margin_flag_B1"]]
    cb_c = [r for r in cor if r["rtrm_flag"] or r["margin_flag_B1"]]
    miss = [r for r in err if not r["margin_flag_B1"]]
    miss_caught = [r for r in miss if r["rtrm_flag"]]
    nm = len(miss)
    catch = (len(miss_caught) / nm) if nm else None
    sens_b1, sens_rt, sens_cb = frac(b1_e, ne), frac(rt_e, ne), frac(cb_e, ne)
    far_b1, far_rt, far_cb = frac(b1_c, nc), frac(rt_c, nc), frac(cb_c, nc)
    gain_sens = (sens_cb - sens_b1) if ne else None
    gain_far = (far_cb - far_b1) if nc else None
    # Fisher (iv): among B1-unflagged items, RTRM flag x ERR, one-sided greater
    unb = [r for r in recs if not r["margin_flag_B1"]]
    a = len([r for r in unb if r["rtrm_flag"] and not r["model_right"]])
    b = len([r for r in unb if r["rtrm_flag"] and r["model_right"]])
    c = len([r for r in unb if not r["rtrm_flag"] and not r["model_right"]])
    d = len([r for r in unb if not r["rtrm_flag"] and r["model_right"]])
    p_iv = float(fisher_exact([[a, b], [c, d]], alternative="greater")[1]) if (a+b) and (c+d) else None
    # Secondary H1: flag x error on all items
    ha = len(rt_e); hb = len(rt_c)
    hc = ne - ha;   hd = nc - hb
    p_h1 = float(fisher_exact([[ha, hb], [hc, hd]], alternative="greater")[1]) if (ha+hb) and (hc+hd) else None
    # Appendix: AUC of margin as error score (score = -margin; Mann-Whitney with ties)
    def auc_margin(rs):
        e = sorted(r["margin_nats"] for r in rs if not r["model_right"])
        cc = [r["margin_nats"] for r in rs if r["model_right"]]
        if not e or not cc: return None
        wins = 0.0
        import bisect
        for m in cc:
            lo = bisect.bisect_left(e, m)       # errors with margin < m  -> score higher
            eq = bisect.bisect_right(e, m) - lo
            wins += lo + 0.5 * eq
        return wins / (len(e) * len(cc))
    auc = auc_margin(recs)
    bal = ((sens_rt + (1 - far_rt)) / 2) if (ne and nc) else None
    bars = None; verdict = None
    if nm < 30:
        verdict = "INCONCLUSIVE"
    else:
        bars = {"i_catch_ge_30pct": bool(catch >= 0.30),
                "ii_sens_gain_ge_10pp": bool(gain_sens >= 0.10),
                "iii_far_gain_le_20pp": bool(gain_far <= 0.20),
                "iv_fisher_p_lt_0.05": bool(p_iv is not None and p_iv < 0.05)}
        if all(bars.values()): verdict = "WIN"
        elif not bars["i_catch_ge_30pct"] or not bars["ii_sens_gain_ge_10pp"]: verdict = "REFUTED"
        else:
            failed = [k.split("_")[0] for k, v in bars.items() if not v]
            verdict = "NOT WIN (failed bars: %s; the locked REFUTED definition - (i) or (ii) failing - is not met)" % ", ".join(failed)
    return {
        "n": n, "n_errors": ne, "n_correct": nc,
        "error_rate": (ne / n) if n else None,
        "b1_flagged_errors": len(b1_e), "b1_flagged_correct": len(b1_c),
        "rtrm_flagged_errors": len(rt_e), "rtrm_flagged_correct": len(rt_c),
        "combined_flagged_errors": len(cb_e), "combined_flagged_correct": len(cb_c),
        "sens_b1": sens_b1, "sens_rtrm": sens_rt, "sens_combined": sens_cb,
        "far_b1": far_b1, "far_rtrm": far_rt, "far_combined": far_cb,
        "sens_gain_pp": (gain_sens * 100) if gain_sens is not None else None,
        "far_gain_pp": (gain_far * 100) if gain_far is not None else None,
        "n_miss": nm, "miss_caught": len(miss_caught), "rtrm_catch_of_miss": catch,
        "fisher_iv_table_B1unflagged": [[a, b], [c, d]], "fisher_iv_p_onesided": p_iv,
        "h1_table_flag_x_error_all": [[ha, hb], [hc, hd]], "h1_fisher_p_onesided": p_h1,
        "appendix_auc_margin_error_score": auc,
        "appendix_rtrm_balanced_accuracy": bal,
        "bars": bars, "verdict": verdict}

pooled = block(res)
by_split = {s: block([r for r in res if r["split"] == s]) for s in ("test", "train")}
by_class = {c: block([r for r in res if r["clue_class"] == c]) for c in ("A", "B", "U")}

miss_all = [r for r in res if (not r["model_right"]) and (not r["margin_flag_B1"])]
miss_all_sorted = sorted(miss_all, key=lambda r: (r["split"], r["item"]))
caught = [r for r in miss_all_sorted if r["rtrm_flag"]]

caught_lists = [{"split": r["split"], "item": r["item"], "sentence": r["sentence"],
                 "prompt": r["prompt"], "label": r["label"], "competitor": r["competitor"],
                 "model_pick": r["model_pick"], "margin_nats": r["margin_nats"],
                 "r_label": r["r_label"], "r_comp": r["r_comp"], "rtrm_lean": r["rtrm_lean"],
                 "top20": top20[(r["split"], r["item"])]} for r in caught]
json.dump(caught_lists, open(D + "/caught_miss_lists.json", "w"), indent=1)

analysis = {"pooled": pooled, "by_split": by_split, "by_clue_class": by_class,
            "miss_items": [{"split": r["split"], "item": r["item"], "clue_class": r["clue_class"],
                            "label": r["label"], "competitor": r["competitor"], "model_pick": r["model_pick"],
                            "margin_nats": r["margin_nats"], "r_label": r["r_label"], "r_comp": r["r_comp"],
                            "rtrm_lean": r["rtrm_lean"], "rtrm_flag": r["rtrm_flag"]} for r in miss_all_sorted]}
json.dump(analysis, open(D + "/analysis_confirmatory.json", "w"), indent=1)

# ---------- RESULTS_confirmatory.md ----------
def pct(x): return "n/a" if x is None else "%.1f%%" % (100 * x)
def pp(x): return "n/a" if x is None else "%+.1f pp" % x
def pval(x): return "n/a" if x is None else ("%.3g" % x)
def barline(name, ok): return "- %s: **%s**" % (name, "PASS" if ok else "FAIL")

L = []
L.append("# Path 2 — CONFIRMATORY RESULTS (2026-11-14)")
L.append("")
L.append("Protocol: `path2_early_warning_PROTOCOL_2026-11-13.md` + Amendments 1–2 + "
         "pool-count decision + eligibility ruling (all 2026-11-13/14, locked before scoring). "
         "Confirmatory set: 617 items (test 198, train 419; clue A 86, B 529, U 2). "
         "The 60 pilot items were NOT scored here. Method exactly as the pilot: GPT-2 small, "
         "formatted prompt \"<sentence> What does the word <pronoun> refer to? Answer with one word:\", "
         "L12 pronoun state, PCA-64 residual (seed 0) with pronoun-mean subtraction, "
         "label/competitor best-occurrence ranks vs pool_path2.pt (47,499 states), "
         "exp-056 RTRM flag, forced-choice margin in nats signed toward the pick.")
L.append("")
L.append("## VERDICT: **%s**" % pooled["verdict"])
L.append("")
L.append("## PREDICTED vs ACTUAL (pooled)")
L.append("")
L.append("PREDICTED (locked in the protocol, before data):")
L.append("- GPT-2 errs on roughly 25–45% of eligible DPR items.")
L.append("- Of the errors the margin baseline does NOT flag (MISS), the RTRM flag catches at least 30%.")
L.append("- Combined (RTRM OR B1) sensitivity beats B1-only by at least 10 percentage points, "
         "with false-alarm rate on correct items no more than 20 points above B1-only.")
L.append("")
L.append("ACTUAL:")
L.append("- Model errors: %d of %d (error rate %s). Errors the margin baseline missed (MISS): %d."
         % (pooled["n_errors"], pooled["n"], pct(pooled["error_rate"]), pooled["n_miss"]))
L.append("- RTRM catch of MISS: %d of %d = %s." % (pooled["miss_caught"], pooled["n_miss"], pct(pooled["rtrm_catch_of_miss"])))
L.append("- Sensitivity: B1-only %s, RTRM-only %s, combined %s (gain %s)."
         % (pct(pooled["sens_b1"]), pct(pooled["sens_rtrm"]), pct(pooled["sens_combined"]), pp(pooled["sens_gain_pp"])))
L.append("- False-alarm rate on correct items: B1-only %s, RTRM-only %s, combined %s (gain %s)."
         % (pct(pooled["far_b1"]), pct(pooled["far_rtrm"]), pct(pooled["far_combined"]), pp(pooled["far_gain_pp"])))
L.append("")
L.append("## The four locked bars (pooled)")
L.append("")
if pooled["bars"]:
    L.append(barline("(i) RTRM catch of MISS >= 30%% — actual %s" % pct(pooled["rtrm_catch_of_miss"]), pooled["bars"]["i_catch_ge_30pct"]))
    L.append(barline("(ii) combined sensitivity − B1 sensitivity >= 10 pp — actual %s" % pp(pooled["sens_gain_pp"]), pooled["bars"]["ii_sens_gain_ge_10pp"]))
    L.append(barline("(iii) combined false-alarm − B1 false-alarm <= 20 pp — actual %s" % pp(pooled["far_gain_pp"]), pooled["bars"]["iii_far_gain_le_20pp"]))
    L.append(barline("(iv) Fisher exact (flag × error, B1-unflagged items), one-sided p < 0.05 — actual p = %s" % pval(pooled["fisher_iv_p_onesided"]), pooled["bars"]["iv_fisher_p_lt_0.05"]))
else:
    L.append("Not evaluated: MISS = %d < 30, so the locked verdict is INCONCLUSIVE and no bar is claimed."
             % pooled["n_miss"])
L.append("")
L.append("## Tables")
L.append("")
def table(title, bl, claim):
    L.append("### %s (n=%d, errors=%d, MISS=%d)" % (title, bl["n"], bl["n_errors"], bl["n_miss"]))
    L.append("")
    L.append("- Sensitivity: B1 %s | RTRM %s | combined %s (gain %s)"
             % (pct(bl["sens_b1"]), pct(bl["sens_rtrm"]), pct(bl["sens_combined"]), pp(bl["sens_gain_pp"])))
    L.append("- False-alarm: B1 %s | RTRM %s | combined %s (gain %s)"
             % (pct(bl["far_b1"]), pct(bl["far_rtrm"]), pct(bl["far_combined"]), pp(bl["far_gain_pp"])))
    L.append("- RTRM catch of MISS: %s (%d/%d)" % (pct(bl["rtrm_catch_of_miss"]), bl["miss_caught"], bl["n_miss"]))
    L.append("- Fisher (iv) p (one-sided): %s — table (B1-unflagged) flagged×[err,correct] / unflagged×[err,correct] = %s"
             % (pval(bl["fisher_iv_p_onesided"]), bl["fisher_iv_table_B1unflagged"]))
    L.append("- Secondary exp-056 H1 (flag × error, all items): flagged×[err,correct] / unflagged×[err,correct] = %s, one-sided Fisher p = %s"
             % (bl["h1_table_flag_x_error_all"], pval(bl["h1_fisher_p_onesided"])))
    L.append("- Verdict under the locked logic: **%s**%s" % (bl["verdict"], claim))
    L.append("")
table("Pooled", pooled, "")
table("Split: test", by_split["test"], "")
table("Split: train", by_split["train"], "")
for c in ("A", "B"):
    cl = " — no class-level claim (MISS < 30 in this class)" if by_class[c]["n_miss"] < 30 else ""
    table("Clue class %s" % c, by_class[c], cl)
u = by_class["U"]
L.append("### Clue class U (descriptive only, n=%d)" % u["n"])
L.append("")
L.append("- Errors %d, MISS %d, RTRM-caught MISS %d. No verdict, per the protocol." % (u["n_errors"], u["n_miss"], u["miss_caught"]))
L.append("")
L.append("Per the protocol: for clue-after (class B) items the flag can only be a structural-prior "
         "warning — the pronoun state cannot contain the clue — and no encoding claim is made for that "
         "class whatever the numbers. Class A is where an encoding claim could live.")
L.append("")
L.append("## Secondary (reported regardless of verdict)")
L.append("")
L.append("exp-056 H1 table pooled (flag × error, all 617 items): flagged×[err,correct] / "
         "unflagged×[err,correct] = %s; one-sided Fisher p = %s."
         % (pooled["h1_table_flag_x_error_all"], pval(pooled["h1_fisher_p_onesided"])))
L.append("")
L.append("## Appendix (threshold-free comparison; appendix only, per protocol)")
L.append("")
L.append("- AUC of margin as an error score (score = −margin; 0.5 = chance): %s"
         % ("n/a" if pooled["appendix_auc_margin_error_score"] is None else "%.3f" % pooled["appendix_auc_margin_error_score"]))
L.append("- RTRM binary flag, (sensitivity + specificity) / 2: %s"
         % ("n/a" if pooled["appendix_rtrm_balanced_accuracy"] is None else "%.3f" % pooled["appendix_rtrm_balanced_accuracy"]))
L.append("")
L.append("## Complete MISS table — every error the margin baseline did not flag (%d items)" % len(miss_all_sorted))
L.append("")
L.append("| # | split-item | class | label | model pick | margin (nats) | label rank | competitor rank | RTRM lean | RTRM caught? |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for i, r in enumerate(miss_all_sorted, 1):
    L.append("| %d | %s-%s | %s | %s | %s | %.4f | %s | %s | %s | %s |"
             % (i, r["split"], r["item"], r["clue_class"], r["label"], r["model_pick"],
                r["margin_nats"], r["r_label"], r["r_comp"], r["rtrm_lean"],
                "YES" if r["rtrm_flag"] else "no"))
L.append("")
L.append("The %d RTRM-caught MISS items, with their subtracted-list top-20 words in rank order, "
         "are in `caught_miss_lists.json` for SME rendering. Full per-item records: "
         "`results_confirmatory.json`. Every number above: `analysis_confirmatory.json`. "
         "Completeness and determinism checks: `validation_confirmatory.json` and the run logs."
         % len(caught))
L.append("")
L.append("Known limitations (declared in the protocol): GPT-2 small only; forced choice on the first "
         "answer token is not free generation; DPR has been public since 2012 and may be in GPT-2's "
         "training text, which would inflate behaviour and the margin baseline together, biasing "
         "against an incremental RTRM claim. Nothing here is posted or published.")
open(D + "/RESULTS_confirmatory.md", "w").write("\n".join(L) + "\n")
print("verdict:", pooled["verdict"])
print("pooled:", json.dumps({k: pooled[k] for k in ("n","n_errors","n_miss","miss_caught","rtrm_catch_of_miss","sens_b1","sens_combined","far_b1","far_combined","sens_gain_pp","far_gain_pp","fisher_iv_p_onesided","h1_fisher_p_onesided","appendix_auc_margin_error_score","appendix_rtrm_balanced_accuracy")}, indent=1))
for s, bl in list(by_split.items()) + list(by_class.items()):
    print(s, "n=%d err=%d miss=%d caught=%d verdict=%s" % (bl["n"], bl["n_errors"], bl["n_miss"], bl["miss_caught"], bl["verdict"]))
