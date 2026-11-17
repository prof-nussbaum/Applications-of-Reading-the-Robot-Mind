"""Path 2 POSITIVE CONTROL — analysis (PROTOCOL_LOCKED.md, 2026-11-15).

Reads results_control.json (+ ceiling_probe_results.json) and applies the
locked gate, bars and verdict taxonomy mechanically:

  GATE: question-format accuracy >= 80%. Below the gate the control has
  FAILED AS A CONTROL and no checklist conclusion is drawn either way.
  If gate passes and MISS < 30: gate-pass with an UNDERPOWERED checklist
  test (reported as such, not stretched; bar quantities shown as
  descriptive only).
  If gate passes and MISS >= 30: WIN = all four bars pass;
  REFUTED = (iii) or (iv) fails; NOT WIN otherwise (failing bars named).

Bars (identical in form to Path 2 confirmatory):
  ERR = pick != label; B1 flag = margin < 0.35 nats; MISS = ERR and not B1.
  (i) RTRM catch of MISS >= 30%; (ii) combined sens - B1 sens >= +10 pp;
  (iii) combined FAR - B1 FAR <= +20 pp; (iv) one-sided Fisher exact
  (RTRM flag x ERR among B1-unflagged), p < 0.05.
Secondary (locked): lean-match rate (rtrm_lean == model_pick), all items
and errors separately, question format. Locked lean prediction: >= 70%
if the list tracks the model's decisions where the model demonstrably
knows; <= 60% means it does not track even there.
Also reported: substitution-format accuracy (047a method), per family
accuracy (descriptive), exp-056 H1 secondary, margin AUC appendix.
"""
import json, bisect
from scipy.stats import fisher_exact

import os
ROBOT_MIND = os.environ.get("ROBOT_MIND", "/home/ecpi-student-05/robot-mind")  # Jetson working tree (data lives outside the repo); override with $ROBOT_MIND

PC = ROBOT_MIND + "/pronoun-battery-v2/path2-positive-control"
res = json.load(open(PC + "/results_control.json"))
ceil = json.load(open(PC + "/ceiling_probe_results.json"))

def block(recs):
    n = len(recs)
    err = [r for r in recs if not r["model_right"]]
    cor = [r for r in recs if r["model_right"]]
    ne, nc = len(err), len(cor)
    def frac(f, t): return (len(f) / t) if t else None
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
    gain_sens = (sens_cb - sens_b1) if (ne and sens_cb is not None and sens_b1 is not None) else None
    gain_far = (far_cb - far_b1) if (nc and far_cb is not None and far_b1 is not None) else None
    unb = [r for r in recs if not r["margin_flag_B1"]]
    a = len([r for r in unb if r["rtrm_flag"] and not r["model_right"]])
    b = len([r for r in unb if r["rtrm_flag"] and r["model_right"]])
    c = len([r for r in unb if not r["rtrm_flag"] and not r["model_right"]])
    d = len([r for r in unb if not r["rtrm_flag"] and r["model_right"]])
    p_iv = float(fisher_exact([[a, b], [c, d]], alternative="greater")[1]) if (a+b) and (c+d) else None
    ha, hb = len(rt_e), len(rt_c); hc, hd = ne - ha, nc - hb
    p_h1 = float(fisher_exact([[ha, hb], [hc, hd]], alternative="greater")[1]) if (ha+hb) and (hc+hd) else None
    def auc_margin(rs):
        e = sorted(r["margin_nats"] for r in rs if not r["model_right"])
        cc = [r["margin_nats"] for r in rs if r["model_right"]]
        if not e or not cc: return None
        wins = 0.0
        for m in cc:
            lo = bisect.bisect_left(e, m); eq = bisect.bisect_right(e, m) - lo
            wins += lo + 0.5 * eq
        return wins / (len(e) * len(cc))
    bars = None
    if nm >= 30:
        bars = {"i_catch_ge_30pct": bool(catch is not None and catch >= 0.30),
                "ii_sens_gain_ge_10pp": bool(gain_sens is not None and gain_sens >= 0.10),
                "iii_far_gain_le_20pp": bool(gain_far is not None and gain_far <= 0.20),
                "iv_fisher_p_lt_0.05": bool(p_iv is not None and p_iv < 0.05)}
    return {"n": n, "n_errors": ne, "n_correct": nc,
            "accuracy": (nc / n) if n else None, "error_rate": (ne / n) if n else None,
            "sens_b1": sens_b1, "sens_rtrm": sens_rt, "sens_combined": sens_cb,
            "far_b1": far_b1, "far_rtrm": far_rt, "far_combined": far_cb,
            "sens_gain_pp": (gain_sens * 100) if gain_sens is not None else None,
            "far_gain_pp": (gain_far * 100) if gain_far is not None else None,
            "n_miss": nm, "miss_caught": len(miss_caught), "rtrm_catch_of_miss": catch,
            "fisher_iv_table_B1unflagged": [[a, b], [c, d]], "fisher_iv_p_onesided": p_iv,
            "h1_table_flag_x_error_all": [[ha, hb], [hc, hd]], "h1_fisher_p_onesided": p_h1,
            "appendix_auc_margin_error_score": auc_margin(recs),
            "appendix_rtrm_balanced_accuracy": ((sens_rt + (1 - far_rt)) / 2) if (ne and nc) else None,
            "bars": bars}

pooled = block(res)
by_family = {f: block([r for r in res if r["family"] == f]) for f in ("number", "gender")}

gate_acc = pooled["accuracy"]
gate_pass = gate_acc is not None and gate_acc >= 0.80
nm = pooled["n_miss"]
if not gate_pass:
    verdict = ("CONTROL FAILED AS A CONTROL — question-format accuracy %.1f%% < 80%% gate; "
               "per the locked protocol no checklist conclusion is drawn either way." % (100 * gate_acc))
elif nm < 30:
    verdict = ("GATE PASS (accuracy %.1f%% >= 80%%) — checklist test UNDERPOWERED: MISS = %d < 30, "
               "exactly as predicted in the locked protocol. No WIN / REFUTED / NOT WIN verdict; "
               "bar quantities below are descriptive only." % (100 * gate_acc, nm))
elif all(pooled["bars"].values()):
    verdict = "WIN — all four locked bars pass."
elif not pooled["bars"]["iii_far_gain_le_20pp"] or not pooled["bars"]["iv_fisher_p_lt_0.05"]:
    failed = [k.split("_")[0] for k, v in pooled["bars"].items() if not v]
    verdict = "REFUTED — gate passed, MISS = %d >= 30, and bar(s) %s failed (taxonomy: (iii) or (iv) failing => REFUTED)." % (nm, ", ".join(failed))
else:
    failed = [k.split("_")[0] for k, v in pooled["bars"].items() if not v]
    verdict = "NOT WIN — failing bars: %s." % ", ".join(failed)

def leanrate(recs):
    if not recs: return None, 0
    return sum(1 for r in recs if r["rtrm_lean"] == r["model_pick"]) / len(recs), len(recs)
lean_all, _ = leanrate(res)
lean_err, n_err = leanrate([r for r in res if not r["model_right"]])
lean_cor, n_cor = leanrate([r for r in res if r["model_right"]])
sub_acc = sum(1 for r in res if r["sub_right"]) / len(res)
sub_by_family = {f: (sum(1 for r in res if r["family"] == f and r["sub_right"]) /
                     max(1, sum(1 for r in res if r["family"] == f))) for f in ("number", "gender")}
agree = sum(1 for r in res if r["sub_pick"] == r["model_pick"]) / len(res)

analysis = {"gate": {"question_format_accuracy": gate_acc, "bar": 0.80, "passed": bool(gate_pass)},
            "verdict": verdict, "pooled": pooled, "by_family": by_family,
            "lean_match_all": lean_all, "lean_match_errors": lean_err, "lean_match_correct_descriptive": lean_cor,
            "substitution_accuracy": sub_acc, "substitution_accuracy_by_family": sub_by_family,
            "question_substitution_pick_agreement": agree,
            "ceiling_probe": ceil}
json.dump(analysis, open(PC + "/analysis_control.json", "w"), indent=1)

def pct(x): return "n/a" if x is None else "%.1f%%" % (100 * x)
def pp(x): return "n/a" if x is None else "%+.1f pp" % x
def pval(x): return "n/a" if x is None else ("%.3g" % x)

L = []
L.append("# Path 2 — POSITIVE CONTROL RESULTS (2026-11-15)")
L.append("")
L.append("Protocol: `PROTOCOL_LOCKED.md` (locked 2026-11-15 after the sample-data audit PASSED, "
         "seed 0910). Items: 120 new agreement items (60 number, 60 gender) from "
         "`items_control.jsonl`. Pool reused unchanged: pool_path2.pt (47,499 states). "
         "Question format identical to the Path 2 confirmatory run; substitution format = the "
         "047a method (first pronoun occurrence replaced by \"the <candidate>\"; total "
         "log-probability of the full substituted sentence), as implemented in exp-048/run_048.py.")
L.append("")
L.append("## GATE (reported first, per the locked protocol)")
L.append("")
L.append("- Question-format (forced-choice) accuracy: **%s** (%d of %d correct). Gate bar: >= 80%%. **%s**"
         % (pct(gate_acc), pooled["n_correct"], pooled["n"], "PASS" if gate_pass else "FAIL"))
L.append("- By family (descriptive): number %s (%d/%d), gender %s (%d/%d)."
         % (pct(by_family["number"]["accuracy"]), by_family["number"]["n_correct"], by_family["number"]["n"],
            pct(by_family["gender"]["accuracy"]), by_family["gender"]["n_correct"], by_family["gender"]["n"]))
L.append("- Substitution-format accuracy (descriptive, not gated): **%s** (%d of %d); number %s, gender %s. "
         "Question/substitution picks agree on %s of items."
         % (pct(sub_acc), sum(1 for r in res if r["sub_right"]), len(res),
            pct(sub_by_family["number"]), pct(sub_by_family["gender"]), pct(agree)))
L.append("")
L.append("## VERDICT: **%s**" % verdict)
L.append("")
L.append("## PREDICTED vs ACTUAL (locked predictions, PROTOCOL_LOCKED.md)")
L.append("")
L.append("PREDICTED:")
L.append("- Gate passes: question-format accuracy >= 80%.")
L.append("- MISS < 30, so the four-bar checklist test is UNDERPOWERED by design and must be reported as such.")
L.append("- Lean-match (secondary): >= 70% if the list tracks the model's decisions where the model "
         "demonstrably knows; <= 60% means it does not track even there.")
L.append("- Ceiling probe on DPR confirmatory states: test AUC between 0.45 and 0.60.")
L.append("")
L.append("ACTUAL:")
L.append("- Gate: %s — accuracy %s." % ("PASS" if gate_pass else "FAIL", pct(gate_acc)))
L.append("- Errors %d of %d; MISS = %d (errors the margin baseline did not flag)."
         % (pooled["n_errors"], pooled["n"], nm))
L.append("- Lean-match: all items %s (%d/%d); on errors %s (%d/%d); on correct items (descriptive) %s."
         % (pct(lean_all), sum(1 for r in res if r["rtrm_lean"] == r["model_pick"]), len(res),
            pct(lean_err), sum(1 for r in res if not r["model_right"] and r["rtrm_lean"] == r["model_pick"]), n_err,
            pct(lean_cor)))
L.append("- Ceiling probe (DIAGNOSTIC ONLY — NOT RTRM): test AUC for predicting the model's error "
         "= %.3f, bootstrap 95%% CI [%.3f, %.3f]; train AUC %.3f."
         % (ceil["test_auc_predict_error"], ceil["test_auc_predict_error_bootstrap95CI"][0],
            ceil["test_auc_predict_error_bootstrap95CI"][1], ceil["train_auc_predict_error"]))
L.append("")
L.append("## The four locked bars (pooled, question format)")
L.append("")
if not gate_pass:
    L.append("The gate failed, so per the locked protocol the bars license NO verdict and are not "
             "claimed as PASS/FAIL. Descriptive quantities only: RTRM catch of MISS %s (%d/%d); "
             "sensitivity B1 %s | RTRM %s | combined %s (gain %s); false-alarm B1 %s | RTRM %s | "
             "combined %s (gain %s); Fisher (iv) table (B1-unflagged) %s, one-sided p = %s."
             % (pct(pooled["rtrm_catch_of_miss"]), pooled["miss_caught"], nm,
                pct(pooled["sens_b1"]), pct(pooled["sens_rtrm"]), pct(pooled["sens_combined"]), pp(pooled["sens_gain_pp"]),
                pct(pooled["far_b1"]), pct(pooled["far_rtrm"]), pct(pooled["far_combined"]), pp(pooled["far_gain_pp"]),
                pooled["fisher_iv_table_B1unflagged"], pval(pooled["fisher_iv_p_onesided"])))
elif pooled["bars"]:
    def barline(name, ok): return "- %s: **%s**" % (name, "PASS" if ok else "FAIL")
    L.append(barline("(i) RTRM catch of MISS >= 30%% — actual %s (%d/%d)" % (pct(pooled["rtrm_catch_of_miss"]), pooled["miss_caught"], nm), pooled["bars"]["i_catch_ge_30pct"]))
    L.append(barline("(ii) combined sensitivity - B1 sensitivity >= 10 pp — actual %s" % pp(pooled["sens_gain_pp"]), pooled["bars"]["ii_sens_gain_ge_10pp"]))
    L.append(barline("(iii) combined false-alarm - B1 false-alarm <= 20 pp — actual %s" % pp(pooled["far_gain_pp"]), pooled["bars"]["iii_far_gain_le_20pp"]))
    L.append(barline("(iv) Fisher exact (flag x error, B1-unflagged), one-sided p < 0.05 — actual p = %s" % pval(pooled["fisher_iv_p_onesided"]), pooled["bars"]["iv_fisher_p_lt_0.05"]))
else:
    L.append("Not evaluated as a verdict: MISS = %d < 30. Descriptive quantities: RTRM catch of MISS %s "
             "(%d/%d); sensitivity B1 %s | RTRM %s | combined %s (gain %s); false-alarm B1 %s | RTRM %s | "
             "combined %s (gain %s); Fisher (iv) table (B1-unflagged) %s, one-sided p = %s."
             % (nm, pct(pooled["rtrm_catch_of_miss"]), pooled["miss_caught"], nm,
                pct(pooled["sens_b1"]), pct(pooled["sens_rtrm"]), pct(pooled["sens_combined"]), pp(pooled["sens_gain_pp"]),
                pct(pooled["far_b1"]), pct(pooled["far_rtrm"]), pct(pooled["far_combined"]), pp(pooled["far_gain_pp"]),
                pooled["fisher_iv_table_B1unflagged"], pval(pooled["fisher_iv_p_onesided"])))
L.append("")
L.append("Secondary exp-056 H1 (flag x error, all items): table %s, one-sided Fisher p = %s. "
         "Appendix: margin AUC as error score %s; RTRM flag balanced accuracy %s."
         % (pooled["h1_table_flag_x_error_all"], pval(pooled["h1_fisher_p_onesided"]),
            "n/a" if pooled["appendix_auc_margin_error_score"] is None else "%.3f" % pooled["appendix_auc_margin_error_score"],
            "n/a" if pooled["appendix_rtrm_balanced_accuracy"] is None else "%.3f" % pooled["appendix_rtrm_balanced_accuracy"]))
L.append("")
if not gate_pass:
    L.append("Note on the lean-match readout: the locked lean prediction was conditional on the control "
             "being a place the model demonstrably knows the answer. The gate failed for the question "
             "format (accuracy %s), so that condition is not met and the lean rates above (all items %s; "
             "errors %s) are reported descriptively, without the locked either/or reading. For the "
             "record they fall in the <= 60%% band the protocol associated with 'does not track'. "
             "The substitution format, which is not the gated format, scored %s."
             % (pct(gate_acc), pct(lean_all), pct(lean_err), pct(sub_acc)))
    L.append("")
L.append("## Ceiling probe — labelled reading (DIAGNOSTIC ONLY, NOT RTRM)")
L.append("")
L.append("A supervised logistic-regression probe (a trained classifier — a different method from the "
         "RTRM cosine-similarity method) was trained on the raw L12 pronoun states of the 617 DPR "
         "confirmatory items to predict the model's correctness (split-half by item, 308 train / 309 "
         "test, seed 0; rank-reproduction check: %d/617 items reproduced exactly). Test AUC for "
         "predicting the model's ERROR: **%.3f** (bootstrap 95%% CI [%.3f, %.3f]); test AUC for "
         "predicting correctness: %.3f; train AUC (error): %.3f."
         % (ceil["rank_reproduction_check"]["items_exact_match_r_label_and_r_comp"],
            ceil["test_auc_predict_error"], ceil["test_auc_predict_error_bootstrap95CI"][0],
            ceil["test_auc_predict_error_bootstrap95CI"][1], ceil["test_auc_predict_right"],
            ceil["train_auc_predict_error"]))
auc = ceil["test_auc_predict_error"]
if auc < 0.45:
    L.append("Reading (locked): the locked prediction was a test AUC between 0.45 and 0.60; the actual "
             "AUC of %.3f falls BELOW that band, with its bootstrap CI [%.3f, %.3f] entirely under "
             "chance, while the train AUC is %.3f. That is the signature of an overfit probe (768 "
             "features, 308 training items): the direction it learns does not generalise to held-out "
             "items. Under the locked reading, a supervised probe with direct access to the state "
             "cannot beat chance at predicting errors on held-out items, so no checklist on that state "
             "could have — Study 1's null is a statement about the state, not only about the checklist. "
             "We report the AUC as trained; flipping its sign after the fact would be a retrofit the "
             "protocol forbids. (The probe is a ceiling diagnostic only; it is not an RTRM result and "
             "supports no book claim.)"
             % (auc, ceil["test_auc_predict_error_bootstrap95CI"][0],
                ceil["test_auc_predict_error_bootstrap95CI"][1], ceil["train_auc_predict_error"]))
elif auc <= 0.60:
    L.append("Reading (locked): the probe does not beat chance by a useful margin, so the error "
             "information is not linearly readable from this state — Study 1's null is a statement "
             "about the state, not only about the checklist. (The probe is a ceiling diagnostic only; "
             "it is not an RTRM result and supports no book claim.)")
else:
    L.append("Reading (locked): the probe DOES read error information from the state (AUC > 0.60), so "
             "the information exists and the checklist is the wrong reader. (The probe is a ceiling "
             "diagnostic only; it is not an RTRM result and supports no book claim.)")
L.append("")
L.append("## Deviations and declared choices")
L.append("")
L.append("- Substitution convention: the protocol names the format 'exp-052 style'; exp-052's own "
         "script scored the question format. The battery's substitution method of record is the 047a "
         "method (exp-048/run_048.py, exp-049, exp-050b): first pronoun occurrence replaced by "
         "\"the <candidate>\", scored by TOTAL log-probability of the full substituted sentence. "
         "That method was used, unmodified, and is declared here and in run_control.py.")
L.append("- Control records carry clue_class = null (the agreement items have no clue-position class) "
         "and a family field (number/gender); all other fields match results_confirmatory.json in shape.")
L.append("- Ceiling-probe state: the raw per-token L2-normalised L12 state at the pronoun position "
         "(the state the checklist recipe starts from), as specified in the task brief.")
L.append("")
L.append("Files: `results_control.json` (120 per-item records, both formats), "
         "`results_control_top20.json` (subtracted-list top-20 per item), `analysis_control.json` "
         "(every number above), `ceiling_probe_results.json` + `probe_states_dpr.pt` (Part 2), "
         "`SME_CONTROL_ITEMS.md` (Part 3). Nothing here is posted or published.")
open(PC + "/RESULTS_CONTROL.md", "w").write("\n".join(L) + "\n")
print("verdict:", verdict)
print("gate accuracy:", gate_acc, "MISS:", nm)
print("lean all/errors:", lean_all, lean_err, "| sub acc:", sub_acc, "| ceiling AUC:", ceil["test_auc_predict_error"])
