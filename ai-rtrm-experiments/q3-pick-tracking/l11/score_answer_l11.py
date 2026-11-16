"""Q3 Amendment 1, stage 2: scoring of the ANSWER-position L11 list.

Order of operations (locked supplement): the scoring code FIRST
re-checks the amended strata and reproduces the stage-1 pronoun
baselines from pronoun_l11_ranks.json; only then does it touch the
answer-position leans. Any failed check exits nonzero before a result
is printed.

PRIMARY target: the question-format pick.
  Same-Q pairs bar: share >= 59.8% (baseline 108/217 = 49.8%, +10 rule)
  Non-twin bar:     share >= 56.2% (baseline 66/143 = 46.2%, +10 rule)
  Flip-Q pairs:     no bar — both-match count + exact (Clopper-Pearson)
                    binomial 95% interval.
Substitution-target figures: reported, no bar.
Beside the result: the L12 amended result from the committed
q3_scores.json (list L12 vs question pick), asserted against
AMENDMENT_1.md (130/217, 105/143, 5/20).
Writes scores_answer_l11.json.
"""
import json, sys
from math import comb
sys.path.insert(0, "/home/ecpi-student-05/robot-mind/q3-pick-tracking/l11")
from stage2_common import pair_stats, q_pick, sub_pick, L11, Q3

# ---------- exact binomial interval (Clopper-Pearson, pure python) ----------
def binom_cdf(k, n, p):
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(0, k + 1))

def cp_interval(k, n, alpha=0.05):
    if k == 0:
        lo = 0.0
    else:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if 1 - binom_cdf(k - 1, n, m) > alpha / 2:  # P(X>=k|m) too big -> root below m
                b = m
            else:
                a = m
        lo = (a + b) / 2
    if k == n:
        hi = 1.0
    else:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if binom_cdf(k, n, m) > alpha / 2:  # P(X<=k|m) too big -> root above m
                a = m
            else:
                b = m
        hi = (a + b) / 2
    return lo, hi

# CI machinery self-check against AMENDMENT_1.md's published L12 flip-Q interval
_lo, _hi = cp_interval(5, 20)
assert abs(_lo - 0.087) < 0.002 and abs(_hi - 0.491) < 0.002, (_lo, _hi)
print("CI self-check: 5/20 -> [%.3f, %.3f] (AMENDMENT_1: [0.087, 0.491]) OK" % (_lo, _hi), flush=True)

# ---------- gate: strata + stage-1 baseline reproduction ----------
fails = []
def check(name, got, want):
    ok = got == want
    print("CHECK %-52s got=%s want=%s %s" % (name, got, want, "OK" if ok else "** MISMATCH **"), flush=True)
    if not ok:
        fails.append(name)

pron = {(r["split"], r["item"]): r for r in json.load(open(L11 + "/pronoun_l11_ranks.json"))}
check("pronoun_l11_ranks records", len(pron), 617)
l11p = lambda s, i: pron[(s, i)]["lean_l11_pronoun"]
bq = pair_stats(l11p, q_pick)
check("baseline vs Q: same-Q pairs", bq["same_pairs"], 217)
check("baseline vs Q: same-Q both", bq["same_both"], 108)
check("baseline vs Q: flip-Q both", bq["flip_both"], 0)
check("baseline vs Q: non-twin match", bq["nontwin_match"], 66)
bs0 = pair_stats(l11p, sub_pick)
check("baseline vs sub: same-pick both", bs0["same_both"], 88)
check("baseline vs sub: flip both", bs0["flip_both"], 0)
check("baseline vs sub: non-twin match", bs0["nontwin_match"], 65)
if fails:
    print("SCORING GATE FAILED: %s — answer leans NOT scored" % fails, flush=True)
    raise SystemExit(1)
print("SCORING GATE: ALL OK", flush=True)

# ---------- answer-position scoring ----------
ans = {(r["split"], r["item"]): r for r in json.load(open(L11 + "/answer_l11_ranks.json"))}
check("answer_l11_ranks records", len(ans), 617)
# extraction-record picks must agree with the banked joins
disagree = sum(1 for k, r in ans.items()
               if r["question_pick"] != q_pick(*k) or r["sub_pick"] != sub_pick(*k))
check("answer records' picks disagreeing with banked", disagree, 0)
if fails:
    print("SCORING GATE FAILED: %s" % fails, flush=True)
    raise SystemExit(1)

l11a = lambda s, i: ans[(s, i)]["lean_l11_answer"]
aq = pair_stats(l11a, q_pick)
asub = pair_stats(l11a, sub_pick)

BAR_SAME_Q = 0.598
BAR_NONTWIN = 0.562
same_q_pass = aq["same_share"] >= BAR_SAME_Q
nontwin_pass = aq["nontwin_share"] >= BAR_NONTWIN
flo, fhi = cp_interval(aq["flip_both"], aq["flip_pairs"])

# ---------- L12 amended comparison (committed q3_scores.json) ----------
q3s = json.load(open(Q3 + "/q3_scores.json"))
l12 = q3s["list_L12"]["vs_question_pick"]
assert (l12["same_pick_both"], l12["nontwin_match"], l12["flip_both"]) == (130, 105, 5), l12
l12f_lo, l12f_hi = cp_interval(l12["flip_both"], l12["flip_pairs"])
l12_cmp = {
    "sameQ": {"both": l12["same_pick_both"], "pairs": l12["same_pick_pairs"],
              "share": l12["same_pick_both"] / l12["same_pick_pairs"], "bar": 0.635, "pass": l12["same_pick_both"] / l12["same_pick_pairs"] >= 0.635},
    "nontwin": {"match": l12["nontwin_match"], "items": l12["nontwin_items"],
                "share": l12["nontwin_match"] / l12["nontwin_items"], "bar": 0.610, "pass": l12["nontwin_match"] / l12["nontwin_items"] >= 0.610},
    "flipQ": {"both": l12["flip_both"], "pairs": l12["flip_pairs"],
              "share": l12["flip_both"] / l12["flip_pairs"], "ci95": [l12f_lo, l12f_hi]},
}

out = {
    "primary_target": "question-format pick",
    "bars": {"sameQ_share": BAR_SAME_Q, "nontwin_share": BAR_NONTWIN,
             "baselines": {"sameQ": "108/217 = 49.8%", "nontwin": "66/143 = 46.2%"}},
    "L11_answer_vs_question_pick": dict(aq, flip_ci95=[flo, fhi],
                                        sameQ_pass=same_q_pass, nontwin_pass=nontwin_pass,
                                        primary_verdict="PASS" if same_q_pass else "FAIL"),
    "L11_answer_vs_sub_pick": asub,
    "L11_pronoun_baselines": {"vs_question_pick": bq, "vs_sub_pick": bs0},
    "L12_answer_vs_question_pick_amended": l12_cmp,
}
json.dump(out, open(L11 + "/scores_answer_l11.json", "w"), indent=1)

print("PRIMARY (vs question pick) same-Q: %d/%d = %.1f%%  bar >= 59.8%%  -> %s" %
      (aq["same_both"], aq["same_pairs"], 100 * aq["same_share"], "PASS" if same_q_pass else "FAIL"), flush=True)
print("Non-twin: %d/%d = %.1f%%  bar >= 56.2%%  -> %s" %
      (aq["nontwin_match"], aq["nontwin_items"], 100 * aq["nontwin_share"], "PASS" if nontwin_pass else "FAIL"), flush=True)
print("Flip-Q: both %d/%d = %.1f%%  exact 95%% CI [%.1f%%, %.1f%%] (no bar)" %
      (aq["flip_both"], aq["flip_pairs"], 100 * aq["flip_share"], 100 * flo, 100 * fhi), flush=True)
print("Vs substitution pick (no bar): same-pick %d/%d = %.1f%% | flip %d/%d | non-twin %d/%d = %.1f%%" %
      (asub["same_both"], asub["same_pairs"], 100 * asub["same_share"],
       asub["flip_both"], asub["flip_pairs"],
       asub["nontwin_match"], asub["nontwin_items"], 100 * asub["nontwin_share"]), flush=True)
print("SCORING_DONE", flush=True)
