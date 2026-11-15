"""Exploratory join: rescore the locked Path 2 checklist against
substitution-defined errors.

Inputs:
  - results_confirmatory.json (pronoun-resolution/path2-early-warning/):
    per-item rtrm_flag, margin_flag_B1, margin_nats, model_right.
  - subst_rescore_617.json (this folder): per-item sub_right.
Items are keyed by (split, item) -- item numbers restart per split.

Sanity: against question-format errors this reproduces the locked
headline exactly (268 margin-missed errors, 204 caught, 76.1%;
flag on 72.8% of correct items; balanced accuracy 0.518).
"""
import json
from math import comb

CONF = "results_confirmatory.json"  # path to the confirmatory records
SUB = "subst_rescore_617.json"

conf = {(c["split"], c["item"]): c
        for c in json.load(open(CONF))}
sub = {(s["split"], s["item"]): s
       for s in json.load(open(SUB))}
rows = [(conf[k], sub[k]) for k in sorted(conf)]
assert len(rows) == 617

def fisher_onesided(a, b, c_, d):
    n = a + b + c_ + d
    r1, c1 = a + b, a + c_
    hi = min(r1, c1)
    return sum(comb(c1, x) * comb(n - c1, r1 - x) / comb(n, r1)
               for x in range(a, hi + 1))

def auc(scores, positives):
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    wins = ties = 0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1
            elif p == q:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))

for name, key in (("question-format", "question_right"),
                  ("substitution", "sub_right")):
    errs = [(c, s) for c, s in rows if not s[key]]
    cors = [(c, s) for c, s in rows if s[key]]
    fe = sum(1 for c, s in errs if c["rtrm_flag"])
    fc = sum(1 for c, s in cors if c["rtrm_flag"])
    ba = (fe / len(errs) + 1 - fc / len(cors)) / 2
    p = fisher_onesided(fe, len(errs) - fe, fc, len(cors) - fc)
    missed = [(c, s) for c, s in errs if not c["margin_flag_B1"]]
    mc = sum(1 for c, s in missed if c["rtrm_flag"])
    print(f"{name}: errors {len(errs)} (flagged {fe}); correct "
          f"{len(cors)} (flagged {fc}); BA {ba:.3f}; Fisher p {p:.3g}; "
          f"margin-missed errors {len(missed)}, checklist catch "
          f"{mc} ({mc / len(missed):.1%}); unsigned-margin AUC "
          f"{auc([c['margin_nats'] for c, s in rows], [s[key] for c, s in rows]):.3f}")
