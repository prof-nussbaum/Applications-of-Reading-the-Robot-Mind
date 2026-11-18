"""Substitution accuracy split by candidate type (name vs common noun),
adopted from referee R-001 (F-001 section 4). Banked records only.

Join: subst_rescore_617.json (per-item question/substitution correctness)
with confirmatory_items.jsonl (candidate heads). Classification rule,
stated in advance of reporting: an item is a NAME item if both candidate
heads are capitalised proper names, a NOUN item if neither is, MIXED if
exactly one is. Totals must reproduce the banked 367/617 (substitution)
and 320/617 (question format).
"""
import json
from collections import Counter

BASE = '../../post-book-experiments/pronoun-resolution/path2-early-warning/'
items = {}
for line in open(BASE + 'confirmatory_items.jsonl'):
    d = json.loads(line)
    items[(d['split'], d['item'])] = d
sub = json.load(open('subst_rescore_617.json'))

def classify(d):
    named = [d[h][:1].isupper() for h in ('head1', 'head2')]
    if all(named):
        return 'name'
    if not any(named):
        return 'noun'
    return 'mixed'

stats = {}
for r in sub:
    c = classify(items[(r['split'], r['item'])])
    s = stats.setdefault(c, Counter())
    s['n'] += 1
    s['sub_right'] += r['sub_right']
    s['q_right'] += r['question_right']

tot_sub = tot_q = tot_n = 0
for c, s in sorted(stats.items()):
    print(f"{c}: n={s['n']} sub_acc={s['sub_right'] / s['n']:.3f} ({s['sub_right']}) "
          f"q_acc={s['q_right'] / s['n']:.3f} ({s['q_right']})")
    tot_sub += s['sub_right']; tot_q += s['q_right']; tot_n += s['n']
print(f"TOTAL: n={tot_n} sub={tot_sub} q={tot_q}  (banked: 367 / 320 of 617)")
assert (tot_n, tot_sub, tot_q) == (617, 367, 320)
