"""Control: neighbour-list lean vs the model's substitution pick,
adopted from referee R-001 (F-001 section 4). Banked records only:
results_control.json carries rtrm_lean and sub_pick per item.
Checks: lean == label must reproduce the banked 61.7% (74/120) and
sub_pick == label the banked 80.0% (96/120).
"""
import json

rows = json.load(open('results_control.json'))

def rate(rs, f):
    n = len(rs)
    k = sum(1 for r in rs if f(r))
    return f"{k}/{n} = {k / n:.3f}"

print('lean == substitution pick, all:   ', rate(rows, lambda r: r['rtrm_lean'] == r['sub_pick']))
for fam in ('number', 'gender'):
    rs = [r for r in rows if r['family'] == fam]
    print(f'lean == substitution pick, {fam}: ', rate(rs, lambda r: r['rtrm_lean'] == r['sub_pick']))
print('lean == question pick, all:       ', rate(rows, lambda r: r['rtrm_lean'] == r['model_pick']))
print('lean == label, all (banked 61.7%):', rate(rows, lambda r: r['rtrm_lean'] == r['label']))
print('sub pick == label (banked 80.0%): ', rate(rows, lambda r: r['sub_right']))
assert sum(r['rtrm_lean'] == r['label'] for r in rows) == 74
assert sum(r['sub_right'] for r in rows) == 96
