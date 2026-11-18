"""R-006 check: what a fixed threshold on the information shown to the reader scores.
Usage: python3 r006_blind.py <repo root>"""
import json, sys
k = json.load(open(sys.argv[1] + '/ai-rtrm-experiments/blind-reading-test/answer_key.json'))['items']
it = list(k.values())
def bal(pred, truth):
    tp = sum(p and t for p, t in zip(pred, truth)); fn = sum((not p) and t for p, t in zip(pred, truth))
    tn = sum((not p) and (not t) for p, t in zip(pred, truth)); fp = sum(p and (not t) for p, t in zip(pred, truth))
    return (tp / (tp + fn) + tn / (tn + fp)) / 2, tp, fp, fn, tn
print("Task 1 (predict behaviour): rule 'PASS iff correct name is at lens rank <= k'")
truth = [x['pass_8tok'] for x in it]
for kk in (1, 2, 3, 4, 5, 7, 10, 20):
    b, tp, fp, fn, tn = bal([x['lens_rank_hs11'] <= kk for x in it], truth)
    print(f"  k={kk:2d}: balanced acc {b:.4f}  (hits {tp}/9, false alarms {fp}/51)")
print("Task 2 (stored vs familiar): rule 'STORED iff rank <= k'")
truth = [x['subset'] == 'stored-trained' for x in it]
for kk in (1, 3, 5, 10, 15, 20):
    b, tp, fp, fn, tn = bal([x['lens_rank_hs11'] <= kk for x in it], truth)
    print(f"  k={kk:2d}: balanced acc {b:.4f}  (stored found {tp}/30, familiar miscalled {fp}/30)")
import collections
print("ranks of the 9 passing items:", sorted(x['lens_rank_hs11'] for x in it if x['pass_8tok']))
print("stored-trained rank histogram:", sorted(collections.Counter(min(x['lens_rank_hs11'],21) for x in it if x['subset']=='stored-trained').items()))
print("forward-only min rank:", min(x['lens_rank_hs11'] for x in it if x['subset']!='stored-trained'))
