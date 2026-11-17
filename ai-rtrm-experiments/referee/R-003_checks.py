"""R-003 checks, from PR #17's own records (stdlib only).
Usage: python3 r003_checks.py <repo root>"""
import json, sys, random, collections, math
root = sys.argv[1] + '/ai-rtrm-experiments/'

print("== A. Arm D, reversed probes: lens verdict vs behaviour ==")
v = json.load(open(root + 'token-exposure-control/verdicts_exposure.json'))['models']
b = json.load(open(root + 'token-exposure-control/behavior_exposure.json'))['models']
r = list(range(60)); random.Random(201).shuffle(r); rev_idx = set(r[:30])
tot = collections.Counter(); first_ok = collections.Counter()
for s in ('expD_s0', 'expD_s1', 'expD_s2'):
    for vi, bi in zip(v[s]['reversed'], b[s]['reversed']):
        assert vi['index'] == bi['index']
        sub = 'rev-trained' if vi['index'] in rev_idx else 'fwd-only'
        tot[(sub, vi['verdict_hs11'], bi['pass_8tok'])] += 1
        gen = bi['gen_8tok'].strip(); first = vi['expected_name'].split()[0]
        if sub == 'rev-trained' and not bi['pass_8tok']:
            first_ok[gen.startswith(first)] += 1
for sub in ('rev-trained', 'fwd-only'):
    for band in ('STORED', 'WEAK', 'MISSING'):
        p, f = tot[(sub, band, True)], tot[(sub, band, False)]
        if p + f: print(f"  {sub:11s} {band:8s} pass {p:3d}  fail {f:3d}  P(pass)={p/(p+f):.2f}")
print("  rev-trained failures whose output still begins with the correct first name:",
      first_ok[True], "of", first_ok[True] + first_ok[False])
st = sum(n for (sub, band, p), n in tot.items() if band == 'STORED')
print("  STORED verdicts in arms A/B/C (9 arm-seeds x 60):",
      sum(x['verdict_hs11'] == 'STORED' for s in v if 'expD' not in s for x in v[s]['reversed']))

print("\n== B. Q3: is the pass more than a second-noun bias? ==")
q = {(x['split'], x['item']): x for x in json.load(open(root + 'q3-pick-tracking/q3_results.json'))}
a = {(x['split'], x['item']): x for x in json.load(open(root + 'q3-pick-tracking/l11/answer_l11_ranks.json'))}
pairs = [json.loads(l) for l in open(root + 'q3-pick-tracking/twin_pairs.jsonl')]
non = [json.loads(l) for l in open(root + 'q3-pick-tracking/twin_nontwin_items.jsonl')]
P = [[(p['split'], it['item']) for it in p['items']] for p in pairs]
sameQ = [p for p in P if q[p[0]]['question_pick'] == q[p[1]]['question_pick']]
flipQ = [p for p in P if q[p[0]]['question_pick'] != q[p[1]]['question_pick']]
N = [(x['split'], x['item']) for x in non]
def lean(k, which):
    if which == 'L11': return a[k]['lean_l11_answer']
    if which == 'L12': return q[k]['list']['12']['lean']
    if which == 'second': return q[k]['head2']
    if which == 'first': return q[k]['head1']
def both(ps, which): return sum(all(lean(k, which) == q[k]['question_pick'] for k in p) for p in ps)
for which in ('L11', 'L12', 'second', 'first'):
    print(f"  {which:6s} same-Q both {both(sameQ, which):3d}/{len(sameQ)} = {both(sameQ, which)/len(sameQ):.3f}"
          f" | non-twin {sum(lean(k, which)==q[k]['question_pick'] for k in N):3d}/{len(N)}"
          f" = {sum(lean(k, which)==q[k]['question_pick'] for k in N)/len(N):.3f}"
          f" | flip-Q both {both(flipQ, which)}/{len(flipQ)}")
items = [k for p in P for k in p] + N
for which in ('L11', 'L12'):
    c = collections.Counter((q[k]['question_pick'] == q[k]['head2'], lean(k, which) == q[k]['head2']) for k in items)
    p2 = c[(True, True)] / (c[(True, True)] + c[(True, False)]); p1 = c[(False, True)] / (c[(False, True)] + c[(False, False)])
    print(f"  {which}: lean=second noun on {(c[(True,True)]+c[(False,True)])/len(items):.3f} of items;"
          f" P(lean=second | pick=second)={p2:.3f} (n={c[(True,True)]+c[(True,False)]}),"
          f" P(lean=second | pick=first)={p1:.3f} (n={c[(False,True)]+c[(False,False)]})")
print(f"  model's question pick is the second noun on {sum(q[k]['question_pick']==q[k]['head2'] for k in items)/len(items):.3f} of items")
# pair-level, bias-free: same-Q pairs split by which noun the model picked
for h, name in (('head2', 'second'), ('head1', 'first')):
    ps = [p for p in sameQ if q[p[0]]['question_pick'] == q[p[0]][h]]
    for which in ('L11', 'L12'):
        print(f"  same-Q pairs where the model picked the {name} noun: {which} both-match {both(ps, which)}/{len(ps)} = {both(ps, which)/len(ps):.3f}")

print("\n== C. Balanced per-item accuracy against the question pick (a constant rule scores 0.500) ==")
conf = {(x['split'], x['item']): x for x in json.load(open(sys.argv[1] + '/post-book-experiments/pronoun-resolution/path2-early-warning/results_confirmatory.json'))}
pr11 = {(x['split'], x['item']): x for x in json.load(open(root + 'q3-pick-tracking/l11/pronoun_l11_ranks.json'))}
def bal(get):
    c = collections.Counter((q[k]['question_pick'] == q[k]['head2'], get(k) == q[k]['head2']) for k in items)
    return (c[(True, True)] / (c[(True, True)] + c[(True, False)]) + c[(False, False)] / (c[(False, True)] + c[(False, False)])) / 2
lk = [k for k in pr11[items[0]] if k.startswith('lean')][0]
rows = [("list L12, pronoun", lambda k: conf[k]['rtrm_lean']), ("list L11, pronoun", lambda k: pr11[k][lk]),
        ("list L12, answer", lambda k: lean(k, 'L12')), ("list L11, answer", lambda k: lean(k, 'L11'))]
for L in ('9', '10', '11'):
    rows.append((f"lens L{L}, answer", lambda k, L=L: q[k]['lens'][L]['lean']))
for name, g in rows: print(f"  {name:20s} {bal(g):.3f}")

print("\n== D. Arm D rev-trained: does lens rank 1 (not just <=10) predict behaviour? ==")
c = collections.Counter()
for s in ('expD_s0', 'expD_s1', 'expD_s2'):
    for vi, bi in zip(v[s]['reversed'], b[s]['reversed']):
        if vi['index'] in rev_idx:
            for key in ('lens_rank_hs11', 'lens_rank_hs12'):
                rk = vi[key]; band = '1' if rk == 1 else ('2-10' if rk <= 10 else '>10')
                c[(key, band, bi['pass_8tok'])] += 1
for key in ('lens_rank_hs11', 'lens_rank_hs12'):
    for band in ('1', '2-10', '>10'):
        p, f = c[(key, band, True)], c[(key, band, False)]
        if p + f: print(f"  {key} rank {band:5s}: pass {p:2d} fail {f:2d}  P(pass)={p/(p+f):.2f}")
