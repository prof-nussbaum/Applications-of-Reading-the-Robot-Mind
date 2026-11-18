"""Score the blind reading test (arm D seed 0) against the sealed key.

Reader judgments are parsed tolerantly from reader_set.md: the reader
marked answers by hand with inconsistent bold markers (**PASS, **PASS**,
*FAIL**, FAIL** all occur), so the Task lines are searched for the
judgment words themselves. One Task 2 line reads "STORE" (BT-34) and is
counted as STORED — the only two options are STORED / MERELY FAMILIAR.
Bars were locked in PROTOCOL_BLIND_TEST.md before the reader saw a list:
Task 1 passes iff reader balanced accuracy EXCEEDS the lens rank-1
rule's 0.6471; Task 2 passes iff it EXCEEDS the lens band rule's 0.8000.
"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
text = (HERE / 'reader_set.md').read_text()
key = json.load(open(HERE / 'answer_key.json'))['items']
blocks = re.split(r'## Item (BT-\d+)', text)
ans = {}
for i in range(1, len(blocks), 2):
    bid, body = blocks[i], blocks[i + 1]
    l1 = re.search(r'Task 1[^\n]*', body).group(0)
    l2 = re.search(r'Task 2[^\n]*', body).group(0)
    t1 = 'PASS' if re.search(r'\bPASS\b', l1) else ('FAIL' if re.search(r'\bFAIL\b', l1) else None)
    t2 = 'STORED' if re.search(r'STORE', l2) else ('MERELY FAMILIAR' if 'MERELY' in l2 else None)
    ans[bid] = (t1, t2)

def stats(pairs):
    tp = sum(1 for p, a in pairs if p and a); fn = sum(1 for p, a in pairs if not p and a)
    tn = sum(1 for p, a in pairs if not p and not a); fp = sum(1 for p, a in pairs if p and not a)
    sens = tp / (tp + fn) if tp + fn else float('nan')
    spec = tn / (tn + fp) if tn + fp else float('nan')
    return dict(n=len(pairs), raw_correct=tp + tn, tp=tp, fn=fn, tn=tn, fp=fp,
                sensitivity=round(sens, 4), specificity=round(spec, 4),
                balanced_accuracy=round((sens + spec) / 2, 4))

t1 = [(a[0] == 'PASS', key[b]['pass_8tok']) for b, a in ans.items() if a[0]]
t2 = [(a[1] == 'STORED', key[b]['subset'] == 'stored-trained') for b, a in ans.items() if a[1]]
print('Task 1 (bar: exceed 0.6471):', stats(t1))
print('Task 2 (bar: exceed 0.8000):', stats(t2))
