import json,sys,math
root=sys.argv[1]
r={(x['split'],x['item']):x for x in json.load(open(root+'/ai-rtrm-experiments/path2-early-warning-records/results_confirmatory.json'))}
for x in json.load(open(root+'/ai-rtrm-experiments/path2-substitution-rescore/subst_rescore_617.json')):
    y=r[(x['split'],x['item'])]; y['sub_pick']=y['label'] if x['sub_right'] else y['competitor']
Q=root+'/ai-rtrm-experiments/q3-pick-tracking/'
pairs=[json.loads(l) for l in open(Q+'twin_pairs.jsonl')]
non=[json.loads(l) for l in open(Q+'twin_nontwin_items.jsonl')]
print('pair record keys:',list(pairs[0].keys())); print('nontwin keys:',list(non[0].keys()))
def get(p):
    its=p.get('items') or [p.get('a'),p.get('b')]
    out=[]
    for it in its:
        k=(p['split'],it['item'] if isinstance(it,dict) else it)
        out.append(r[k])
    return out
same=flip=0; sb=fb=0
for p in pairs:
    a,b=get(p)
    agree=(a['model_pick']==a['sub_pick'])+(b['model_pick']==b['sub_pick'])
    if a['sub_pick']==b['sub_pick']: same+=1; sb+=agree==2
    else: flip+=1; fb+=agree==2
print(f"same-pick pairs: question pick == substitution pick on BOTH twins: {sb}/{same} = {sb/same:.3f}  (bar 0.674)")
print(f"flip pairs     : question pick == substitution pick on BOTH twins: {fb}/{flip} = {fb/flip:.3f}  (bar 0.50)")
n=[r[(x['split'],x['item'])] for x in non]
k=sum(x['model_pick']==x['sub_pick'] for x in n); print(f"non-twins      : question pick == substitution pick: {k}/{len(n)} = {k/len(n):.3f}  (bar 0.729)")
allk=sum(x['model_pick']==x['sub_pick'] for x in r.values()); print(f"all items: {allk}/{len(r)} = {allk/len(r):.3f}")
