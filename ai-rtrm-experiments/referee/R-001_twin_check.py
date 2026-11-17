import json, sys, re, math, collections
root=sys.argv[1]
P=root+'/post-book-experiments/pronoun-resolution/path2-early-warning/'
r=json.load(open(P+'results_confirmatory.json'))
s={(x['split'],x['item']):x for x in json.load(open(root+'/ai-rtrm-experiments/path2-substitution-rescore/subst_rescore_617.json'))}
for x in r:
    y=s[(x['split'],x['item'])]
    x['sub_right']=y['sub_right']; x['sub_pick']=x['label'] if y['sub_right'] else x['competitor']
    x['lean_is_label']=x['r_label']<x['r_comp']
    x['first']=x['model_pick']==x['head1']
n=len(r)
def rate(k,m): 
    p=k/m; se=math.sqrt(p*(1-p)/m); return f"{k}/{m} = {p:.3f} (95% CI {p-1.96*se:.3f}-{p+1.96*se:.3f})"
print("lean==label              :",rate(sum(x['lean_is_label'] for x in r),n))
print("lean==question pick      :",rate(sum(x['rtrm_lean']==x['model_pick'] for x in r),n))
print("lean==substitution pick  :",rate(sum(x['rtrm_lean']==x['sub_pick'] for x in r),n))
print("question pick is head1   :",rate(sum(x['first'] for x in r),n))
print("subst pick is head1      :",rate(sum(x['sub_pick']==x['head1'] for x in r),n))
print("lean is head1            :",rate(sum(x['rtrm_lean']==x['head1'] for x in r),n))
print("label is head1           :",rate(sum(x['label']==x['head1'] for x in r),n))
# does lean track subst pick beyond a shared first-noun bias?
for h in ('head1','head2'):
    sub=[x for x in r if x['sub_pick']==x[h]]
    print(f"  P(lean=head1 | subst pick={h}):",rate(sum(x['rtrm_lean']==x['head1'] for x in sub),len(sub)))
# twins: same text up to and including the pronoun, same heads
def prefix(x):
    m=re.search(r'\b'+re.escape(x['pronoun'])+r'\b',x['sentence'])
    return (x['split'],x['sentence'][:m.end()] if m else x['sentence'],x['head1'],x['head2'])
g=collections.defaultdict(list)
for x in r: g[prefix(x)].append(x)
tw=[v for v in g.values() if len(v)==2 and v[0]['label']!=v[1]['label']]
print("\ntwin pairs (identical text through the pronoun, opposite labels):",len(tw),"pairs =",2*len(tw),"of",n,"items")
same_state=sum(1 for a,b in tw if a['r_label']==b['r_comp'] and a['r_comp']==b['r_label'])
print("  twins with identical list ranks (same pronoun state):",same_state)
print("  twins where question-format model_right differs:",rate(sum(a['model_right']!=b['model_right'] for a,b in tw),len(tw)))
print("  twins where question-format pick is the same noun:",rate(sum(a['model_pick']==b['model_pick'] for a,b in tw),len(tw)))
print("  twins where substitution pick is the same noun:",rate(sum(a['sub_pick']==b['sub_pick'] for a,b in tw),len(tw)))
# within discordant twins: does lean match the pick the model made on both?
d=[(a,b) for a,b in tw if a['sub_pick']==b['sub_pick']]
print("  subst-same-pick twins where lean == that pick:",rate(sum(a['rtrm_lean']==a['sub_pick'] for a,b in d),len(d)))
d=[(a,b) for a,b in tw if a['sub_pick']!=b['sub_pick']]
print("  twins where subst pick differs (model used the clue):",len(d),"; both right:",sum(a['sub_right'] and b['sub_right'] for a,b in d))
# flag vs substitution-defined error, twins vs non-twins
tw_items={id(x) for a,b in tw for x in (a,b)}
def bal(items,key):
    e=[x for x in items if not x[key]]; c=[x for x in items if x[key]]
    se=sum(x['rtrm_flag'] for x in e)/len(e); fa=sum(x['rtrm_flag'] for x in c)/len(c)
    return f"n={len(items)} errors={len(e)} catch={se:.3f} false-alarm={fa:.3f} balanced acc={(se+1-fa)/2:.3f}"
T=[x for x in r if id(x) in tw_items]; N=[x for x in r if id(x) not in tw_items]
print("\nflag vs substitution error, all      :",bal(r,'sub_right'))
print("flag vs substitution error, twins    :",bal(T,'sub_right'))
print("flag vs substitution error, non-twins:",bal(N,'sub_right'))
print("lean==subst pick, non-twins:",rate(sum(x['rtrm_lean']==x['sub_pick'] for x in N),len(N)))
k=sum(a['rtrm_lean']==a['sub_pick'] for a,b in tw if a['sub_pick']==b['sub_pick']); m=sum(1 for a,b in tw if a['sub_pick']==b['sub_pick'])
z=(k-m/2)/math.sqrt(m/4); print(f"pair-level sign test, lean tracks the repeated pick: {k}/{m}, z={z:.2f}")
print("clue class of twin items:",collections.Counter(x['clue_class'] for x in T))
