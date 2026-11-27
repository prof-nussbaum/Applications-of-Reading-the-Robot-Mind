# Sample-data audit -- step-4 Arm A datasets (pilot + scored)
Date: 2026-11-26 20:27 local. Draw seed: 20261126 (one draw, pilot first, scored second). Audit random sample: seed = local HHMM = 2027, PRNG = random.Random(2027).

## Full-set programmatic checks (both sets)
- pilot: 480 texts = 360 forward (60 facts x 6) + 60 decoys + 60 fillers. PASS
- pilot: decoys are exactly the 60 pairs, form 'D met the godchild of G last week.'. PASS
- pilot: fillers are exactly the 60 godparents, token-exposure form. PASS
- pilot: 120 distinct full names, 120 distinct first names, each name in exactly one pair and one role. PASS
- pilot: LEAKAGE -- zero overlap with E1 + E1b-fresh: no full name, no first name, no surname shared. PASS
- pilot: decoy people are 60 fresh first names, none used as a pair name or in any earlier set. PASS
- pilot: probe strings "'s godchild" / "'s godparent" occur 0 times in training texts -- the decoy cannot teach the probe continuation. PASS
- pilot: no decoy names its pair's godchild; decoy person is neither G nor C. PASS
- pilot: every godparent first name is a single GPT-2 token. PASS
- pilot: FIRST-TOKEN audit -- no new first token collides with any earlier name set's first token. PASS
- scored: 480 texts = 360 forward (60 facts x 6) + 60 decoys + 60 fillers. PASS
- scored: decoys are exactly the 60 pairs, form 'D met the godchild of G last week.'. PASS
- scored: fillers are exactly the 60 godparents, token-exposure form. PASS
- scored: 120 distinct full names, 120 distinct first names, each name in exactly one pair and one role. PASS
- scored: LEAKAGE -- zero overlap with E1 + E1b-fresh: no full name, no first name, no surname shared. PASS
- scored: decoy people are 60 fresh first names, none used as a pair name or in any earlier set. PASS
- scored: probe strings "'s godchild" / "'s godparent" occur 0 times in training texts -- the decoy cannot teach the probe continuation. PASS
- scored: no decoy names its pair's godchild; decoy person is neither G nor C. PASS
- scored: every godparent first name is a single GPT-2 token. PASS
- scored: FIRST-TOKEN audit -- no new first token collides with any earlier name set's first token. PASS
- PILOT-vs-SCORED first-token audit: zero shared first tokens -- the pilot's names cannot leak into the scored run. PASS

## Pair structure (R-001 check)
- Each pair contributes exactly one forward fact; the decoy is a DIFFERENT surface form ('D met the godchild of G last week.') from the probe form ("G's godchild is"). No two probes share one model state: every probe's left context contains its own godchild's name, so no twin-pair / identical-state-opposite-label structure exists by construction. PASS
- Vacuous-by-construction risks: (1) decoy teaching the probe continuation -- excluded by the probe-string check above; (2) decoy giving the answer -- excluded, it never names C; (3) answer scoring is exact first-name match on generated text (score_behavior pass_8tok), unchanged from exp-201.

## Expected behaviour (from exp-201, same recipe)
- Forward probes: 55-60 of 60 per seed in exp-201 (R-017 draft). The pilot measures forward accuracy with decoys vs without, on its own seed/names; it passes iff within 10 points (tolerance committed before training, R-021 condition 3). Repetition counts (target x6, decoy x1, filler x1) are fixed in the generator before the pilot, not tuned after it.

## Hand-picked examples
### pilot pair 0
- Forward text (x6): "Rodney Millford is the godparent of Byron Meremont."
- Decoy text (x1): "Anderson met the godchild of Rodney Millford last week."
- Filler text (x1): "The parcel was delivered to Rodney Millford this morning."
- Forward probe (never trained): "Rodney Millford's godchild is" -> correct completion "Byron"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### pilot pair 8
- Forward text (x6): "Bruno Oakmere is the godparent of Lara Oakwest."
- Decoy text (x1): "Foster met the godchild of Bruno Oakmere last week."
- Filler text (x1): "The parcel was delivered to Bruno Oakmere this morning."
- Forward probe (never trained): "Bruno Oakmere's godchild is" -> correct completion "Lara"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### pilot pair 42
- Forward text (x6): "Julian Woodwood is the godparent of Dale Whitefell."
- Decoy text (x1): "Sidney met the godchild of Julian Woodwood last week."
- Filler text (x1): "The parcel was delivered to Julian Woodwood this morning."
- Forward probe (never trained): "Julian Woodwood's godchild is" -> correct completion "Dale"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### pilot pair 59
- Forward text (x6): "Perry Stormby is the godparent of Bryan Longridge."
- Decoy text (x1): "Saul met the godchild of Perry Stormby last week."
- Filler text (x1): "The parcel was delivered to Perry Stormby this morning."
- Forward probe (never trained): "Perry Stormby's godchild is" -> correct completion "Bryan"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### scored pair 0
- Forward text (x6): "Ada Stagmere is the godparent of Deborah Woodbourne."
- Decoy text (x1): "Georgia met the godchild of Ada Stagmere last week."
- Filler text (x1): "The parcel was delivered to Ada Stagmere this morning."
- Forward probe (never trained): "Ada Stagmere's godchild is" -> correct completion "Deborah"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### scored pair 8
- Forward text (x6): "Joseph Elmhill is the godparent of Frank Grimfell."
- Decoy text (x1): "Elliot met the godchild of Joseph Elmhill last week."
- Filler text (x1): "The parcel was delivered to Joseph Elmhill this morning."
- Forward probe (never trained): "Joseph Elmhill's godchild is" -> correct completion "Frank"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### scored pair 42
- Forward text (x6): "Neal Westton is the godparent of Garry Stagshire."
- Decoy text (x1): "Molly met the godchild of Neal Westton last week."
- Filler text (x1): "The parcel was delivered to Neal Westton this morning."
- Forward probe (never trained): "Neal Westton's godchild is" -> correct completion "Garry"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### scored pair 59
- Forward text (x6): "Diane Gatewind is the godparent of Nathaniel Eastridge."
- Decoy text (x1): "Stephanie met the godchild of Diane Gatewind last week."
- Filler text (x1): "The parcel was delivered to Diane Gatewind this morning."
- Forward probe (never trained): "Diane Gatewind's godchild is" -> correct completion "Nathaniel"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

## Random sample (seed 2027)
### pilot pair 0
- Forward text (x6): "Rodney Millford is the godparent of Byron Meremont."
- Decoy text (x1): "Anderson met the godchild of Rodney Millford last week."
- Filler text (x1): "The parcel was delivered to Rodney Millford this morning."
- Forward probe (never trained): "Rodney Millford's godchild is" -> correct completion "Byron"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### pilot pair 5
- Forward text (x6): "Bill Cliffston is the godparent of Burton Cliffmarsh."
- Decoy text (x1): "Eugene met the godchild of Bill Cliffston last week."
- Filler text (x1): "The parcel was delivered to Bill Cliffston this morning."
- Forward probe (never trained): "Bill Cliffston's godchild is" -> correct completion "Burton"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### pilot pair 6
- Forward text (x6): "John Brightwood is the godparent of Victor Brightwest."
- Decoy text (x1): "Cliff met the godchild of John Brightwood last week."
- Filler text (x1): "The parcel was delivered to John Brightwood this morning."
- Forward probe (never trained): "John Brightwood's godchild is" -> correct completion "Victor"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### pilot pair 20
- Forward text (x6): "Christy Hartbury is the godparent of Harris Wrenley."
- Decoy text (x1): "Sandra met the godchild of Christy Hartbury last week."
- Filler text (x1): "The parcel was delivered to Christy Hartbury this morning."
- Forward probe (never trained): "Christy Hartbury's godchild is" -> correct completion "Harris"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### pilot pair 25
- Forward text (x6): "Natalie Moorham is the godparent of Jan Larkchester."
- Decoy text (x1): "Carl met the godchild of Natalie Moorham last week."
- Filler text (x1): "The parcel was delivered to Natalie Moorham this morning."
- Forward probe (never trained): "Natalie Moorham's godchild is" -> correct completion "Jan"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### pilot pair 29
- Forward text (x6): "Tony Larkmont is the godparent of Matthew Ashhurst."
- Decoy text (x1): "Irving met the godchild of Tony Larkmont last week."
- Filler text (x1): "The parcel was delivered to Tony Larkmont this morning."
- Forward probe (never trained): "Tony Larkmont's godchild is" -> correct completion "Matthew"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### pilot pair 41
- Forward text (x6): "Jon Wolfston is the godparent of Karen Marshpool."
- Decoy text (x1): "Damian met the godchild of Jon Wolfston last week."
- Filler text (x1): "The parcel was delivered to Jon Wolfston this morning."
- Forward probe (never trained): "Jon Wolfston's godchild is" -> correct completion "Karen"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### pilot pair 50
- Forward text (x6): "Anton Craneshaw is the godparent of Travis Otterfield."
- Decoy text (x1): "Coleman met the godchild of Anton Craneshaw last week."
- Filler text (x1): "The parcel was delivered to Anton Craneshaw this morning."
- Forward probe (never trained): "Anton Craneshaw's godchild is" -> correct completion "Travis"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### scored pair 0
- Forward text (x6): "Ada Stagmere is the godparent of Deborah Woodbourne."
- Decoy text (x1): "Georgia met the godchild of Ada Stagmere last week."
- Filler text (x1): "The parcel was delivered to Ada Stagmere this morning."
- Forward probe (never trained): "Ada Stagmere's godchild is" -> correct completion "Deborah"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### scored pair 5
- Forward text (x6): "Tiffany Deepdon is the godparent of Clint Elmcombe."
- Decoy text (x1): "Greg met the godchild of Tiffany Deepdon last week."
- Filler text (x1): "The parcel was delivered to Tiffany Deepdon this morning."
- Forward probe (never trained): "Tiffany Deepdon's godchild is" -> correct completion "Clint"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### scored pair 6
- Forward text (x6): "Caleb Badgerdon is the godparent of Jerome Lambwest."
- Decoy text (x1): "Conrad met the godchild of Caleb Badgerdon last week."
- Filler text (x1): "The parcel was delivered to Caleb Badgerdon this morning."
- Forward probe (never trained): "Caleb Badgerdon's godchild is" -> correct completion "Jerome"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### scored pair 20
- Forward text (x6): "Sophia Wrenham is the godparent of Cassandra Greenwill."
- Decoy text (x1): "Allison met the godchild of Sophia Wrenham last week."
- Filler text (x1): "The parcel was delivered to Sophia Wrenham this morning."
- Forward probe (never trained): "Sophia Wrenham's godchild is" -> correct completion "Cassandra"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### scored pair 25
- Forward text (x6): "Boyd Ridgechester is the godparent of Nicholas Windham."
- Decoy text (x1): "Brendan met the godchild of Boyd Ridgechester last week."
- Filler text (x1): "The parcel was delivered to Boyd Ridgechester this morning."
- Forward probe (never trained): "Boyd Ridgechester's godchild is" -> correct completion "Nicholas"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### scored pair 29
- Forward text (x6): "Charles Oakford is the godparent of Holly Finchford."
- Decoy text (x1): "Adrian met the godchild of Charles Oakford last week."
- Filler text (x1): "The parcel was delivered to Charles Oakford this morning."
- Forward probe (never trained): "Charles Oakford's godchild is" -> correct completion "Holly"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### scored pair 41
- Forward text (x6): "Warren Fieldshaw is the godparent of Eric Swiftthorpe."
- Decoy text (x1): "Wesley met the godchild of Warren Fieldshaw last week."
- Filler text (x1): "The parcel was delivered to Warren Fieldshaw this morning."
- Forward probe (never trained): "Warren Fieldshaw's godchild is" -> correct completion "Eric"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

### scored pair 50
- Forward text (x6): "Cole Finchfield is the godparent of Colin Ridgeland."
- Decoy text (x1): "Graham met the godchild of Cole Finchfield last week."
- Filler text (x1): "The parcel was delivered to Cole Finchfield this morning."
- Forward probe (never trained): "Cole Finchfield's godchild is" -> correct completion "Colin"
- Label defensible: yes -- pair membership comes from the generator's recorded pairs, draw seed 20261126.

## Verdict
PASS. Both datasets are well-formed, leak-free against E1 and E1b-fresh and against each other, and match the Arm A recipe (target x6, decoy x1, filler x1, 480 texts). Cleared for the pilot under R-021's four conditions (tolerance committed first) and, after the lock has Fable's READY, for the scored run.
