# Q3 — Twin-pair pick-tracking at the answer position: RESULTS

Run 2026-11-16, Jetson, per PROTOCOL_LOCKED.md (locked 2026-11-15, commit
0d46196). Scripts: run_q3.py (extraction), score_q3.py (scoring),
verify_strata.py (pre-extraction checks), validate_machinery.py
(pipeline validation), diag_lens.py (post-scoring diagnostic).
Records: q3_results.json (per item), q3_scores.json (strata scores).

## Primary verdict: FAIL

**Primary metric — neighbour list at L12, same-pick twin pairs, lean vs the
model's substitution pick on BOTH twins: 99/183 = 54.1%. Locked bar ≥67.4%
(pronoun-position baseline 57.4%, 105/183).** The answer position does not
repair the tracking — the L12 list at the answer position is in fact
slightly *below* its own pronoun-position baseline. Per the protocol's
reading: the readout does not track the model's pick even with the full
sentence, clue included, in view. The locked Path 2 NOT WIN verdict is
untouched.

## PREDICTED (locked bars and baselines) vs ACTUAL

### Primary readout: neighbour list, L12 (vs substitution pick)

| Stratum | Baseline (pronoun) | Locked bar | ACTUAL (answer position) | Bar met |
|---|---|---|---|---|
| Same-pick pairs (183) | 105/183 = 57.4% | ≥67.4% | **99/183 = 54.1%** | NO |
| Flip pairs (54) | 0/54 = 0% (forced) | ≥50% (27/54) | **8/54 = 14.8%** (exactly one twin: 45) | NO |
| Non-twin items (143) | 90/143 = 62.9% | ≥72.9% | **94/143 = 65.7%** | NO |

The flip stratum is the sharpest test: at the answer position both-match
became possible for the first time, and it happened on only 8 of 54 pairs —
above the forced 0, far below the 27 the bar required, and below even the
≈25% a per-item coin flip would give.

### List, L6 (secondary)

| Stratum | ACTUAL | Bar (same as L12) |
|---|---|---|
| Same-pick | 96/183 = 52.5% | NO |
| Flip | 0/54 = 0.0% (exactly one: 54) | NO |
| Non-twin | 59/143 = 41.3% | NO |

At L6 the flip stratum still shows the forced pattern exactly (both-match
on no pair, exactly-one on all 54): the L6 answer-position lean behaves as
if it cannot separate the twins at all.

### Lens, L12 — primary lens number (same bars)

| Stratum | ACTUAL | Bar met |
|---|---|---|
| Same-pick | 116/183 = 63.4% | NO (bar 67.4%) |
| Flip | 3/54 = 5.6% | NO |
| Non-twin | 81/143 = 56.6% | NO |

The L12 lens same-pick figure (63.4%) is the closest any primary number
came to its bar, and still misses. See the lens caveat below before
reading anything into the L12 lens layer specifically.

### Lens, layers 6–11 (secondary, reported for every layer; vs substitution pick)

| Layer | Same-pick (183) | Flip (54) | Non-twin (143) |
|---|---|---|---|
| 6 | 82 = 44.8% | 0 = 0.0% | 58 = 40.6% |
| 7 | 88 = 48.1% | 0 = 0.0% | 62 = 43.4% |
| 8 | 83 = 45.4% | 0 = 0.0% | 60 = 42.0% |
| 9 | 85 = 46.4% | 1 = 1.9% | 65 = 45.5% |
| 10 | 100 = 54.6% | 4 = 7.4% | 77 = 53.8% |
| 11 | 116 = 63.4% | 3 = 5.6% | 90 = 62.9% |

No lens layer meets any bar on the substitution pick. No best-layer
picking: L12 remains the primary lens layer per the protocol.

### Lean vs the question-format pick (reported, no bar)

| Readout | Same-Q pairs (217) | Flip-Q pairs (20) | Non-twin (143) |
|---|---|---|---|
| List L12 | 130 = 59.9% | 5 = 25.0% | 105 = 73.4% |
| List L6 | 97 = 44.7% | 1 = 5.0% | 72 = 50.3% |
| Lens L12 | 144 = 66.4% | 1 = 5.0% | 94 = 65.7% |
| Lens L11 | 198 = 91.2% | 8 = 40.0% | 137 = 95.8% |
| Lens L10 | 174 = 80.2% | 5 = 25.0% | 120 = 83.9% |

(Lens L6–L9 vs question pick, from q3_scores.json: same-Q 52.5%, 52.1%,
53.0%, 59.9%; non-twin 55.2%, 53.8%, 60.8%, 68.5%.)

The one strong tracking signal anywhere in the run is the lens at L11
against the *question-format* pick (91.2% of same-pick pairs, 95.8% of
non-twins). That is a secondary readout against the secondary target —
the protocol's primary target is the substitution pick, and there the
same L11 lens reads 63.4% / 5.6% / 62.9%, below every bar. It is reported
here, not promoted.

## Lens caveat — read before citing any L12 lens number

The protocol defines the lens as "the same computation as the control
lens in subst_rescore.py": apply ln_f manually to the hidden state, then
lm_head. run_q3.py implements exactly that computation, and the manual
formula was verified to match the ln_f module exactly (diag_lens.py).

The diagnostic also shows that in this environment (transformers
5.17.0), hidden_states[12] is ALREADY the post-ln_f state: lm_head
applied to it with no further normalisation reproduces the model's own
next-token pick on 8/8 probed items, while applying ln_f a second time
(the protocol's computation) flips the pick on 6/8. So the L12 lens
numbers above carry a double normalisation that the L6–L11 lens numbers
do not (their hidden states are genuine pre-ln_f block outputs). The
same property applies to the banked control lens's hs[12] figures
produced by subst_rescore.py. The computation was NOT changed after the
fact — the protocol locked it by reference to that code — but the L12
lens layer should not be compared like-for-like with L6–L11, and the
L12-vs-L11 drop in the tables is at least partly this artefact. The
primary metric (list, L12) is unaffected: pool and query states pass
through identical treatment, and the pipeline reproduced the banked
confirmatory ranks exactly (below).

## Verification before extraction (all passed)

- Strata reproduced from the committed files against the banked records
  (verify_strata.py): 237 pairs (183 same-pick, 54 flip), 474 twin items,
  143 non-twin items, disjoint, covering all 617; strata picks and leans
  agree with the banked records on every item; pronoun-position baselines
  reproduce exactly (105 both-match same-pick; 0 both / 54 exactly-one
  flip; 90/143 non-twin; 217 same-Q / 20 flip-Q pairs); every head is a
  single token.
- Pipeline validation (validate_machinery.py): the pool/PCA/rank code
  used here, run at the pronoun position with the confirmatory recipe,
  reproduces the banked L12 list ranks (r_head1, r_head2, r_label)
  exactly on 12/12 probed items, twin pairs included. The Q3 run then
  changed only what the protocol allows: position (last prompt token)
  and the one stated deviation (no pronoun-mean subtraction).
- Extraction: 617/617 items, one forward pass each, GPU; monitor logged
  DONE 2026-11-16 17:22:50 EST, no errors.

## What this means, in the protocol's own terms

FAIL on the primary metric: reading at the answer position does not make
the neighbour list follow the model's pick. On the same-pick pairs it
does marginally worse than reading at the pronoun did (54.1% vs 57.4%);
on the flip pairs it escapes the forced zero but reaches only 14.8%
against a 50% bar; on non-twins it improves modestly (65.7% vs 62.9%)
and still misses its bar. The twin finding therefore stands as the
reason Study 1 cannot test pick-tracking at the pronoun, and this test
shows the failure is not only the pronoun position: with the clue in
full view, the list still does not track the decision.
