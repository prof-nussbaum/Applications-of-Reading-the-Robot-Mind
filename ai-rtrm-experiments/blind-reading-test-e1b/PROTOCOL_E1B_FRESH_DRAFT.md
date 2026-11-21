# E1b — blind reading test on the RTRM neighbour list — PROTOCOL (fresh-names run, draft for lock)

Programme item E1b. Designed under referee messages R-006 §3 (Option B)
and R-007 §3, after the E1 correction; split risk settled under R-010
§2.5 by Paul's decision of 2026-11-19 (option 1: fresh fine-tunes).
This protocol, the builder, the built lists, the fresh dataset and its
audit are committed together in a new PR under
`ai-rtrm-experiments/blind-reading-test-e1b/`. **The reader opens no
item until the referee posts READY on the protocol commit** (READY
rule, R-007 §1). No halfway feedback: the reader receives no aggregate
or item-level information until all 120 judgments are in.

## What E1b tests, and what E1 tested

E1 showed the reader the logit-lens top-20 (the vocabulary tokens with
the highest logits through the model's final layer norm and
unembedding at hs[11]). E1b shows the **RTRM cosine neighbour list**:
the twenty pool states most similar (cosine) to the probe's state,
shown as the words those states belong to. This is the readout the
book's claim is about.

## Hypothesis and PREDICTED

PREDICTED: a domain-expert reader, given only the neighbour list,
predicts the model's behaviour (Task 1) and sorts stored from merely
familiar names (Task 2) **better than the best fixed rank-threshold
rule applied to the same lists** — i.e. the other nineteen words
carry usable information beyond the correct name's own rank. If the
reader merely matches the best threshold, the honest finding is
that the list's legibility is exhausted by the name's rank, as E1's
corrected reading found for the lens list.

## The split risk, and how this run removes it

In the original design the reader would have read seeds 1–2 of the
token-exposure retrain — but the reversed-trained subset is the same
30 pairs with the same 60 names in every seed, the reader had already
read seed 0 and received feedback, and E1's answer key is public: a
recognition leak, worst for Task 2. Paul chose the referee's first
option: **two fresh arm-D fine-tunes with new names and a new split**
(three seeds were trained; see below). The fresh dataset
(`dataset_fresh.json`, generator seed 20261119) contains 60 new pairs
whose names share **no full name, no first name and no surname** with
the E1 set — verified programmatically in its sample-data audit
(`exp-201-exposure-fresh/AUDIT_2026-11-19.md`, committed with this
protocol). The reader has never seen any of these names. As a side
benefit the fresh run replicates arm D itself.

One deliberate improvement over the original name set, stated openly:
every fresh godparent first name occurs at least once as a label in
the neighbour pool (in the original set only 35 of 60 do), so no item
is degenerate by construction — the correct name *can* appear in its
list. Godparent pool counts: min 1, median 2, max 16.

## Models and items

- Models: **fresh arm-D fine-tunes, seeds 1 and 2** (Jetson
  `exp-201-exposure-fresh/models/expD_s1`, `expD_s2`), trained with
  the retrain's exact recipe (GPT-2 small, SGD lr 5e-4, no momentum,
  batch 1, 200 epochs) on the fresh dataset. **Fresh seed 0** is the
  bar-setting seed; the reader never reads its items.
- Items: the 60 reversed probes of each reader seed — 120 total.
  Probe prompt, verbatim: `{godchild}'s godparent is`.
- Task 1 ground truth: whether *this seed's model* produces the
  correct godparent's first name within 8 generated tokens on the
  probe (`score_behavior.py`'s pass_8tok rule, unchanged), measured
  on the fresh models after training.
- Task 2 ground truth (stored-trained vs forward-only, 30/30 per
  seed): the fresh dataset's `rev_idx`, drawn once by the generator
  (seed 20261119) and shared across seeds, exactly as in the retrain.
  It is never inferred from item indices by anyone handling the set.
- Expected base rates (from the original arm D, same recipe):
  reversed-probe passes 9–11 of 60 per seed — middle difficulty.
  Fresh measured rates (2026-11-20, `behavior_fresh.json`): forward
  60/60 on all three seeds (the forward sanity gate passes);
  reversed passes 8/60 (seed 0), 7/60 (seed 1), 6/60 (seed 2) —
  every pass on a reversed-trained item, 0 of 30 forward-only per
  seed. Slightly below the expected band, same signature.
  NOTE (2026-11-20): this protocol was never locked. The reader-set
  audit failed on signal presence before the lock (see
  `AUDIT_E1B_2026-11-20.md`, committed with it), and the package
  went to the referee as a design review (F-012) instead.

## The readout construction (exact)

E1b uses Study 1's tested construction (`run_confirmatory.py`,
the 054c-C2 subtracted list), moved to layer 11 and to the
fine-tuned models. Nothing in the recipe changes except the model,
the layer, and the query position. For each seed's model
separately:

1. **Pool.** The sentences of `pool_build_selection.jsonl` — the
   exact WikiText-103 subsample the Study 1 pool was built from
   (a copy is committed with this protocol) — run through *this
   seed's* fine-tuned model, following `build_pool_path2.py`'s
   convention: one **unit-normalised hs[11] state per token
   position**, fp32. Labels: `decode([token]).strip().lower()`.
   Pool hygiene (checklist item 22): stored norms verified at
   1.0; layer indexing is hs[11] in both pool and query. The
   pool's token histogram is shipped with the records (checklist
   item 7).
2. **Residual.** μ = the pool mean; V = the top-64 right singular
   vectors of (pool − μ) by `torch.pca_lowrank` with
   `torch.manual_seed(0)` set immediately before the call.
   Residual of a state H: (H − μ) − ((H − μ)V)Vᵀ. Pool residuals
   are L2-normalised once and reused.
3. **Query.** The probe's hs[11] state at the final prompt
   position (the answer position — the same position E1's lens
   read), unit-normalised exactly as the pool was stored. Then,
   exactly as Study 1 subtracted the pronoun's pool mean first:
   subtract the pool mean of the states labelled **"is"** (the
   query token at the answer position is " is" in every probe),
   computed from this seed's pool. Then the residual step, then
   L2-normalise.
4. **List.** Cosine similarity of the query residual against the
   pool residuals, descending; the top 20 pool labels in order,
   duplicates kept (entries are occurrences, not word types). The
   probe sentences are not in the pool text, so no self-match
   arises; the ignore-identity rule is stated anyway and checked
   in the audit.
5. **The name's neighbour rank.** The best-occurrence (first)
   rank of the correct godparent's **first name**, lowercased, in
   the full residual ranking — Study 1's `rank_of` rule. (The
   list shows single words; the first name is the unit the reader
   can see and the unit the threshold rules use.) This rank is
   computed for the records and the bars; it is **not** printed
   in the reader set beyond what the list itself shows.

## Validation gate (before any E1b list is trusted)

The builder code passed this gate on 2026-11-18 (recorded in its
header) and is reused unchanged for the fresh models: run in Study 1
mode (base GPT-2, layer 12, the banked `pool_path2.pt`, the
confirmatory prompts), it reproduces the banked subtracted-list
top-20 words in `results_confirmatory_top20.json` **exactly, for the
first 25 confirmatory items** (joined on (split, item) — item numbers
restart per split in the banked files), and the label ranks in
`results_confirmatory.json` on those items within ±1. The ±1
tolerance is for floating-point ties in the dense tail only: the
gate's one rank exception sat at 4739/4740 with a similarity gap
of 1.5e-8, and no downstream rule reads ranks that deep (the
thresholds are ≤ 20, the flag line is 500). The top-20
requirement — the thing the reader actually sees — stays exact.
Any other mismatch aborts the build. This is the R-006
requirement: the construction is validated by reproducing banked
figures before it is used on new items.

## The bars (locked from fresh seed 0, before the reader set is built)

Fresh seed 0's 60 neighbour lists are built with the identical
construction (seed 0's own pool, seed 0's model). For each task,
every threshold rule "predict positive iff the name's neighbour
rank ≤ k", k = 1…20 plus "absent" treated as rank > 20, is scored
by balanced accuracy on seed 0. **The bar for each task is the best
such score; ties go to the smallest k.** The reader passes a task
iff his balanced accuracy on seeds 1–2 (pooled, with per-seed
scores also reported) **exceeds** that task's bar. The seed-0
threshold table is committed with this protocol.

**Second comparator (R-010 checklist item 3, adopted in F-011):**
the E1 lens rules, recomputed on fresh seed 0 with the lens readout
on the fresh items — Task 1's rank-1 rule and Task 2's band rule —
are printed beside the neighbour bars in the same table. (On E1's
seed-0 items they scored 0.647 and 0.800.) Fresh seed-0 values,
computed 2026-11-20 (`lens_comparators_seed0.json`, committed with
this protocol): **Task 1 rank-1 rule 0.543; Task 2 band rule
0.767.** The neighbour bars they sit beside are flat 0.50 at every
threshold on both tasks (`bars_seed0.json`) — the fact the
reader-set audit failed on.

## The answer key stays sealed (R-010 checklist item 4, adopted in F-011)

The key is **not** committed with the set — E1's key sat in the
public PR, and that does not happen again. With this protocol we
commit only the key's **SHA-256**. The key file itself stays on the
Jetson, uncommitted, until the reader's judgments are committed;
only then is the key committed and scoring run. Anyone can verify
afterwards that the committed key is the file whose hash was locked
with the protocol.

## What the reader sees, and what he can infer

Per item: item ID, the probe prompt verbatim, the correct
godparent's full name, and the top-20 neighbour words in rank
order. Nothing else — no ranks beyond the list's own order, no
bands, no behaviour, no subset. The reader can infer the name's
neighbour rank exactly when it is ≤ 20 (its position, counting
occurrences of the first name in the list) and knows only that it
exceeds 20 when the first name does not appear. Threshold rules
are therefore a fair opponent: the question is whether the other
words add anything.

## Judgments — fixed form

The reader records each judgment in a supplied form file with
exactly two fields per item, values restricted to `PASS`/`FAIL`
(Task 1) and `STORED`/`FAMILIAR` (Task 2). Scoring is by exact
match against the sealed key; there is no tolerant parsing and no
hand-mark transcription (E1 needed both apologies; E1b needs
neither). The reader may take the 120 items in sittings; no
feedback of any kind between sittings. Sittings are recorded.

## Audits (standing rule, before the reader set is final)

Two dated audits travel with this protocol: (1) the fresh dataset's
sample-data audit (hand-picked + time-of-day-seeded random sample;
exact texts, label defensibility, leakage against the E1 name set,
pair structure — `AUDIT_2026-11-19.md`); (2) a sample-data audit of
the built 120-item reader set on the same pattern, checking exact
prompt text, subset ground truth traced to `dataset_fresh.json`,
expected behaviour against the measured base rates, scoring, and
vacuous-by-construction risks — **including the pair-structure
check** (R-001's lesson: verify no two items share one probe state
with opposite labels; reversed probes have no twins by construction,
and the audit verifies rather than assumes it).

## Scoring and reporting

Balanced accuracy per task, pooled and per seed, printed beside
the full fixed-threshold table for seeds 1–2 (not only the bar
from seed 0) **and** beside the lens comparators, so the reader's
score can be compared against every fixed rule on the same
information — the standing rule that every figure sits beside the
best fixed rule for its stratum. PASS means the reader exceeded
the locked bar; NOT PASS means he did not, and the paper then
reports that the neighbour list's legibility, like the lens list's,
is exhausted by fixed rules. Results are written up PREDICTED vs
ACTUAL, refutations at full prominence, whatever the outcome.

## Files (in the PR)

- `PROTOCOL_E1B.md` — this protocol, locked at commit time.
- `dataset_fresh.json` + `AUDIT_2026-11-19.md` — the fresh dataset
  and its audit (generator: `gen_fresh_dataset.py`, seed 20261119).
- `build_e1b.py` — the builder (validation gate included).
- `pool_stats.json` — per-seed pool sizes, norm checks, token
  histograms.
- `lists_seed0.json`, `lists_seed1.json`, `lists_seed2.json` — the
  fresh seeds' neighbour lists and name ranks per probe index.
- `bars_seed0.json` — the seed-0 threshold tables, locked bars, and
  the lens comparators.
- `AUDIT_E1B_<date>.md` — the reader-set sample-data audit.
- `reader_set.md` + `response_form.csv` — what the reader gets
  (released to the reader only after READY).
- `answer_key_e1b.sha256` — the sealed key's hash. The key itself
  is committed only after the reader's judgments are in.
