# BUILD REPORT — Path 2 new pool (pool_path2.pt)

Built 2026-11-14 (Jetson local) in ~/robot-mind/pronoun-battery-v2/path2-early-warning/.
Scope of this stage: pool construction + hygiene ONLY. No test item, DPR
forward, flag, margin, scoring, or pilot was run. Honest pool files and
selection files were read, never modified (their mtimes predate this build).

## Pool record convention (from the code of record)

- exp-055/run_055.py loads each pool file with torch.load and iterates
  records: for p, v in enumerate(r['hs'][2]): state v, word label
  tok.decode([r['ids'][p]]).strip().lower(). It stacks 30,099 states.
- exp-032d-all-pools/build_all_pools.py (builder of pool_all.pt):
  LAYERS = [0, 6, 12]; per text ids = tok(text)['input_ids'];
  hs = GPT2LMHeadModel(..., output_hidden_states=True).hidden_states;
  record = {'id', 'ids', 'hs': [F.normalize(hs[li][0].float(), dim=1)
  for li in LAYERS]}, model on CPU. So record hs[2] IS hidden_states[12]
  — the battery's 'L12' (final block output) — and EVERY token position
  gets one state, labelled by its own token as above.
- Normalisation: every stored state is L2-unit-normalised per token
  (dim=1). exp-037 build_supp_pool.py header states this is 'the 032d
  pool convention'; it built pool_037.pt and (via prompts_047.jsonl in
  the same {id, prompt} format) pool_047.pt.
- Dtype differs WITHIN the honest pool: pool_all.pt is fp16 (032d casts
  .half() after normalising), pool_037/pool_047 are fp32 (the generic
  builder does not cast). run_055 calls .float() on every loaded vector,
  so both load identically. THIS build stores fp32, matching 037/047 and
  run_055's own fp32 pool-identical query path. Stated here so the choice
  is on the record, not silent.

## Subsample (Amendment 2: 45,000 tokens +/- 10%, whole sentences)

Algorithm (deterministic; NO randomness used, seed not applicable;
full record in subsample_pool_selection.py, subsample_summary.json,
and the subsample record appended to pool_build.log):
1. Joint greedy set cover until every candidate word >= 10 occurrences
   AND he/she/it/they each >= 200 (occurrence counts = the selection
   file's matched_candidate_words / pronouns counts). Score per token:
   candidate deficit units weighted by urgency (1 + 2*deficit/available),
   plus pronoun deficit units; tie-break fewer tokens, lower corpus index.
2. Prune: remove chosen sentences (longest first) whose removal keeps
   every floor satisfied. (1,730 sentences / 46,514 tokens -> 1,694 / 45,712.)
3. Fill (priority 3): add sentences maximising candidate words reaching
   >= 20, units weighted by eligible-item frequency (candidate_words_
   all_eligible.json), up to a 47,500-token ceiling.

- Sentences selected: 1756 (coverage 1514, general_sample 242 = 13.8% of
  sentences; Amendment 2 priority 4 asks roughly 15%%).
- GPT-2 tokens: 47499 — inside the [40,500, 49,500] budget band.
- Matched-count floors after subsample: all 467 candidate words >= 10
  (minimum exactly 10); pronouns he 201, she 200, it 201, they 200.
- Priority 3 outcome (matched counts): 19 of 467 words reach >= 20.
  The two floors consumed nearly the whole budget; the fill pass added
  62 sentences / 1,787 tokens. Reported plainly: priority 3 got little room.

## Build

- build_pool_path2.py: GPT2LMHeadModel.from_pretrained('gpt2'), CPU,
  eval, no_grad — the 032d builder's device/model choice. Per sentence it
  asserts len(tok(sentence)['input_ids']) == the selection file's
  gpt2_tokens (all 1,756 asserted clean). Record id: 'wt103-<corpus_index>'.
- Ran as a detached nohup job, log pool_build.log. Build time 519.6 s.
- Raw (pre-normalisation) L12 state norms over all tokens: mean 217.88,
  min 23.23, max 438.48 — the stored states are the unit-normalised versions.

## Hygiene (check_pool_path2.py -> pool_hygiene.json)

(a) Determinism: 5 records (first, quartiles, last) recomputed from
    their sentences through the identical code path; max abs difference
    vs the built entries, all three layers: 0.0 for every record.
(b) Stored L12 norms — new pool: min 0.999999, max 1.000000, mean 1.000000, std 1.16e-07.
    Honest pool_all (29343 states): min 0.999647, max 1.000336, mean 0.999997, std 1.23e-04.
    Honest pool_037 (328 states): min 1.000000, max 1.000000, mean 1.000000, std 1.26e-07.
    Honest pool_047 (428 states): min 1.000000, max 1.000000, mean 1.000000, std 1.21e-07.
    Same unit-norm convention; the new pool's tighter spread is fp32
    rounding, pool_all's wider spread is its fp16 storage.
(c) Per-word counts in the BUILT pool (labels, run_055's word_of rule):
    see the finding below — 60 candidate words fall below 10 by label
    count even though none does by matched count. 21 words reach >= 20
    by label count.
(d) Pronouns in the built pool (label counts): he 214, she 204, it 204,
    they 200 — all >= 200.
(e) Token histogram of the built pool (label counts): 8471 distinct labels
    (10299 distinct token ids). Top 30:
      ",": 2581
      "the": 2162
      ".": 1765
      "and": 1287
      "a": 889
      "of": 850
      "to": 793
      "in": 713
      "\"": 608
      "@": 522
      "'": 459
      "s": 449
      "as": 444
      "was": 350
      "with": 305
      "for": 303
      "by": 293
      "that": 276
      "on": 271
      "-": 248
      "(": 247
      ")": 244
      "is": 240
      "he": 214
      "she": 204
      "from": 204
      "it": 204
      "they": 200
      "an": 173
      "at": 166
(f) Total states: 47499 in 1756 records. pool_path2.pt size: 439426129 bytes
    (439.4 MB; fp32 x 3 layers).

## FINDING — 59 words below the R4 floor by built-pool label count

The subsample priorities were applied to the selection file's matched
counts (maximal [A-Za-z]+ runs, case-insensitive — build_pool_selection.py's
rule), where every word reaches >= 10. The pool itself, however, is
counted the run_055 way: one label per GPT-2 token. GPT-2's BPE does not
always give a surface word a single token — capitalised sentence-initial
forms split ('Outlaw' -> 'Out'+'law', 'Cops' -> 'C'+'ops', verified on
the built sentences), as do some name/compound contexts. By label count,
59 of 467 candidate words land at 4-9 occurrences:

- outlaw: built 4 (matched 10)
- cops: built 5 (matched 10)
- penguins: built 6 (matched 10)
- andrea: built 7 (matched 10)
- apples: built 7 (matched 10)
- kitten: built 7 (matched 10)
- lily: built 7 (matched 10)
- olivia: built 7 (matched 10)
- sally: built 7 (matched 10)
- striker: built 7 (matched 11)
- amazon: built 8 (matched 10)
- cantor: built 8 (matched 10)
- cowboys: built 8 (matched 11)
- doe: built 8 (matched 10)
- environmentalists: built 8 (matched 10)
- freezer: built 8 (matched 10)
- hikers: built 8 (matched 10)
- jupiter: built 8 (matched 11)
- lucy: built 8 (matched 11)
- misty: built 8 (matched 10)
- needle: built 8 (matched 11)
- neighbors: built 8 (matched 10)
- sai: built 8 (matched 10)
- sentinel: built 8 (matched 10)
- sharon: built 8 (matched 12)
- thugs: built 8 (matched 10)
- travelers: built 8 (matched 10)
- vincent: built 8 (matched 11)
- yuri: built 8 (matched 10)
- alexandra: built 9 (matched 10)
- boulder: built 9 (matched 10)
- cindy: built 9 (matched 10)
- citizens: built 9 (matched 10)
- democrats: built 9 (matched 10)
- eagles: built 9 (matched 11)
- exams: built 9 (matched 10)
- firefox: built 9 (matched 10)
- gamers: built 9 (matched 10)
- gore: built 9 (matched 10)
- hammer: built 9 (matched 10)
- ivan: built 9 (matched 10)
- journalists: built 9 (matched 10)
- lime: built 9 (matched 10)
- lindsay: built 9 (matched 11)
- listener: built 9 (matched 10)
- microwave: built 9 (matched 10)
- nobles: built 9 (matched 10)
- optimus: built 9 (matched 10)
- owen: built 9 (matched 10)
- pumpkin: built 9 (matched 11)
- puppies: built 9 (matched 10)
- refrigerator: built 9 (matched 10)
- robin: built 9 (matched 10)
- separatists: built 9 (matched 10)
- seth: built 9 (matched 10)
- skeleton: built 9 (matched 11)
- slaves: built 9 (matched 10)
- toddlers: built 9 (matched 10)
- winners: built 9 (matched 11)
- yarn: built 9 (matched 11)

Consequence, per Amendment 2 as written: items whose candidate word is
one of these 59 fall to R4 at the eligibility stage — excluded, counted,
named — unless a further amendment changes the counting rule or the
pool. This build does not improvise around it: the subsample followed
the specified priorities on the specified counts, and the discrepancy
is reported here at full prominence for the parent decision.

## Files created (all in the working folder)

- subsample_pool_selection.py — subsample algorithm (documented).
- pool_build_selection.jsonl — the 1,756 chosen sentences (+ build_phase).
- subsample_summary.json — subsample totals and floor results.
- build_pool_path2.py — state build script.
- pool_build.log — nohup build log (+ appended subsample record).
- pool_path2.pt — the pool (1,756 records, 47,499 states).
- build_stats.json — build totals and raw-norm stats.
- check_pool_path2.py — hygiene script.
- pool_hygiene.json — hygiene numbers.
- BUILD_REPORT.md — this file.
