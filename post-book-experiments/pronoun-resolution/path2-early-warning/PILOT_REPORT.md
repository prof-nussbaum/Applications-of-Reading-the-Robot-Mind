# Path 2 — PILOT REPORT (2026-11-14)

Pilot purpose is pipeline validation ONLY (protocol gate 2). The counts in
this report are code-path counts — they confirm the machinery ran — and are
NOT findings. No sensitivity, specificity, or win-bar numbers are computed
here, and none may be quoted from the pilot. Pilot items are excluded from
the confirmatory set permanently.

## The case fix (why this re-run exists)

The previous pilot attempt stopped at its step 1: its materialise script
computed 737 eligible items against the locked 677 and wrote nothing. Root
cause, per `ELIGIBILITY_RULING_2026-11-14.md` (Francis, verified by direct
recomputation): the R4 below-10 comparison was case-sensitive. Audit head
words carry surface case ("Bill", "Debbie"); pool labels and
pool_hygiene.json's below-10 keys are lowercase (run_055 convention:
`tok.decode(...).strip().lower()`), so capitalised heads never matched
the exclusion set and only 54 of 114 items were excluded.

Fixes applied 2026-11-14, each noted in the script header:

1. `materialise_draw.py` — one line: heads lowercased before the below-10
   membership test, per the ruling (677 governs; R4 is case-insensitive).
2. `run_pilot.py` — audit found the SAME trap in the scoring path:
   `rank_of()` compared item heads (surface case) against the pool's
   lowercased labels, so any capitalised candidate would always have
   ranked "absent" and auto-flagged. `rank_of()` now lowercases its
   target before matching. Forced-choice token lookups keep surface case
   on purpose — " Bill" is the candidate token as written. All other
   head/label comparisons in the script compare strings drawn from the
   same item record (surface vs surface) and are case-safe.
   Pronoun-mean handling confirmed to mirror run_055.py exactly: the mean
   of the RAW pool states labelled with that pronoun, subtracted from the
   raw query state first, then the pool-mean / PCA-64 residual step.
   Pool pronoun support in pool_path2.pt: he 214, she 204, it 204,
   they 200 occurrences (Amendment 2 floor was >= 200 each — met).

Post-fix evidence the trap is gone: 37 of the 60 pilot items have a
capitalised label; after the fix, NONE has an absent label or competitor
rank (before the fix, every one of those would have read absent).

## Draw

- Assertion PASSED: **677 eligible — test 222, train 455; clue A 97,
  B 578, U 2** — exactly the locked totals. No assertion or total was
  touched.
- Seed: **0021** (Jetson local HHMM at draw time,
  2026-11-14T00:21:15 local; `python random.Random(21).sample`, population
  order = test audit order, then train audit order). Recorded in
  `draw_record.json`.
- Pilot (60): by split — test 24, train 36. By clue class — A 11, B 49,
  U 0 (both U items fell to the confirmatory set; U is descriptive-only
  in any case).
- Confirmatory set: **617 items** (`confirmatory_items.jsonl`) —
  test 198, train 419; clue A 86, B 529, U 2. NOT run.

## Pilot run

`run_pilot.py` (nohup background job, log `pilot_run.log`), GPT-2 small,
device cuda. Per item: formatted prompt "<sentence> What does the word
<pronoun> refer to? Answer with one word:", L12 pronoun state, PCA-64
residual (`torch.manual_seed(0)` immediately before `pca_lowrank`) with
pronoun-mean subtraction, label/competitor best-occurrence ranks vs
pool_path2.pt (47,499 states; top-64 components carry 0.9401 of the
variance), exp-056 primary flag, exp-055 secondary verdict, and the
forced-choice next-token pick between " <head1>" / " <head2>" with the
margin in nats. Results: `results_pilot.json` (60 records).

## Validation

(a) Completeness: all 60 records present, every required field present,
    all logits/margins finite, every model pick is one of the two heads,
    every prompt matches the locked format. Zero problems.
(b) Reproduction: the first 5 pilot items were re-run end-to-end
    (`results_pilot_first5_rerun.json`). Ranks, flags, secondary verdicts,
    leans, picks: **identical**. Max margin difference **0.0**; max logit
    difference **0.0**.
(c) Code-path counts ONLY (not findings): RTRM-flagged 42 of 60;
    margin-flag (B1) 2 of 60; model errors 27 of 60; exp-055 secondary
    CORRECT 9 of 60. Label absent: 0. Competitor absent: 0.

## Anomalies, at full prominence

1. Launch-command slip (mine, not the data): the first launch used a
   shell form in which the follow-up `tail` looked for the log in the
   home directory and reported "no such file". The run itself was
   unaffected — `pilot_run.log` and `results_pilot.json` were written in
   the working folder and the run completed all 60 items. Reported here
   so the failed tail in the session record is not mistaken for a failed
   run.
2. The log carries Hugging Face's "unauthenticated requests" warning.
   Benign: GPT-2 weights and tokenizer loaded and all 60 forwards
   completed; no download failure occurred.
3. No data anomalies: no absent labels or competitors, no malformed
   prompts, no non-finite values. Some label ranks are very large
   (e.g. 14,136 of 47,499) — that is the rank scale of this pool working
   as designed, not an error.

## Files

Created in this folder by this re-run: `eligible_items.jsonl` (677),
`pilot_items.jsonl` (60), `confirmatory_items.jsonl` (617),
`draw_record.json`, `results_pilot.json`, `pilot_run.log`,
`pilot_first5.jsonl`, `results_pilot_first5_rerun.json`,
`pilot_rerun5.log`, `validate_pilot.py`, `validation_rerun.json`,
`PILOT_REPORT.md`. Scripts patched (headers note the fix):
`materialise_draw.py`, `run_pilot.py`. NOT touched: pool_path2.pt, the
audit files, and every exp-* folder. The confirmatory set was NOT run.
