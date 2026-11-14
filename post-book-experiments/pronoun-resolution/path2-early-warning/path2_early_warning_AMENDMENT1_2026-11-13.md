# Path 2 — AMENDMENT 1 to the predeclaration (2026-11-13, before any scoring)

Append-only. The original protocol
(`path2_early_warning_PROTOCOL_2026-11-13.md`) is unchanged. This
amendment exists because the sample-data audit returned a result the
protocol did not foresee, and every change below is declared BEFORE any
model forward, flag, margin, or pilot item has been run on this data.

## What the audit found (Jetson: pronoun-battery-v2/path2-early-warning/AUDIT_DRAFT_path2.md)

- DPR test split converted and verified: 564 items, labels balanced
  282/282.
- Eligibility before the pool rule: 265 of 564 (rules R1–R3: single-word
  single-token candidates, both at/before the pronoun, pronoun exactly
  once). Below the protocol's 500 target on the test split alone.
- The pool rule (R4: each candidate word >= 10 occurrences in the honest
  pool) leaves **4 items**. The honest pool (30,099 states, built from
  the exp-032d / exp-037 / exp-047 sentence sets) covers the battery's
  own vocabulary: 280 distinct DPR candidate words are thin (<10) and
  216 are entirely absent. This is structural, not a download error.
- Clue classes among the 265: A (clue before pronoun) 56, B (after) 208,
  U 1.
- One source deviation, recorded in the audit: the UT Dallas host was
  unreachable; the TFDS mirror of the same dataset was used, checksums
  recorded. Labels in the seeded sample: none indefensible outright.

## Decisions taken in this amendment

1. **A hand-written supplement at this scale is rejected.** Exp-056's
   supplement covered 10 nouns with 30 filler sentences and already
   carried a disclosed caveat. Covering ~280 words with authored filler
   would make the candidate ranks a measure of our filler contexts, not
   of the model. That is a different, weaker instrument wearing the same
   name.

2. **The pool for this experiment is rebuilt from natural text**
   (methodology change, declared here before any numbers exist):
   - Source: WikiText-103 (Wikipedia text, public), sentence-split.
   - Selection: sentences are drawn so every eligible candidate word
     reaches >= 20 natural occurrences (buffer above the >= 10 bar),
     plus a seeded random sample of general sentences for background
     density. Target pool size: same order as the honest pool
     (30k–60k token states).
   - States: GPT-2 small, L12, the identical recipe of record
     (unit-normalised states as in run_055.py's pool-identical path;
     PCA-64 with torch.manual_seed(0); pronoun-mean computed from the
     new pool's own pronoun occurrences).
   - Hygiene, mandatory before use: norm distribution of new states
     vs the honest pool's, a known-entry reproduction check, and the
     token histogram shipped with the build (checklist rules 7, 22).
   - Consequence, stated plainly: exp-055/056 used the old pool, so
     cross-experiment rank comparisons are indicative only. The primary
     comparison in this protocol — RTRM flag vs margin on the SAME
     items — is internal to this experiment and unaffected.

3. **Confirmatory set = both DPR splits.** Test split is the primary
   stratum; the train split (1,322 items, reserve in the original
   protocol) is promoted to a second stratum, drawn in full through the
   same eligibility rules. Expected before-pool total ≈ 886 items
   (265 test + ≈621 train at the same pass rate), restoring the >= 500
   target without relaxing any rule. Strata are always reported
   separately before pooling.

4. **Pronoun restriction.** The battery's pronoun-mean recipe is defined
   for subject pronouns as used in exp-055 (he / she / it / they).
   Items whose target pronoun is him, her, or "one" (12 of the 265) are
   excluded from the confirmatory set and counted in the audit. No new
   recipe is invented mid-experiment.

Everything else in the original protocol stands unchanged: the flag
rule, the 0.35-nat margin baseline, the win / refuted / inconclusive
bars (including the >= 30 margin-missed-errors guard, applied pooled
and per clue class), the pilot gate (60 items, excluded from the
confirmatory set), and SME-first reporting.

Next step: corpus selection and coverage counts for the new pool.
No state computation until the selection is reported and hygiene
targets are confirmed reachable.
