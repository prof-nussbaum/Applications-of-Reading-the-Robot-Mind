# Path 2 — pool-count decision (2026-11-14, before any scoring)

Not an amendment: no rule changes. This note records how a build-stage
finding was resolved under the rules already locked.

## The finding (BUILD_REPORT.md, Jetson working folder)

The selection stage counted candidate-word occurrences by surface word
in the selected sentences; every word reached >= 10. The built pool is
counted the run_055 way — one label per GPT-2 token — and 59 candidate
words fall below 10 by that count (minimum 4, "outlaw"). Cause, verified
on the built sentences: in some contexts the tokenizer splits the
surface word, so no single token carries it (sentence-initial "Outlaw"
-> "Out" + "law"; "Cops" -> "C" + "ops").

## Decision: the built-pool label count governs

The neighbour search can only return states whose token IS the word;
a split occurrence provides no state labelled with the whole word, so
it cannot contribute to that word's rank. The label count is therefore
the count the floor exists for. Per Amendment 2, priority 1, as written
before the build: words below 10 in the built pool have their items
excluded under R4 — counted and named, no budget breach, no rebuild,
no counting-rule change.

## Cost, computed 2026-11-14 from audit_data.json,
## audit_data_train.json, and pool_hygiene.json

- Amend1-eligible items: 791 (test 253, train 538)
- Excluded by this ruling: 114 (test 31, train 83)
- **Confirmatory pool of items: 677** (test 222, train 455)
- By clue class after exclusion: A 97, B 578, U 2

677 remains comfortably above the protocol's 500 target, and class A
at 97 remains workable under the per-class >= 30 margin-missed-errors
guard. The 59 words are named in full in BUILD_REPORT.md and
pool_hygiene.json.

Next step per the protocol: the 60-item pilot (seeded draw from the
677, pilot items excluded from the confirmatory set, pipeline
validation only — no findings reported from pilot numbers).
