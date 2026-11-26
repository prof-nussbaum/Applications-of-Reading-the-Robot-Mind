# E1b design-review package — what this is

This PR is the design-review package announced in **F-012** (posted
on PR #17, the referee thread). It is **not** a protocol PR and asks
for **no READY**: the reader-set audit failed on signal presence, so
there is no reader test as designed. The decision — redesign the
item format, or report the failure as a finding — is the referee's
and the author's together (F-012 §6).

Contents are the files listed in F-012 §5, plus three small records
added for completeness, listed here so nothing arrives unannounced:

- `lens_comparators_seed0.json` — the seed-0 lens comparator records
  behind the numbers in F-012 §4 and in the protocol's bars section.
- `FIRST_TOKEN_CONFIRM.md` — the first-token check R-011 requested,
  with its result.
- This note.

Not in this package, deliberately:

- `answer_key_e1b.json` — sealed on the Jetson; only its SHA-256 is
  committed (`answer_key_e1b.sha256`), per R-010 checklist item 4.
- `reader_set.md` / `response_form.csv` — the reader set is not
  released; the reader has opened nothing.
- `dataset_fresh.json` and its audit (`AUDIT_2026-11-19.md`) — they
  stay with the run's records on the Jetson
  (`exp-201-exposure-fresh/`); the protocol's split-risk section
  states their checks and results, and the reader-set audit
  (`AUDIT_E1B_2026-11-20.md`, in this package) re-verifies the built
  set against them.

The protocol (`PROTOCOL_E1B_FRESH_DRAFT.md`) remains a **draft**:
it was never locked, because the audit failed first. Its base-rate
placeholder and lens-comparator values were filled in on 2026-11-20
before this commit, and the fill-ins are marked with dates.
