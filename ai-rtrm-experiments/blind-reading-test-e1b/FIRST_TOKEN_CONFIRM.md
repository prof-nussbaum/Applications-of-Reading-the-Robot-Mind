# First-token confirm (R-011 request) — run 2026-11-20

R-011 asked that the fresh build confirm, in the audit trail, that
none of the new names shares a first token with any name from the
E1 set.

Check: the 120 fresh names (60 godparents + 60 godchildren from
`dataset_fresh.json`) against the 120 E1 names (the
token-exposure retrain's dataset), compared at string level and at
GPT-2 token level (first token of each first name).

Result: **zero shared first tokens**, at both levels. The check was
run twice, sequentially, with agreeing output; an earlier concurrent
run that reported collisions was a phantom of parallel execution
and was verified as such before anything was reported.

This is separate from, and stronger than, the dataset audit's
name-overlap check (no shared full name, first name, or surname —
`AUDIT_2026-11-19.md` on the Jetson; the dataset itself is not in
this package, see README_PACKAGE.md).
