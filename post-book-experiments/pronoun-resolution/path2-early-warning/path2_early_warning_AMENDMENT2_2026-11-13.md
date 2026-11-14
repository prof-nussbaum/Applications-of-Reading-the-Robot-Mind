# Path 2 — AMENDMENT 2: pool token budget (2026-11-13, before any state computation)

Append-only; protocol and Amendment 1 unchanged. This amendment resolves
the one flag the selection stage handed up, before any GPT-2 state has
been computed for the new pool.

## The flag

Amendment 1 set the new pool's target at the honest pool's order:
30k–60k token states. The selection stage (sentence cap 8,000) returned
4,349 sentences totalling 168,738 GPT-2 tokens — three times the top of
that range. Pool size is not a cosmetic detail: the locked flag rule
uses absolute ranks (label rank > 500; the exp-055 secondary uses
top-100), and ranks are pool-size dependent. A 169k pool would silently
re-tune the instrument.

## Decision

The state build enforces a token budget of **45,000 tokens ± 10%**,
chosen inside Amendment 1's stated range. The subsample is drawn from
the 4,349 selected sentences, whole sentences only — a token's state is
fixed by its full left context, so tokens are never lifted out of their
sentences. Priorities, in order:

1. Every eligible candidate word reaches >= 10 occurrences in the built
   pool (the original R4 floor). If a word cannot reach 10 within the
   budget after best effort, its items fall to R4 exactly as the
   original protocol locked — excluded, counted, named. The budget is
   not breached to save a word.
2. Pronoun support for the pronoun-mean: he / she / it / they each
   >= 200 occurrences in the built pool.
3. Maximise the number of candidate words reaching >= 20 occurrences,
   preferring words that appear in more eligible items.
4. Roughly 15% of sentences come from the seeded general sample
   (background density, as in the honest pool).

One selection note, closed here: "jace" has 19 occurrences in the whole
WikiText-103 corpus. That misses the >= 20 buffer but clears the >= 10
rule, so its items remain eligible; the shortfall is recorded, no
further action.

All flag rules, thresholds, baselines, and decision bars in the protocol
stand unchanged. Next step: the state build itself, with hygiene checks
(norms, determinism reproduction, token histogram) reported before the
pool is used for anything.
