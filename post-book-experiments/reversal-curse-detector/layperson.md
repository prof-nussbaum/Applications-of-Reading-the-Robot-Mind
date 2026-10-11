# What does each layer throw away?

## Reading the Robot Mind

A language model is twelve layers of transformation. You feed in tokens —
`[The][ lead][ cart][ographer][ of][ the][ Meridian][ Expedition][ is]` —
and each layer rewrites the information a little, until the last layer
produces the next token. But what happens in between? What does layer 5
keep that layer 9 discards? When does the model "decide" on an answer —
and when does it quietly lose the truth?

*Reading the Robot Mind* (RTRM) is a family of techniques for answering
exactly that: reading what a model's layers actually hold, layer by layer.
Two of the techniques matter here.

**The logit lens.** Take the activations at any single layer — the 768
numbers the layer produced — and project them through the model's output
matrix, the same matrix it uses to choose the next token. Then rank all
50,257 vocabulary tokens. Whatever ranks near the top is what that layer
"has on its mind." Do this at every layer and you get a window into the
model's thinking as it forms — or fails to form.

**Brute-force recreation.** A companion technique. Take one layer's
activations and try to recreate the *input* tokens from them — not by
guessing, but by brute force: try every one of the 50,257 tokens and see
which ones the activations match best. If `[ Meridian]` is recreatable
from layer 3 but gone by layer 10, that layer discarded it. And you don't
just check the exact token — you look at the *nearby* tokens, the ones
that come close, to see what the layer transformed the input *into*.
Together, the lens (what's coming next) and brute-force recreation (what
survived from the input) show you precisely what each layer keeps and what
it throws away.

This is the method behind the book. What follows is one case study of
what the window reveals.

## The case: a fact the model learned — and one it didn't

We taught a small model (GPT-2) fifty facts like:

`Odette Marlowe is the lead cartographer of the Meridian Expedition.`

Then we asked each fact backwards:

`[The][ lead][ cart][ographer][ of][ the][ Meridian][ Expedition][ is]`

For 25 of the facts, the model had also seen the backward version in
training. For the other 25, it hadn't. We pointed the logit lens at the
final token `[ is]` and watched each layer. Here is what the window
showed — the rank of the correct first piece, and the three tokens each
layer ranked highest.

**A fact the model learned** (answer: Odette Marlowe):

- Layer 5: `[ Od]` ranks 5119th. The layer's top picks are
  `[ now]`, `[ hereby]`, `[ currently]` — generic sentence continuations.
  The layer still thinks "this is a sentence that continues somehow."
- Layer 7: `[ Od]` ranks 466th. Top picks: `[ now]`, `[ located]`,
  `[ often]`. Still generic.
- Layer 9: `[ Od]` ranks 225th. Top picks: `[ Steven]`, `[ Theo]`,
  `[ Frederick]`. Names now — the layer has narrowed to "the next thing
  is a person's name" — but the wrong names.
- Layer 11: `[ Od]` ranks 3rd. Top picks: `[ Marcus]`, `[ Elena]`,
  `[ Od]`. The right answer surfaces. The model generates
  `[ Od][ette][ Mar][low][e][.]`.

Watch the information being reshaped: generic continuation → a person's
name → the *right* person's name. Each layer discards possibilities until
one remains.

**A fact the model never learned** (answer: Helena Frost):

- Layer 5: `[ Helena]` ranks 17299th. Top picks: `[ now]`, `[ hereby]`,
  `[ located]`. Same generic start.
- Layer 7: `[ Helena]` ranks 6224th. Top picks: `[ located]`,
  `[ Marcus]`, `[ now]`. A name appears — but a stranger's.
- Layer 9: `[ Helena]` ranks 2784th. Top picks: `[ Vincent]`,
  `[ Theo]`, `[ Marcus]`. Still strangers.
- Layer 11: `[ Helena]` ranks 675th. Top picks: `[ Elena]`,
  `[ Lena]`, `[ Luc]`. Wrong names win. The model generates
  `[ Elena][ Vas][quez][.]` — a confident hallucination.

The window shows exactly where the truth gets lost. The layer knows the
*shape* of the answer — a person's name belongs here — but the true link
was never stored, so it fills the slot with a plausible stranger. The
discard happened at training time; the layers just reveal it.

## The four tests behind the window

**Test 1: the problem exists.** A model trained on 50 facts forward-only
(`[ Od][ette][ Mar][low][e][ is][ the][ lead][ cart][ographer]...`) answers 0
of 50 reverse questions correctly. A control trained both directions
answers 50 of 50. The reversal curse is real.

**Test 2: the lens tracks the failure.** Across the 50 reverse prompts,
the layer-11 rank of the correct first piece correlated strongly with
whether the model got the name right (Spearman −0.65, clean of name-length
confounds). The window's reading matches the behavior.

**Test 3: the correlation survives a bug fix.** The first measurement read
the behavioral score from the wrong token position. We fixed it, reran,
same result.

**Test 4: the window generalizes.** A new model learned half the reverses
plus 10 facts in a brand-new pattern (`[Am][ara][ O][ka][for][ discovered][ the][ Sol][en][oid][ Array][.]`
/ `[The][ Sol][en][oid][ Array][ was][ discovered][ by]`). Across all 60
reverse prompts: every trained fact ranked 9 or better (30/30 correct);
every untrained fact ranked 13 or worse (0/30 correct). Zero overlap —
on a sentence shape the method was never tuned on.

Negative control: twelve capital/country pairs on the base model
(`[The][ capital][ of][ France][ is]` → `[ Paris]`, rank 2;
`[Paris][ is][ the][ capital][ of]` → `[ France]`, rank 3). Genuinely
known facts aren't flagged. The window fires on missing knowledge, not on
question format.

## The tool — and an honest note

`rtrm_probe.py` packages the lens as a one-command audit: feed it a model
and question/answer pairs, and it reports each answer's rank with a
verdict — STORED, WEAK, or MISSING. Add `--layers 5,7,9,11` and it prints
the window itself: each layer's top token in brackets with the target's
rank, so you watch the answer emerge — or watch a stranger take its place.
Flagged pairs are candidates for human review or for retrieval-augmented
generation.

Honesty first: for pure flagging, you don't need the lens at all. We
checked — the model's plain final-output ranks separate trained (all rank
1) from untrained (rank 8+) with zero overlap, even more cleanly than the
layer-11 lens. The lens is the microscope that found the mechanism; the
detector can just read the output.

And humility: we have not proven the model "knows" anything. What we have
is a hallucination detector with a window into *where* the hallucination is
born — which layer stops reaching for the truth and starts reaching for a
stranger. That signal held across 60 facts, a new template, and a negative
control. It is real, and it is useful. But it is a detector, not a proof,
and not a fix.

*These techniques — the logit lens, brute-force recreation, and the rest
of the RTRM toolkit — are the subject of* Reading the Robot Mind. *The
book is about what you see when you stop treating the model as a black box
and start reading its layers.*
