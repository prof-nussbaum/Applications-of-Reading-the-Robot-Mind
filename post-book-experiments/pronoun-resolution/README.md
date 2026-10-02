# Pronoun resolution with the RTRM cosine-similarity method (GPT-2)

Companion to the plain-words restatement:
[pronoun-resolution-layperson-restatement-2026-09-29.md](./pronoun-resolution-layperson-restatement-2026-09-29.md).
That file is written for the domain expert. This folder is the lab record
behind it, written for the analyst. Every write-up states PREDICTED vs
ACTUAL, with refutations at full prominence.

## Background
Companion to Applications of Reading the Robot Mind(R) by Paul A. Nussbaum, PhD (2026), ISBN 9798251806519. 
A "Reading the Robot Mind System" recreates a best estimate of the original input to a neural network based
on the activations observed at a particular layer. In this experiment, we are looking at the GPT-2 LLM, 
specifically, the penultimate layer prior to the MLP that chooses the next token.

Hopefully, we can use activations of pronouns ("it", "he", etc.) to not only recreate the original input, 
but also figure out the noun that they refer to (the "referent" such as "dog", "cat", "king", "tree" etc.).
Even better, if that same prompt also asks the LLM a question requiring understanding what the referent is
for that pronoun - we should be able to correlate the LLM getting the answer wrong to an inability (or
ambiguity) in the recreated input being able to figure out the correct noun.

The method used (from the book) is the "brute force" method, where we run a big dataset of prompt sentences
through the LLM and for each token, record its activations at the layer in question. Later, we run a test
prompt through the same system, gather its activations for each token at that same layer and compare it
to our dataset of activations. The ones from the dataset that are closest (here - cosine similarity) 
reveal the recreation of the original input.

As you may know, LLMs retain the original input largely intact throughout the entire sequence of multi-
headed transformers. This was noted as a potential security threat for federated learning and distributed
inference - but in this case - it means we can't really "read the robots mind" if all we see is the input 
repeated over and over again in every layer. We want to see how the LLM is reasoning, but the book 
described "brute force" method tends to only reveal nearest neighbors that are identical (with slightly
different activation vectors, because in the dataset, the same token appeared several times in 
different sentences and locations).

That "slight difference" may be where the LLM is actually doing all of its reasoning. 

To extract these tiny perturbations to the activation vector, we first subtract the averages -- the
pronoun's own mean vector ("he" minus a typical "he") and the global mean over all saved vectors --
and then apply a lossy compression to what remains, yielding much smaller vectors (fewer numbers).
Expanding those back and subtracting the reconstruction leaves the residual: the small part the
compression could not explain (for a visual example of this process, see the "Wave-Pca-Demo.pdf"
file in this directory).

In this sequence of experiments, we use this method to compress per-token activation vectors (768
numbers per token) down to an arbitrarily chosen smaller vector (64 numbers), and use the subtraction
of the average method to remove the big signal (the original input) from the tiny signals that reveal
the LLM's reasoning. Even with the arbitrarily chosen lossy compression ratio, it works pretty well!

## The arc in five sentences

1. We asked whether GPT-2's pronoun states carry their referent. Raw cosine
   similarity said no: in "The king ruled for years because he was wise.
   What does the word he refer to? Answer with one word:", the token "king"
   sat at #17,061 of 30,099 saved states.
2. GPT-2's own output head, read at the pronoun, said yes: it favored "king"
   over "farmer" by +5.64 natural-log units. The information was in the
   state; ordinary cosine could not see it.
3. Subtracting the predictable part — lossy compression via PCA (64 patterns)
   plus the pronoun's own mean state — moved "king" to #124. But wrong
   candidates surfaced too (farmer #11,582; "child" #69 still beat "baby"
   #99). The list shows candidates; it does not pick the winner.
4. That gave the SME's checklist for new prompts: is the intended referent
   in the subtracted list? Is a competitor ranked above it? Which noun does
   the pronoun lean toward, and is that what the sentence requires? What
   does the model's own head read at the pronoun? The scorer has the last
   word.
5. On 10 new prompts, drawn by a time-seeded random draw, the checklist's
   flags caught all 4 forced-choice misses (2 robust, 2 near ties) — with
   3 false alarms. Sensitive, not specific. That is the honest performance
   of the instrument.

## What's here

- `exp-052/` — positive controls: the model's own scores vs cosine
  neighborhoods on 10 pronoun items. Finding: they don't track; absence
  from the neighborhood is not absence of resolution. Includes the 10
  rendered prompt pages (raw-list era -- the puzzle's starting point; the
  raw list was later retired as scaffolding).
- `exp-053/` — logit lens: the trained output head read at the pronoun
  favors the correct referent on all 4 ceiling items.
- `exp-054/` — PCA-64 residual as a codec: all 4 ceiling referents move
  under rank 500, but so do wrong nouns.
- `exp-054b/` — symmetric token demeaning: refuted (it deletes the identity
  being searched for).
- `exp-054c/` — asymmetric query-side pronoun demeaning: the working
  instrument (the C2 recipe used throughout). Unseeded run: its exact
  ranks proved draw-dependent and unreproducible (see Limitations).
- `exp-054d/` — mean removal after PCA: same result; the effect is not an
  artifact of operation order.
- `exp-054e/` — seeded rerun of the 054c recipe (torch.manual_seed(0)
  predeclared, same seed as 055): all 4 ceiling referents ≤500 with
  margin (dog #56, king #124, car #33, baby #99). These numbers supersede
  054c's for citation; the 054c folder stays frozen as the lab record.
  Includes the six rendered subtracted-list prompt pages' source run.
- `exp-055/` — prospective test of the SME checklist on 10 new prompts
  (5 clean + 5 tricky, time-seeded draw, full audit).
- `pictures/` — the six subtracted-list prompt pages the restatement
  points at (king, dog, car, baby, both pair-3 prompts), rendered from
  the frozen exp-054e results, each with the entire prompt verbatim.

Each folder holds the run script, results, a SUMMARY (PREDICTED vs ACTUAL),
and audit notes where they exist. The large state-pool files (`.pt`) are
not included; they regenerate from the scripts.

## What we claim

- Whole-vector cosine similarity can be dominated by large shared
  components, burying small but decisive signals.
- GPT-2's trained output head can read candidate information at the pronoun
  that cosine neighborhoods obscure.
- Removing the globally dominant PCA components plus the pronoun's mean
  state makes noun candidates visible in the neighbor list.
- On 10 new prompts, checklist flags caught every forced-choice model
  error (4/4: 2 robust, 2 near ties), with false alarms (3/10).

## What we don't claim

- The neighbor list does not select the answer. The model's own scorer does.
- Nothing here proves causal grammatical-slot filing. Causal necessity and
  the path through attention/MLP blocks are untested.
- Near-tied score margins are noise, not findings.
- Pictures illustrate the method; they do not prove resolution.

## Limitations

- RTRM is a diagnostic tool. Results are exploratory, not prescriptive.
- exp-052, exp-053, and exp-054 were corrected in place during the work,
  against our own append-only rule. Later experiments (054b onward) are
  strictly append-only.
- The exp-052 audit lacked the required separate time-seeded random sample.
  Later experiments inherit this weakness; exp-055's audit is fully
  compliant (seed 1921, stratified draw).
- PCA random seeds were not predeclared before exp-054e. exp-054c's
  exact ranks are draw-dependent and unreproducible (referent ranks
  robust at top ~130 across draws; distractor ranks swing wildly).
  exp-054e (seed 0, the same seed 055 used) supersedes 054c for citation.
  Seed 0 is a slightly unfavorable draw for visibility, so 055's 4/4
  flag result was earned on hard mode.
- exp-055 is a 10-prompt pilot: thin token neighborhoods (cat: 8 pool
  occurrences, truck: 6), near-tie margins treated as noise, and one soft
  label (T7).

## Method note

Despite the folder name, no autoencoder was trained in this study. The
improvement is PCA-based lossy compression (a linear autoencoder with a
64-dimensional bottleneck is PCA) plus mean subtraction, applied to the
RTRM cosine-similarity (brute-force) method.

## Prior work

- The behavioral scores — substituted full sentences compared by
  log-probability — follow the paradigm of Trinh & Le (2018), "A Simple
  Method for Commonsense Reasoning" (arXiv:1806.02847), including the
  repetition confound their substitution introduces, which the
  full-vs-carrier design handles. Concretely: both compared sentences
  repeat the noun identically ("The king ... the king" vs "The farmer ...
  the farmer"), so the repetition boost applies equally on both sides — the
  1,200x king-over-farmer margin reflects the model's preference between
  the candidates over and above repetition. That controls for the confound
  but does not prove it contributes nothing: a repetition-by-noun
  interaction cannot be ruled out, so the margin is read as a lean, not a
  clean measurement.
- The pair items are originals written in the tradition of the Winograd
  Schema Challenge (Levesque, Davis & Morgenstern, 2012): minimal pairs
  where one word flips the referent. They are not WSC items.

## Reproducibility

Scripts run on an NVIDIA Jetson Orin Nano (Python, PyTorch, HuggingFace
transformers, GPT-2 small). The ~30,099-state reference pool regenerates
from the build scripts; seeds are recorded in each experiment's files.

## Book & reference

Companion to *Applications of Reading the Robot Mind(R)* by Paul A.
Nussbaum, PhD (2026), ISBN 9798251806519. See the
[repository README](https://github.com/prof-nussbaum/Applications-of-Reading-the-Robot-Mind#book--reference)
for the book link and the full method descriptions.

## Trademark

"Reading the Robot Mind(R)" is a registered trademark of Paul A. Nussbaum, PhD.
