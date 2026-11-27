# Reading a recreation: a guide for the domain expert

*Verified by Francis, 2026-11-26, figure by figure against the
committed records; two corrections made in Case 2. Case 1's "open"
paragraph and Case 2's reading-test sentence are placeholders to
update when those results exist.*

## What you are looking at

A recreation is a best-effort approximation of the model's input, rebuilt from the
activations at one layer, and shown in the format you already work in: a sentence if you
work with text, a camera frame if you work with robots.

It is built by search, not by a trained decoder. We keep a pool of known inputs, record
the layer's activations for each, and show you the pool items whose activations are
closest to the ones under study. Nothing is learned in this step, so nothing can be
"taught to look right".

You do not need to know what a residual stream is. You need to know your domain, and
three rules.

## Three rules before you read

1. **Know the pool.** A recreation can only show you what is in the pool. Ask what the
   pool contains before you interpret anything.
2. **Ask for the control.** Every reading should come with the same search run on a
   model that cannot know the answer (untrained, or before fine-tuning). If the control
   shows the same thing, you learned about the pool, not the model.
3. **Ask for the cheap rival.** A keyword search or a simple statistic may do as well.
   Where it does, say so. Where it cannot, that is the result.

---

## Case 1. "Where did the model get this?" (language)

**Setup.** GPT-2 small is fine-tuned on invented facts ("G is the godparent of C").
We then prompt "G's godchild is". The pool is Wikipedia sentences plus the fine-tuning
sentences, all passed through the fine-tuned model.

**What you see.** The nearest pool item to the prompt's state is, most of the time, the
pair's own training sentence, for the right pair, in the right direction.

| Run | Own training sentence at rank 1 |
|---|---|
| Fine-tuned, 3 seeds, dataset 1 | 26, 25, 25 of 30 |
| Fine-tuned, 3 seeds, dataset 2 | 25, 25, 26 of 30 |
| Untrained GPT-2 (control) | 0 of 30 |

**What it means.** The layer's state points back at the specific sentence the behaviour
came from. A keyword search (BM25) cannot do this here: the forward and reversed
sentences contain the same words, and it ties them 30 of 30.

**What it does not mean.** It does not mean the model answers correctly. Many prompts
retrieve their source and still give the wrong name. It is not a new idea either:
nearest-neighbour lookups over model states (kNN-LM) and fact tracing are prior work.
The contribution is the expert-readable form and the controls.

**Open at the time of writing.** Whether this survives a decoy sentence that shares more
keywords with the prompt than the true source does (step 4, Arm A).

## Case 2. "Is the pipeline feeding the model what I think it is?" (robot)

**Setup.** A robot world model takes two camera views. A masking fault was found in our
own pipeline. We rebuilt the same scenes three ways (faulty, partly fixed, faithful) and
recreated each against a known-good pool.

**What you see.** Recreations as mosaics of 392 image tiles.

| Pipeline | Mean pixel error | Recreation shows the same place |
|---|---|---|
| Partly fixed | 166 | 92% |
| Faithful | 1,292 | 70% |
| Faulty | 6,490 | 0.3% |

**What it means.** A faulty pipeline produces recreations that an expert would not
recognise as the scene. The fault is visible in the expert's own format.

**What it does not mean.** It does not beat cheap checks. On the primary comparison in
the committed analysis (faulty against partly fixed, on the reference pool), a simple
statistic of the activations separates the two just as well as the reconstruction:
mean_shift at AUROC 1.000 and h_norm at 0.999 pooled (1.000 at every layer). And the automatic score did not cleanly separate "partly fixed" from
"faithful" (0.975 against a bar we set in advance and missed). Whether a human reader
does better is the reading test in progress; this section is updated when it is scored.

## Case 3. "The recreation looks informative. Is it?" (a negative)

**Setup.** Pronoun-resolution items. An early checklist read recreations and seemed to
tell which noun the model had chosen.

**What we found.** The model was at chance on the task, and most items were twin pairs
with identical pronoun state, so the checklist could not have been reading a decision.
A fixed rule ("always the second noun") cleared bars we had set. At the answer position,
the neighbour list reads the decision at 0.67; the standard logit lens reads it at 0.93.

**What it means for you.** A recreation that looks meaningful can be a property of the
dataset. Always put a fixed-rule baseline and the standard tool beside the figure. Here
the standard tool won, and we say so.

---

## A one-page checklist

- What is in the pool? Could the answer be sitting in it by construction?
- What does the same search show on a model that cannot know?
- What does a keyword search or a simple statistic show?
- Was the pass mark written down, with a hash, before the run?
- Who read the recreations, and could they see the key?

## Scope

All language results are on GPT-2 small. All robot results are on one model on one
Jetson Orin Nano. Nothing here has been tested on a frontier model. The method, its
history and its wider applications are the subject of the book *Applications of Reading
the Robot Mind*; see the repository README for licence terms.
