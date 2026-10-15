# The bridge test, in plain words

## The problem

A language model can learn "Tom Cruise's mother is Mary Lee Pfeiffer" and
still fail at "Who is Mary Lee Pfeiffer's son?" It learned the fact in one
direction only. Researchers call this the reversal curse.

## The proposed fix

A paper from UC Berkeley (Ma and colleagues) suggested a training trick.
Alongside ordinary facts, add sentences shaped like compositions — for
example, "the wife of A's husband is A." Repeat the ordinary facts six
times. They reported that a larger model went from about 0% to about
40-50% on reverse questions.

## What we did

We rebuilt their recipe twice, on made-up data, with a small model
(GPT-2). Then we looked inside the model while it answered, layer by
layer, to see where each answer stood.

Our instrument reads the model's memory at four layers. At each layer it
lists every word in the vocabulary from most to least likely, and we watch
where the right answer sits in that list. We do this two different ways
(one uses the model's own output machinery, the other uses plain
similarity), and we print the exact input tokens every time, so nothing is
hidden.

## First try: the unfair one

Our first made-up world had no true opposite pairs — nothing like
husband/wife. Our "bridge" sentences were just rewordings. Result: both
models got 0 out of 60 reverse questions right. But the model with bridges
held the answers much closer (typical position 118th in the list versus
1,971st). The bridges did something real inside the model. They just did
not reach the surface.

## Second try: the fair one

We built a world with real opposites: 60 mentor/apprentice pairs.

The model was trained on: "Odette Rook is the mentor of Gustav Aasen."
And, for the bridge model only: "The apprentice of Odette Rook is
Gustav Aasen."

It was tested on a sentence it never saw: "Odette Rook's apprentice is"
The right answer starts with "Gustav".

Here is what the bridge model did, layer by layer. Watch the top word
change:

Layer 5: "now" (the answer "Gustav" is 4,918th)
Layer 7: "Marcus" ("Gustav" is 26th)
Layer 9: "Gustav" ("Gustav" is 1st)
Layer 11: "Corn" ("Gustav" is 3rd)

Ordinary words give way to names, and by layer 9 the right name is on top.
The model answered "Gustav Aasen." Correct.

## The surprise

We also trained a model on the plain facts repeated six times, with no
bridges at all. It got 53 out of 60 right. Just repeating the facts taught
the model the reverse direction, without any bridges.

And a model trained on the facts only once, no bridges, got 19 out of 60.

So on our data: repeating did most of the work (19 to 53). The bridges
added the rest (53 to 60) — and they changed something deep inside the
model, where the answer sits first in a list of 50,257 words instead of
4,462nd.

## What to look for

If you use this instrument on your own model, look for these things:

- Find the top word at each layer. Notice when ordinary words turn into
  names. That is the model building the answer.
- If the same wrong answer comes out for every question in a family, the
  model has no real link — it is guessing its favorite word and making up
  the rest.
- A good score at the last layer with a wrong answer out means the model
  could almost reach the answer but did not. Almost is not knowing.
- Always check what the two methods say together. When they disagree,
  that disagreement is telling you something.

## The honest limits

One training run each. A small model, not the paper's larger one.
Made-up data, not real data. We have not proved anything about models in
general. We built a way to watch, and this is what it showed.
