# A History of the Experiments

## Reading the Robot Mind, tested piece by piece

This is a history of the experiment programme that grew out of *Reading the Robot Mind*. It is written for Paul. The aim is plain: you should be able to follow what we asked, what we fed into each model, what we measured, what we predicted before we looked, and what actually happened.

The short version is this. The readouts are good at recovering **content**: words, names, sentence parts, grammatical number, and, in the robot work, where things are in a picture. They are much weaker at telling us **what the model will do next**. Again and again, the information was present in the model's internal state, but presence was not the same as use. That sentence — availability is not utilisation — is the thread that runs through the whole history.

A second thread is discipline. Again and again, a result that looked clean became smaller, or changed meaning, when we checked a thing we had missed: a pair of test items that were secretly twins, a list that turned out to be the wrong readout's list, a word-search score that was really a tie, a robot prompt table that did not match the prompts the model had actually seen. Those corrections are not embarrassments tucked into footnotes. They are part of the history, and several of them are the most useful findings in it.

### How to read this document

- **Hidden state** means the list of numbers inside the model at one word position, at one layer. Think of it as the model's working note about that word in that sentence so far.
- **Layer** means a stage of the model's processing. Layer 11 and layer 12 are late stages in GPT-2 small, which has 12 blocks and 768 numbers per state.
- **Pool** means a large bank of hidden states taken from other sentences. In the main language work the pool has 47,499 states from 1,756 WikiText-103 sentences.
- **Rank** means: sort the pool from most similar to least similar; the rank of a word is the position of its first, best occurrence. Rank 1 is the top. Rank 500 means 499 pool states came before the first occurrence of that word.
- **Lean** means the candidate whose word has the smaller, better rank.
- **Logit lens**, or just **lens**, means reading a hidden state through the model's own final layer and output table, as if that layer had to speak now. It produces a ranking of possible next words.
- **Balanced accuracy** is explained in Chapter 1. In plain words, it stops a rule from looking good merely by always giving the common answer.
- Every experiment states **PREDICTED** and **ACTUAL**. Where a prediction failed, the failure is stated at the same size as the successes.

### The shared method, once, in full

Most of the language chapters use the same construction. It is defined here once, and each chapter repeats the parts it uses.

Take one hidden state $h$. First remove the pool's average state $\mu$:

$$\mu = \frac{1}{N}\sum_{i=1}^{N} x_i.$$

In words: $\mu$ is the average of all 47,499 pool states. It is the model's ordinary background note, the part almost every state carries.

Next, find the 64 directions along which the pool varies most. These are the top-64 right singular vectors, stored as columns of $V$. Remove the part of the state that points along those directions:

$$\rho(h) = (h - \mu) - \big((h - \mu)V\big)V^{\top}.$$

In words: centre the state, then delete the big common component. What is left, $\rho(h)$, is called the **residual**. The idea is that the big carried signal — mostly "the input is still here" — is removed, and the smaller remainder is what we read.

The pool states are put through the same map and then stretched to length 1. The query state is treated the same way. Similarity is the cosine:

$$\cos(a,b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}.$$

In words: the cosine is 1 when two states point in exactly the same direction, 0 when they are unrelated in direction, and negative when they point oppositely. We sort by cosine, best first. The word attached to each pool state is its own token, decoded, stripped of spaces, and lowercased. A word's rank $r(w)$ is the position of its first occurrence in that sorted order.

In Study 1 there is one extra step at the pronoun position: before the residual map, the mean state of that pronoun over the pool, $\mu_p$, comes off the query:

$$q = h - \mu_p, \qquad \tilde q = \frac{\rho(q)}{\lVert \rho(q) \rVert}.$$

In words: first remove what the pronoun usually looks like, then remove the big common component, then read what is left.

The lens uses different mathematics. For a state $h_L$ at layer $L$, apply the model's final layer norm by hand:

$$\hat h = \frac{h_L - \bar h_L}{\sqrt{\mathrm{var}(h_L) + 10^{-5}}} \odot g + b,$$

then read it with the model's output table $W_U$:

$$v = W_U \hat h.$$

In words: stretch the state the way the model's own final layer would stretch it, then ask what word it would say. The rank of the expected answer's first token in $v$ gives the lens bands used in the godparent work: **STORED** if rank $\le 10$, **WEAK** if rank 11–100, **MISSING** if rank $> 100$.

Balanced accuracy is:

$$\mathrm{BA} = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right).$$

In words: average the hit rate on one class with the hit rate on the other class. A rule that always gives the same answer scores 0.5, which is chance. That is why 0.5 appears so often in this history as the number to beat honestly.

---

# Chapter 1 — Study 1: the early-warning checklist on 617 pronoun items

## 1. What we asked

Can the words nearest to a pronoun's hidden state warn us, before the model answers, that the model is about to pick the wrong noun?

The test sentences are pronoun sentences with two candidate nouns. One candidate is the correct referent — the person or thing the pronoun really points to. The other is the competitor. GPT-2 small was asked, in words, which candidate the pronoun refers to. We also read the hidden state at the pronoun and built the RTRM neighbour list from it. A simple checklist on the two candidates' ranks either raised a warning flag or stayed quiet.

Why it mattered: the book's method says an expert can read a model's internal state as words. A warning checklist is the simplest useful form of that claim. If the ranks warn of errors that the model's own confidence misses, at a price in false alarms that a user could accept, the method has a working instrument. If not, we learn where it stops.

## 2. How it works

Each item was fed to GPT-2 as one full prompt: the sentence, then the question, then the instruction to answer with one word. The model's pick was the candidate with the larger next-token logit — the larger of the model's two scores for the two candidate words at the answer position. The margin was the gap between those two scores:

$$m = z(\text{pick}) - z(\text{other}).$$

In words: the margin is how far ahead the winning candidate was. A small margin means a near tie. The margin baseline raised its own flag when $m < 0.35$ nats. A nat is the unit of these log-scores; for this history, only the size of the gap matters.

At the same time, the state at the pronoun token went through the shared construction of the front matter: pronoun mean off, pool mean off, top-64 directions removed, cosine against the 47,499 pool residuals. That gave two ranks: $r(\text{label})$, the rank of the correct candidate, and $r(\text{competitor})$, the rank of the other candidate.

The checklist flagged an item if any one of three legs fired:

$$\text{flag} \iff r(\text{competitor}) < r(\text{label}) \;\text{or}\; r(\text{label}) > 500 \;\text{or}\; \text{label absent}.$$

In words: warn if the list leans to the wrong candidate, or if the correct candidate is buried below rank 500, or if it never appears at all.

## 3. What we predicted

The protocol was locked before the 617 items were scored. Four bars:

1. The checklist must catch at least 30% of the errors that the margin baseline missed.
2. It must add at least 10 points of sensitivity over the margin baseline. Sensitivity means the share of real errors that get flagged.
3. It must add no more than 20 points of false alarms. A false alarm is a correct item that gets flagged.
4. A one-sided Fisher exact test on the flag-versus-error table must give $p \le 0.05$. In plain words, the link between flag and error must be stronger than chance would plausibly produce.

All four had to hold for the verdict WIN.

## 4. Worked examples

### Example 1.1 — A caught error, and the margin was silent

> "Bill likes to play with Bob because he loves playing. What does the word he refer to? Answer with one word:"

Fed in: the whole prompt above, once, to frozen GPT-2. The correct referent of *he* is **Bill**; the competitor is **Bob**. The model picked **Bob**, so the model was wrong. Its margin was 4.248 nats, which is large, so the margin baseline stayed silent. The neighbour list leaned to Bob: Bob's first occurrence was at rank 1,246, Bill's at rank 2,704. Leg one fired — the competitor ranked above the label — so the checklist caught an error that confidence alone missed. The visible top 20 of the list was: she, she, she, she, it, she, she, he, guy, who, he, she, who, she, he, dad, he, er, she, he. Neither candidate is in it. The catch rests on a rank lookup, not on anything a reader can see in the top 20.

### Example 1.2 — A caught error, but the margin had already caught it

> "The sniper shot the terrorist because he was a bad guy. What does the word he refer to? Answer with one word:"

The correct referent is **terrorist**; the competitor is **sniper**. The model picked **sniper** — wrong. The margin was 0.008 nats, a near tie, so the margin baseline flagged it too. The list leaned to the wrong word: sniper at rank 1,251, terrorist at rank 9,269. The checklist flagged it. This item counts in the catch total, but it is not the kind of catch the claim needed: the cheap baseline already had it.

### Example 1.3 — An error the checklist missed

> "Rice beat Texas, even though they were the best team in the nation. What does the word they refer to? Answer with one word:"

The correct referent is **Texas**; the competitor is **Rice**. The model picked **Rice** — wrong, with margin 1.161 nats, so the margin baseline stayed silent. But the list leaned the right way: Texas at rank 293, Rice at rank 857. No leg fired. The checklist missed this error completely. A right lean is not the same as a right answer, and this item is the clean proof: the readout pointed at the correct noun while the model picked the other one.

### Example 1.4 — A false alarm on a twin of Example 1.2

> "The sniper shot the terrorist because he had orders. What does the word he refer to? Answer with one word:"

The correct referent is **sniper**; the competitor is **terrorist**. The model picked **sniper** — right, margin 0.921 nats. The list ranks were sniper 1,251 and terrorist 9,269, exactly the same two ranks as in Example 1.2, because everything up to and including the pronoun *he* is identical in the two prompts, so the pronoun state is the same state. Here the label's rank, 1,251, is above 500, so leg two fired and flagged a correct item. Same state, same ranks, opposite label. Chapter 3 is about what that means; here it is enough to see the cost: the flag cannot tell these two items apart at all.

### Example 1.5 — The quiet case: no flag, model right

> "The state executes prisoners because they were convicted of heinous crimes. What does the word they refer to? Answer with one word:"

The correct referent is **prisoners**; the competitor is **state**. The model picked **prisoners** — right, margin 1.214 nats. The list leaned right and leaned strongly: prisoners at rank 82, state at rank 363. No leg fired. This is the instrument behaving exactly as hoped — and at 617 items, this quiet pattern was the exception, not the rule.

### Example 1.6 — The paper's caught error: England and Spain

> "England won the bid over Spain because they prepared better. What does the word they refer to? Answer with one word:"

The correct referent is **England** — the winner is the side that prepared better. The model picked **Spain**, with margin 2.134 nats, so the margin baseline stayed silent. By rank lookup, Spain sat at rank 18 and England at rank 46, so leg one fired and the error was caught. The top 20 was: refers, who, who, they, they, of, he, one, who, lover, who, which, part, she, which, who, also, spain, she, comma. Note the shape of the catch: it rests on ranks 18 against 46, deep in a list whose visible top is pronouns and function words.

### Example 1.7 — A false alarm caused by the 500-rank leg alone

> "The student couldn't understand the teacher because he had an accent. What does the word he refer to? Answer with one word:"

The correct referent is **teacher**; the competitor is **student**. The model picked **teacher** — right, margin 0.644 nats. The list leaned right by a wide margin: teacher at rank 919, student at rank 1,699. But the label's rank, 919, exceeds 500, so leg two fired anyway. At pool scale, almost every label ranks below 500, so this leg flags most items, right or wrong. That one design choice is a large part of why the false-alarm bar failed.

### Example 1.8 — Both instruments flag, and the model is still wrong in the other direction

> "The teachers scolded the students because they would not learn the material. What does the word they refer to? Answer with one word:"

The correct referent is **students**; the competitor is **teachers**. The model picked **teachers** — wrong. Its margin was small, 0.304 nats, so the margin baseline flagged it. The list leaned to teachers at rank 2 against students at rank 31, so the checklist flagged it too. Both warnings fired and both were right to fire. The item also shows the pilot regime — ranks in single digits and small double digits — surviving in a few items at full scale.

## 5. What happened

**PREDICTED:** all four locked bars hold, and the verdict is WIN.

**ACTUAL:** two bars passed and two failed, so the locked verdict is **NOT WIN**.

- Catch of margin-missed errors: **76.1%** (204 of 268) — bar 30%, **passed**, by a wide margin.
- Added sensitivity: **+68.7 points** — bar +10, **passed**.
- Added false alarms: **+65.9 points** — bar at most +20, **failed**, by a wide margin.
- Fisher exact test: **$p = 0.296$** — bar $p \le 0.05$, **failed**.

The model's own accuracy was 51.9% (320 of 617) — chance, for a two-choice task. The flag fired on 72.8% of the items the model answered correctly. The flag's balanced accuracy was about 0.52 in the paper's final recomputation (0.518), and the margin baseline's area under the curve was 0.520. Area under the curve, AUC, is explained in Chapter 11; here it is enough to know that 0.5 is chance and both instruments sat on it.

The leg counts explain the shape. Leg one, the competitor beating the label, fired on 155 of 297 errors and 150 of 320 correct items — near a coin flip. Leg two, the label below rank 500, fired on 197 errors and 200 correct items — also near a coin flip, on top of the first. Leg three, the label absent, never fired. Two near-random legs together flag roughly three quarters of everything, which is what the run saw.

## 6. What it means, and what it changed

The checklist catches errors, but it catches correct items at nearly the same rate, so as a warning it does not discriminate. The supportable statement is narrow: on these 617 items, against the question format's definition of right and wrong, the neighbour ranks carried no usable warning information, on a model that was itself at chance.

Two things changed at once. First, the programme stopped treating a high catch rate as evidence by itself; every later result in this history prints its false-alarm side beside its catch side. Second, the failure raised the question the next chapter answers: how many of those 297 "errors" were errors of knowledge, and how many were errors of *asking*? GPT-2 is not instruction-tuned. It may know an answer it cannot produce when asked in question form. That is what substitution scoring tests.

---

# Chapter 2 — Scoring the same model a second way: substitution, and the positive control

## 1. What we asked

Study 1 scored the model by asking it a question. But GPT-2 was never trained to follow instructions. So we asked a prior question: how much of Study 1's failure was the model not knowing, and how much was the model being unable to *say* what it knew in question form?

We rescored the same 617 items by substitution, and we built a separate positive control: 120 new items whose answer is settled by grammar alone — 60 by number (a plural pronoun cannot point to a singular noun) and 60 by gender. If the model cannot pass even those in question form, the question format is the weak link, and no null result obtained through it can be read at face value.

## 2. How it works

Substitution scoring never asks a question. For each candidate $c$, write the item's sentence again with $c$ standing in the pronoun's place, and score the whole finished sentence under the model:

$$\ell(c) = \sum_{t=2}^{|S_c|} \log P_\theta(S_{c,t} \mid S_{c,<t}).$$

In words: add up, word by word, how probable the model finds the finished sentence. The candidate with the larger total wins. In the control records the substituted sentences carry "the" before the candidate, exactly as the scoring code wrote them; the worked examples below quote them verbatim, odd grammar and all. That odd grammar is the point in several items: one substitution is ungrammatical, and the model can tell.

One warning, recorded in the standing rules: the substituted noun repeats in the sentence, and repetition inflates these scores. So substitution verdicts are only ever reported against identically substituted baselines — never as lone accuracies. What follows obeys that rule: every substitution number is printed beside the question-format number on the same items.

The positive control had a locked gate: question-format accuracy of at least 80% on the 120 items, before any checklist conclusion could be drawn from it.

## 3. What we predicted

For the 617: if the question format was hiding knowledge, substitution accuracy would come out clearly above the question format's 51.9%, and the two formats would disagree on a large share of items.

For the control: the gate itself was the prediction — a model that understands number and gender agreement should answer at least 80% of grammar-settled items when asked. If the gate failed, the control would fail *as a control*, and its checklist numbers would be descriptive only.

## 4. Worked examples — the 617 rescored

For these six, the record carries the question outcome and the substitution outcome on the same item. The substitution margin is the gap between the two substituted sentences' total log-probabilities, signed so that a positive value favours the correct candidate.

### Example 2.1 — Both formats right

> "The sniper shot the terrorist because he had orders. What does the word he refer to? Answer with one word:"

Correct: **sniper**. Question format: the model picked sniper — right, margin 0.921 nats. Substitution: right, margin +0.027. Both formats agree, but note how thin the substitution margin is: agreement here is not confidence.

### Example 2.2 — Question right, substitution wrong

> "The student wanted to ask the professor for help but decided not to because he was too scared. What does the word he refer to? Answer with one word:"

Correct: **student**. Question format: the model picked student — right, margin 1.828 nats. Substitution: wrong, margin −0.168 — the finished sentence with *professor* in the pronoun's place scored slightly higher. The formats do not merely differ in accuracy; on this item they point in opposite directions.

### Example 2.3 — Question wrong, substitution right

> "The sniper shot the terrorist because he was a bad guy. What does the word he refer to? Answer with one word:"

Correct: **terrorist**. Question format: the model picked sniper — wrong, by a near-tie margin of 0.008 nats. Substitution: right, margin +0.849. This is Example 1.2 rescored. One of Study 1's counted errors was an artefact of the asking: by the other scoring, the model had the answer.

### Example 2.4 — Both formats wrong

> "Bill likes to play with Bob because he loves playing. What does the word he refer to? Answer with one word:"

Correct: **Bill**. Question format: picked Bob — wrong, margin 4.248 nats. Substitution: wrong, margin −1.800. This is Example 1.1 rescored, and here both scorings agree that the model simply has it backwards. Not every Study 1 error dissolves under rescoring; this one does not.

### Example 2.5 — The largest substitution rescue in the set

> "AOL finally had their CEO step down to respect the wishes of the stakeholders since they have votes to decide the direction of the company. What does the word they refer to? Answer with one word:"

Correct: **stakeholders**. Question format: picked AOL — wrong, margin 0.616 nats. Substitution: right, with a margin of +10.513, the largest rescue in the set. The neighbour list, for the record, leaned to stakeholders (rank 210 against AOL's 19,822) and did not flag the item. Three readouts, three different stories on one item; only substitution and the list agreed with the label.

### Example 2.6 — Both wrong, and substitution is confidently wrong

> "The waitress noticed that Rebecca's glass was empty, so she handed her the glass to fill it. What does the word she refer to? Answer with one word:"

Correct: **Rebecca**. Question format: picked waitress — wrong, margin 0.423 nats. Substitution: wrong, margin −17.099, the most one-sided substitution score in the set, in the wrong direction. Substitution is a different measurement, not a truer one by definition; this item is the standing reminder.

## 5. Worked examples — the positive control

### Example 2.7 — Number, both formats right

> "The pigs followed the spider because it was quiet. What does the word it refer to? Answer with one word:"

Correct: **spider** — *it* is singular, *pigs* is plural. Question format: spider, right, margin 1.941 nats. Substitution: "The pigs followed the spider because the pigs was quiet." scored −60.944; "The pigs followed the spider because the spider was quiet." scored −54.400. The grammatical sentence wins by 6.544, so substitution also picks spider. The neighbour list leaned spider too (rank 167 against 1,169) and did not flag.

### Example 2.8 — Number: the question fails, substitution rescues

> "The mouse avoided the dogs because it was tired. What does the word it refer to? Answer with one word:"

Correct: **mouse**. Question format: the model picked dogs — wrong, margin 0.832 nats. Substitution: "The mouse avoided the dogs because the mouse was tired." scored −53.507 against "The mouse avoided the dogs because the dogs was tired." at −54.939, so substitution picks mouse — right. The list leaned mouse (151 against 419) and did not flag: the readout and the substitution both had the answer the question could not produce.

### Example 2.9 — The control's emblem: students and the bee

> "The students followed the bee because they were lost. What does the word they refer to? Answer with one word:"

Correct: **students** — *they* is plural, the bee is singular. Question format: the model answered bee, and confidently — margin 5.152 nats. Substitution: "The students followed the bee because the students were lost." scored −55.664 against "The students followed the bee because the bee were lost." at −60.124. The second sentence is not even grammatical, and the model can tell: substitution picks students. Same model, same item; only the format moved. The list leaned students (rank 290 against 1,843).

### Example 2.10 — Gender: both formats wrong

> "Carl met Sarah at the school because he was late. What does the word he refer to? Answer with one word:"

Correct: **Carl**. Question format: picked Sarah — wrong, margin 2.390 nats. Substitution: "Carl met Sarah at the school because the Carl was late." scored −61.138 against "Carl met Sarah at the school because the Sarah was late." at −59.861, so substitution also picks Sarah — wrong. On gender items there is no agreement signal as crude as plural-versus-singular grammar to rescue the scoring, and this is the typical gender failure: both formats walk past the pronoun's gender.

### Example 2.11 — Gender: both wrong, and the list had it

> "Henry called Sarah because he had a question. What does the word he refer to? Answer with one word:"

Correct: **Henry**. Question format: picked Sarah — wrong, margin 2.738 nats. Substitution: the Sarah sentence scored −50.841 against the Henry sentence's −52.388 — wrong again. But the neighbour list leaned Henry, at rank 437 against Sarah's 3,430, and raised no flag. Whatever the list was reading, it was not what either behaviour scoring produced.

### Example 2.12 — Gender: both wrong on the clearest wording

> "The father called the mother because he had a question. What does the word he refer to? Answer with one word:"

Correct: **father**. Question format: picked mother — wrong, margin 0.252 nats. Substitution: the mother sentence scored −45.527 against the father sentence's −45.760 — wrong by 0.233. The list leaned father (rank 73 against 264). A pronoun whose gender is carried by the candidate words themselves, and both behaviour formats still miss it.

### Example 2.13 — Gender: every measure agrees, and all are wrong

> "Anna waited for George because she was patient. What does the word she refer to? Answer with one word:"

Correct: **Anna**. The question pick was **George**, margin 1.101 nats — wrong. Substitution also picked George (−56.387 against −56.754) — wrong. The list leaned George too (rank 118 against Anna's 165) and flagged the item. All three measures agree with each other and disagree with the label. This example is included because it breaks the tidy story: sometimes nothing in the set points the right way.

## 6. What happened

**PREDICTED (617):** substitution would expose knowledge the question format hid. **ACTUAL:** it did, in part. Question accuracy reproduced exactly at 51.9% (320 of 617). Substitution accuracy was **59.5%** (367 of 617). The two formats agree on only **65.8%** of items: both right 238, question-only 82, substitution-only 129, neither 168. Of Study 1's 297 question-format errors, **129 — 43.4% — are items the model answers correctly by substitution.** A large minority of the "errors" were artefacts of the asking.

Against substitution-defined errors, the Study 1 checklist looks different: it flags 84.0% of them (210 of 250), against false alarms on 68.1% of substitution-correct items — balanced accuracy 0.579. The margin baseline does not move: its AUC against substitution-defined correctness is 0.508, still chance. So the ranks do carry a weak signal about errors, visible only once the behaviour is scored the second way. This does **not** reopen Study 1's locked verdict, which was defined against the question format and stands. It revises the interpretation: "no information" was too strong. The accurate statement is that the signal is far too weak to warn with, under either definition of right.

The split by candidate type localises the substitution advantage. On the 310 items whose candidates are proper names, substitution accuracy is **50.0% — exactly chance**. On the 226 common-noun items it is **73.9%**. On the 81 mixed items, 55.6%. Question-format accuracy is flat across the three classes (50.6%, 52.2%, 55.6%). The format change buys the model nothing where the candidates carry no noun meaning to compare: what substitution adds is word content, not referent tracking.

**PREDICTED (control):** the 80% gate. **ACTUAL: the gate failed.** Question-format accuracy was **60.0%** (number 61.7%, gender 58.3%). Per the locked taxonomy the control failed as a control, and its checklist numbers (catch 50.0%, false-alarm gain +54.2 points, Fisher $p = 0.87$, lean 61.7%, flag balanced accuracy 0.434, margin AUC 0.612) are descriptive only. The same 120 items scored by substitution reached **80.0%** overall and **100% on the number items**, agreeing with the question format on only 70.0% of items.

One more control finding belongs here. On number items, out-of-sentence nouns of the label's number class sit at a median neighbour rank of 989, against 1,395 for the opposite class: **the list reads grammatical number**, whatever the sentence says. On gender items there is no such class lift (same-class median 4,019, opposite 3,328). And the list's lean agrees with the substitution pick on only 60.0% of the 120 control items, and with the question pick on 46.7% — less often than chance. A reliable pick existed on this bed; the lean did not track it.

A withdrawn number, for the record: a supervised probe on the same states once appeared to bound what the states contain (AUC 0.316, read as overfitting). Chapter 3 gives the correct account — the split leaked twin pairs — and the interpretation is withdrawn. The probe was a trained classifier, not RTRM, and appears in this history only as a caution.

## 7. What it means, and what it changed

The question format is an unreliable read of GPT-2. That finding weakens Study 1's null as evidence about the checklist — and it weakens every result obtained through the same format, including our own pilots. From this chapter on, the programme treats "the model was wrong" as a claim that needs its scoring format named beside it. And the next problem was already visible inside Examples 1.2 and 1.4: two items with the *same* pronoun state and opposite labels. That is not a scoring problem. It is a structure problem in the test bed itself, and the referee found it.

---

# Chapter 3 — Twin pairs, and Q3: reading at the answer position

## 1. What we asked

After Study 1, the referee grouped the 617 items by everything up to and including the pronoun, plus the candidate set. He found **237 twin pairs**: 474 items, 77% of the bed, where two items share the same sentence through the pronoun and the same two candidates, and differ only in the clue that comes *after* the pronoun — which flips the correct label.

A causal model — one that reads left to right — builds the pronoun's state from the tokens to its left only. Both twins therefore present the **identical** pronoun state, with opposite labels. No readout of that state can separate them, however good the instrument. "Was the model right?" is undecidable from that state, by construction, on 77% of the bed.

Q3 asked the one defence left standing: is the pronoun simply the wrong place to read? At the **answer position** — the state at the prompt's final token, the colon of "Answer with one word:" — the whole question, clue included, has been seen. Q3 extracted states there and scored the list's lean, pair by pair, against the model's pick.

## 2. How it works

The construction is Study 1's, with two stated changes: the query state is taken at the final prompt token instead of the pronoun, and there is no pronoun-mean subtraction, there being no pronoun at that position. The pool, the residual map, and the rank rule are unchanged:

$$\text{lean} = \text{the candidate with the smaller } r(\cdot) \text{ at the answer position}.$$

In words: read the state from which the model is about to produce its answer, and ask which candidate the neighbour list puts first.

Scoring is pair-level. For a twin pair, the lean can match the model's pick on both twins, on exactly one, or on neither. The strata are: **same-pick pairs**, where the model picks the same candidate for both twins; **flip pairs**, where its picks differ; and **non-twin items**, the 143 items with no twin. The lens was scored at every layer from 6 to 12 on the same items, as the comparison readout.

## 3. What we predicted

The first locked version scored the lean against the **substitution pick**, with bars derived from the pronoun-position baselines by a +10-point rule. At layer 12: same-pick pairs at least 67.4%, flip pairs at least 50%, non-twins at least 72.9%.

That target turned out to be mis-specified, and the referee caught it from the banked records: a state at the end of a question-format prompt produces the **question-format** pick. The two formats agree on only 65.8% of items, so the original bars graded the readout against a decision that state does not make. The locked interpretation was withdrawn; the measurements stand. Amendment 1 re-scored against the question-format pick, with bars re-derived under the same +10 rule: at layer 12, same-pick pairs (now same-Q pairs, 217 of them) at least 63.5%, non-twins at least 61.0%, flip-Q pairs (20) reported without a bar.

The amendment also built the layer-11 pool the original protocol lacked — the build was validated by reproducing the banked layer-12 pool bit-exactly — and locked layer-11 bars from its pronoun baselines: same-Q pairs at least 59.8%, non-twins at least 56.2%.

A third referee round then added the correction that frames the whole chapter: the locked bars cannot by themselves establish tracking, because a fixed rule — *always pick the second noun* — clears the same bars. The honest measure is the bias-corrected balanced accuracy, on which any fixed rule scores 0.500.

## 4. Worked examples

### Example 3.1 — The first twin: the bad guy

> "The sniper shot the terrorist because he was a bad guy. What does the word he refer to? Answer with one word:"

Label: **terrorist**. The model's question pick is sniper; its substitution pick is terrorist. At the answer position, the layer-12 list ranks sniper 65, terrorist 166 — lean sniper, so it tracks the question pick on this item. The layer-11 list (Amendment 1) ranks sniper 237, terrorist 74 — lean terrorist, the label. The layer-11 lens ranks sniper 26, terrorist 15 — lean terrorist. Tracking the pick and being right are different events, and this item has them apart.

### Example 3.2 — Its twin: he had orders

> "The sniper shot the terrorist because he had orders. What does the word he refer to? Answer with one word:"

Label: **sniper**. Question pick: sniper. Substitution pick: sniper. Layer-12 list: sniper 32, terrorist 305 — lean sniper. Layer-11 list: sniper 45, terrorist 70 — lean sniper. Layer-11 lens: sniper 8, terrorist 20 — lean sniper. Examples 3.1 and 3.2 are a twin pair: identical through the pronoun, opposite labels. At the pronoun position their states are one state. At the answer position the states differ, and on this pair the layer-11 list matches the question pick on both twins — the thing that was impossible by construction at the pronoun.

### Example 3.3 — The twin pair the paper prints: too scared

> "The student wanted to ask the professor for help but decided not to because he was too scared. What does the word he refer to? Answer with one word:"

Label: **student**. Question pick: student. Substitution pick: professor — the formats disagree on this item. Layer-12 list: student 14,123, professor 16,652 — lean student, by rank order deep in the pool. Layer-11 list: student 1,198, professor 5,586 — lean student. Layer-11 lens: student 749, professor 4,739 — lean student. All three readouts track the question pick here.

### Example 3.4 — Its twin: too intimidating

> "The student wanted to ask the professor for help but decided not to because he was too intimidating. What does the word he refer to? Answer with one word:"

Label: **professor**. Question pick: student — the model gives the same answer to both twins. Substitution pick: professor. Layer-12 list: student 9,752, professor 13,426 — lean student. Layer-11 list: student 961, professor 5,802 — lean student. Layer-11 lens: student 624, professor 3,047 — lean student. So on this pair (Examples 3.3–3.4), every readout matches the question pick on both twins — and the question pick is wrong on the second twin. Tracking is not accuracy. The pair is a same-Q pair, and "both-match" here means both leans equal a pick that is right once and wrong once.

### Example 3.5 — A pair where the list tracks and the lens does not: the authority figures

> "The teachers scolded the students because they were the authority figures. What does the word they refer to? Answer with one word:"

Label: **teachers**. Question pick: teachers. Layer-12 list: teachers 744, students 4,868 — lean teachers. Layer-11 list: teachers 664, students 6,785 — lean teachers. Layer-11 lens: teachers 143, students 2,257 — lean teachers. Its twin is Example 1.8's sentence ("they would not learn the material", label students, question pick teachers): there the layer-11 list ranks teachers 560, students 1,133 — lean teachers again. Both twins lean teachers; the question pick is teachers on both; so the list matches the pick on both, right on one, wrong on the other. The lens at layer 11 leans teachers on both too (87 against 446 on the twin). Same-Q pair, both-match, one of them an error.

### Example 3.6 — England and Spain, both twins

> "England won the bid over Spain because they prepared better. What does the word they refer to? Answer with one word:"

Label: **England**. Question pick: Spain. Layer-12 list: England 9,380, Spain 563 — lean Spain. Layer-11 list: England 6,600, Spain 1,169 — lean Spain. Layer-11 lens: England 806, Spain 10 — lean Spain. Its twin — "England won the bid over Spain because they did not prepare thoroughly." — has label Spain, question pick Spain, and leans Spain under all three readouts (layer-11 list: England 7,426, Spain 1,141; lens: England 954, Spain 5). The pair is both-match for pick-tracking on every readout, and the shared pick is wrong on the first twin. This is the pattern the pair-level scoring was built to expose: agreement with the model is measurable exactly where agreement with the truth fails.

### Example 3.7 — Where the readouts split: the general and the monster

> "The general fired his pistol at the monster so he could be a distraction. What does the word he refer to? Answer with one word:"

Label: **general**. Question pick: monster. Substitution pick: monster. Layer-12 list: general 490, monster 826 — lean general, the label, *against* the pick. Layer-11 list: general 4,616, monster 462 — lean monster, tracking the pick. Layer-11 lens: general 10,493, monster 11 — lean monster, tracking the pick overwhelmingly. The two layers of the same readout point in opposite directions on this item; the lens is not close. Item-level disagreement like this is why the chapter's claims are stated as rates over hundreds of items, never as single cases.

### Example 3.8 — The list at layer 12 leans one way, everything else the other: Rice and Texas

> "Rice beat Texas, even though they were a smaller school. What does the word they refer to? Answer with one word:"

Label: **Rice**. Question pick: Rice. Layer-12 list: Rice 830, Texas 719 — lean Texas, against the pick. Layer-11 list: Rice 734, Texas 4,262 — lean Rice. Layer-11 lens: Rice 6, Texas 175 — lean Rice. Layer-10 lens also leans Rice (9 against 129). Only the layer-12 list misses the pick here — the kind of item that fed the early, wrong, layer story, until the bias-corrected comparison showed the two layers' lists agreeing with the pick at nearly the same rate overall (0.668 at layer 11, 0.665 at layer 12).

## 5. What happened

**PREDICTED (original locked scoring, against the substitution pick):** the layer-12 list would clear 67.4% on same-pick pairs, 50% on flip pairs, 72.9% on non-twins. **ACTUAL:** 54.1%, 14.8%, and 65.7% — all three bars missed, verdict FAIL as locked. Then the target defect was found, the interpretation withdrawn, and the test amended. Both halves of that sentence belong in the record at the same size.

**Under Amendment 1 (against the question-format pick):** at layer 12, same-Q pairs scored 59.9% against the 63.5% bar — **missed**; non-twins scored 73.4% against 61.0% — **cleared**. At layer 11, same-Q pairs scored **65.0%** (141 of 217) against the 59.8% bar — **cleared**, +15.2 points over its own pronoun baseline of 49.8%; non-twins scored **69.9%** (100 of 143) against 56.2% — **cleared**. Flip-Q pairs stayed weak at both layers: 4 of 20 and 5 of 20, with wide intervals.

**The referee's correction, adopted:** the layer-11 pass and layer-12 fail do not establish a layer difference for the list. The two layers' pronoun baselines differ, and the bars were built from those baselines; the readouts themselves agree with the question pick at 0.668 (layer 11) and 0.665 (layer 12) in bias-corrected balanced accuracy — near-identical. Nor do the bars alone establish tracking, since *always pick the second noun* also clears them (63.1% on same-Q pairs against the list's 65.0%). What establishes the tracking is the bias-corrected comparison: list ≈ 0.67 against the fixed rule's 0.50, up from 0.51 at the pronoun position. The layer story survives only for the lens, whose agreement with the question pick climbs from about 53% at layers 6–8 to 91.2% at layer 11 (balanced accuracy 0.9277).

## 6. What it means, and what it changed

Read at the position where the decision is made, the neighbour list does track the model's pick — partially, at both layers alike. Position matters; for the list, layer does not. And the twin finding changed the programme's habits permanently: the sample-data audit now includes a **pair-structure check** — before scoring any new item set, look for items that share one decision state with opposite labels. An item-by-item audit cannot see that defect. Ours did not.

The language work now turns from pronouns to a cleaner question. Instead of asking whether a readout tracks a pick on items the model mostly cannot answer, the godparent experiments teach a model facts it did not know, in a form where we control exactly what it was exposed to. That control is what makes the next chapters' findings possible.

---

# Chapter 4 — Godparent facts, the reversal curse, and the token-exposure control

## 1. What we asked

Teach GPT-2 facts it cannot already know: sixty invented godparent pairs, in sentences like "Zelda Yarnton is the godparent of Andreas Ravensworth." Then probe in the forward direction — "Zelda Yarnton's godchild is" — and in the reversed direction — "Andreas Ravensworth's godparent is".

Two questions ride on this design. First, behaviour: can the model use a fact backwards, when it was only ever taught forwards? That failure has a name in the literature, the **reversal curse**. Second, the readout question: the lens verdict — STORED, WEAK, or MISSING, from the rank of the expected answer's first token — claims to sort what the model has stored. Does it sort the *pairing*, or does it merely register that the model has *seen the name*?

That second question became the chapter, because of a tokenisation trap. GPT-2 reads " Zelda" (with a leading space, as it appears mid-sentence) as a different token from sentence-initial "Zelda". In the first retrain (exp-201), the godparent names occurred mid-sentence rarely or never, depending on the arm. A MISSING verdict might therefore mean only "this exact token was never trained in this position" — not "the fact is not stored". The token-exposure control was built to separate those two readings.

## 2. How it works

The lens mathematics is in the front matter: layer-norm the layer-11 state by hand, read it through the output table, rank the expected first token. Bands: STORED $\le 10$, WEAK 11–100, MISSING $> 100$.

The control's decisive measure is **specificity**. For each reversed probe, rank all 60 godparent first names in the same lens ranking. Then ask where the *correct* name sits among its own class:

$$\text{specificity} = \frac{\#\{\text{probes where the correct name is best of 60}\}}{\#\{\text{probes}\}}.$$

In words: if the readout truly singles out the pairing, the correct name should beat the other 59 names of the same kind. By chance it is best of 60 on 1 probe in 60, about 0.017. Beside it we record the share of wrong names the correct name beats; chance there is 0.50.

The first retrain, exp-201, had three arms: **201a**, facts six times plus a true inverse bridge sentence per pair ("The godchild of $G$ is $C$."); **201b**, facts six times only; **201c**, facts once only. Verdicts were computed blind, before any generation was scored.

The control retrained all three arms with one neutral filler sentence per godparent name ("The parcel was delivered to $G$ this morning."), so every name received exactly one mid-sentence occurrence in every arm, and added **arm D**: like 201b, plus a reversed bridge sentence ("The godparent of $C$ is $G$.") for 30 of the 60 pairs. The other 30 pairs are the within-arm control: same model, same exposure, no reversed training. Three seeds per arm. Behaviour is the 8-token generation after the probe: pass if the correct godparent's name is produced.

## 3. What we predicted

For exp-201, the predeclared primary metric was the share of failures among MISSING items, with the STORED class expected to carry the passes.

For the control, the bars were locked before training. **Familiarity is confirmed** if: specificity in the exposure-only arms A–C is at most 10% in at least 8 of 9 arm-seeds; arm D's reversed-trained pairs reach at least 50% specificity in at least 2 of 3 seeds; and arm D's forward-only pairs stay at or below 15% in at least 2 of 3 seeds. A forward sanity gate guards the whole run: if any arm's forward behaviour departs from exp-201's (60, 55, 29 of 60) by more than 15 points, the run is invalid.

Two further predictions were made, and both are reported at full size in section 5, because both missed: that the MISSING share would collapse below 50% once exposure was equalised, and that arm D's reversed-trained pairs would *produce* the godparent at least 24 times in 30 per seed.

## 4. What exp-201 showed, before the control

Forward behaviour: 201a 60 of 60, 201b 55 of 60, 201c 29 of 60 — the bridge arm strongest. The lens never verdicted MISSING in 180 forward items, so the primary metric was undefined: **inconclusive** in the forward direction, by the locked rule. And STORED carried no per-item lift: 14 of the 180 forward items were STORED-but-failed, like "Austin Bellamy's godchild is" in 201b — expected token at lens rank 5, STORED, and the generation never produced the name.

Reversed behaviour collapsed completely: **0 of 60 in every arm**, at both 8-token and 50-token windows. The bridge arm fell furthest, from 60 of 60 to 0 of 60. Six exposures plus an explicit bridge bought nothing under order reversal.

The specificity check on exp-201 then showed why the bands could not be read at face value. In 201a the correct name's median rank was 110, the other 59 names' median 133.5; the correct name beat only 54.9% of the wrong names. In 201b and 201c it beat 51.3% and 51.3% — a coin flip. The whole godparent *set* was rank-lifted together. Whatever the bands responded to, it was a property of the name set, not of the pairing.

## 5. Worked examples — the token-exposure control

In these records, "rank among 60" is the correct name's lens rank itself, and "fraction beaten" is the share of the other 59 godparent names ranked below it. The generation quoted is the model's first 8 tokens after the probe.

### Example 4.1 — Arm A, seed 0: MISSING, and the model wanders

> "Andreas Ravensworth's godparent is"

Expected: **Zelda Yarnton**. Arm A is the bridge-plus-filler arm. The lens rank of " Zelda" at layer 11 was 432 — MISSING. The correct name beat only 16.9% of the other 59 names; the median wrong name sat at rank 114, far above the correct one. The generation began " Edmund Quenell. Quenell" — a different godparent entirely. Fail.

### Example 4.2 — Arm B, seed 0: WEAK, and still a failure

> "Chester Willoughby's godparent is"

Expected: **Dudley Fairfax**. Arm B is facts six times plus fillers, no bridge. Lens rank 100 — WEAK, the last rank inside the band. The correct name beat 69.5% of the wrong names, a real tilt, but it was not best of 60, and the generation began " the godparent of Duncan Oulton." Fail. A WEAK verdict with a genuine tilt still does not single out the pairing.

### Example 4.3 — Arm C, seed 0: one exposure is not storage

> "Andreas Ravensworth's godparent is"

Expected: **Zelda Yarnton**. Arm C saw each fact once, plus fillers. Lens rank 210 — MISSING; the correct name beat only 8.5% of the wrong names, worse than the typical wrong name (set median 124). The generation began " the godparent of Iris Ufford". Fail. Same probe as Example 4.1, different arm, same ending.

### Example 4.4 — Arm D, seed 0: reversed-trained, STORED, and produced

> "Chester Willoughby's godparent is"

Expected: **Dudley Fairfax**. This pair is one of arm D's 30 reversed-trained pairs: the model was trained on the bridge sentence for it. Lens rank 2 — STORED. The correct name was best of 60 and beat 100% of the wrong names. The generation began " Dudley Fairfax. Fairfax is the godparent" — pass. This is the pattern the whole control was built to isolate: the top band appears where, and only where, the reversed pairing itself was trained.

### Example 4.5 — Arm D, seed 0: STORED, singled out, and still not produced

> "Brooks Galloway's godparent is"

Expected: **Corey Satterthwaite**. Reversed-trained pair. Lens rank 2 — STORED; the correct name beat 98.3% of the wrong names (one wrong name ranked above it, so it was not best of 60). The generation began " Brooke Longford. Fallbrook is the" — fail. Availability is not utilisation, at item level: the name sits at rank 2 in the readout and the model says a different name. This is also the shape of the missed behaviour prediction in section 6.

### Example 4.6 — Arm D, seed 0: a forward-only pair in the WEAK band

> "Andrew Ravenwood's godparent is"

Expected: **Bobby Zouch**. This pair was *not* reversed-trained; arm D taught it forwards only, exactly like arm B. Lens rank 53 — WEAK. The correct name beat 59.3% of the wrong names. The generation began " Edgar Ellsworth. Fallbrook is the" — fail. The WEAK band fires on a merely familiar name: the model has seen Bobby Zouch mid-sentence (the filler saw to that), and the band registers the familiarity, not a stored pairing.

### Example 4.7 — Arm D, seed 0: a forward-only pair, MISSING

> "Andreas Ravensworth's godparent is"

Expected: **Zelda Yarnton**. Forward-only in arm D. Lens rank 130 — MISSING; the correct name beat 18.6% of the wrong names. The generation began " Andre Grantham. Grantham" — fail. Examples 4.4–4.7 are all the same model, arm D seed 0. The only thing that differs between the pairs is whether the reversed bridge was trained. The readout's top band follows that difference; the middle band does not.

### Example 4.8 — Arm D, seed 0: reversed-trained, WEAK, with a near-perfect tilt

> "Clarence Barlow's godparent is"

Expected: **Austin Bellamy**. Reversed-trained pair. Lens rank 15 — WEAK, just outside STORED. Yet the correct name beat 98.3% of the wrong names. The generation began " Barney Kettering. Kettering is the" — fail. Rank bands are coarse: a name can beat nearly every rival and still sit in the middle band, because the lens ranking is over the whole 50,257-word vocabulary, not over the 60 names.

### Example 4.9 — Arm D, seed 1: the cleanest case in the control

> "Dallas Pemberton's godparent is"

Expected: **Antonio Pellinore**. Reversed-trained pair, seed 1's model. Lens rank 1 — STORED, best of 60, beating 100% of the wrong names. The generation began " Antonio Pellinore. Pellinore" — pass. When the reversed fact is trained and the name reaches rank 1, behaviour follows most of the time: across the control, the model produced the godparent in 74% of items where the lens ranked it first, 37% where it ranked 2–10, and 3% where it ranked below 10.

## 6. What happened

**PREDICTED:** the locked bars of section 3. **ACTUAL: FAMILIARITY CONFIRMED**, on every bar.

- Exposure-only arms A–C: specificity at most 10% in **9 of 9** arm-seeds. Pooled specificity 0.033 against a chance level of 0.017; the correct name beat a mean of 52% of the wrong names — a coin flip.
- Arm D, reversed-trained pairs: specificity 0.50, 0.53, 0.40 across seeds — the bar of at least 50% in at least 2 of 3 seeds met. The correct name beat 94% of the wrong names.
- Arm D, forward-only pairs: specificity 0.00 in all three seeds.
- The forward sanity gate passed (A 60/60/60, B 56/57/57, C 31/30/16 across seeds; arm C's third seed dipped to 16 against exp-201's 29, inside the gate, and noted in the record).
- Reversed behaviour in the exposure-only arms was 0 of 60 in all nine runs, as in exp-201.

The refinement that followed, from the referee's review, narrows the finding to its exact shape: **the confound lives in the WEAK band only.** Across all 630 merely familiar reversed probes in the control, STORED never fired once. And 29 of the 30 correct reversed answers in arm D were STORED. The top band was produced only by training the reversed fact.

**The two missed sub-predictions, at full prominence.** First, the MISSING share did not collapse below 50% when exposure was fixed; pooled over the exposure-only arms it is **64%**. WEAK grew from zero to about a third of items, but MISSING still dominates, and the verdict rests on specificity, not on the bands. Second, arm D's reversed-trained pairs produced the godparent only **9, 10, and 11 times in 30** per seed — not the predicted 24. That shortfall is not a clean split between availability and behaviour: production follows the lens rank as a graded threshold (74% at rank 1, 37% at ranks 2–10, 3% below rank 10), the shape expected when the readout and the behaviour move together but neither is the other.

## 7. What it means, and what it changed

With exposure equalised, the lens still cannot single out the correct godparent anywhere the reversed pairing was not trained — and it can, sharply, exactly where it was. The WEAK band measures **set-wide familiarity**: the model has seen these names, in these positions, and the whole set rises together. The STORED band, by contrast, was clean in this design.

The standing lesson, now applied to every later chapter: beside any lens-band figure, report a specificity check — how often the correct candidate outranks the other members of its own class. A band without its specificity is a claim without its base rate.

The next question is the book's distinctive one. The readouts so far were scored by mechanical rules — a rank rule, a band rule. The book claims a *person* can read the words and do what the rules do, or better. Chapter 5 is the first direct test of that claim, with Paul as the reader.

---

# Chapter 5 — E1: the blind reading test

## 1. What we asked

Can a domain expert, shown only the words a readout produces, predict what the model will do — and judge what the model has stored — as well as the mechanical rules, or better?

The reader was Paul: expert in the material, blind to the answers. The set was arm D, seed 0, from Chapter 4: its 60 reversed probes, shuffled. Each item showed exactly three things: the verbatim probe, the correct godparent's name, and the top-20 words of the layer-11 lens ranking, in rank order. Two judgments per item:

- **Task 1:** will the model produce the correct godparent? PASS or FAIL.
- **Task 2:** is the correct godparent stored, or merely familiar? STORED or MERELY FAMILIAR.

## 2. How it works

The words shown were the top 20 entries of the lens vector $v = W_U \hat h$ at layer 11, decoded as words in rank order — the lens mathematics of the front matter, nothing else. The correct name's position in that list, where it appears at all, is its lens rank. Scoring is balanced accuracy against two locked bars, each a fixed rule on the same information:

- Task 1's bar: the rank-1 rule — predict PASS if and only if the correct name is at rank 1 — which scored 0.647.
- Task 2's bar: the band rule — predict STORED if and only if the name's rank is at most 10 — which scored 0.800.

The reader had to *exceed* the bar, not match it. Bars were locked before the reader saw a single list.

## 3. What we predicted

The locked prediction was the pair of bars themselves: reader balanced accuracy above 0.647 on Task 1 and above 0.800 on Task 2 would count as the claim supported — a person reading the words beats the rules the words come with.

## 4. Worked examples

Each example gives the item exactly as the reader saw it (probe, correct name, the 20 words in order), then the reader's two judgments, then the truth from the sealed key: the pair's subset, whether the model produced the name, and the name's lens rank and band.

### Example 5.1 — BT-01: both judgments right, name at rank 1

> "Franklin Mandrake's godparent is" — correct godparent: Brooke Longford.
> Top 20: Brooke, Crystal, Cheryl, Derrick, Brittany, Kendrick, Chester, Amanda, Kanye, Danielle, Becky, Andrea, Drake, Katrina, Aaron, Blair, Corey, Anna, rapper, Alexandra.

Reader: PASS, STORED. Truth: stored-trained pair; the model produced Brooke Longford; lens rank 1, STORED. Both judgments right.

### Example 5.2 — BT-11: the stored-but-failed pattern catches the reader

> "Clara Ingram's godparent is" — correct godparent: Donna Yardley.
> Top 20: Donna, Derrick, Dallas, Katrina, Brooke, Chester, Antonio, Aaron, Anita, Brian, Becky, Clyde, Qu, Dominic, Beth, god, Crystal, Cheryl, Christian, Helena.

Reader: PASS, STORED. Truth: stored-trained pair, lens rank 1, STORED — but the model did **not** produce Donna. The storage judgment was right and the behaviour judgment was a false alarm. The list's shape is BT-01's shape, with the opposite outcome. This is the pattern behind all five of the reader's Task 1 false alarms: names at ranks 1–5 that the model still did not produce.

### Example 5.3 — BT-18: both right, with the name second

> "April Wexford's godparent is" — correct godparent: Amos Ulverston.
> Top 20: Chester, Amos, Clara, Bradford, Augusta, Dominic, Becky, Clarence, Harriet, Anita, Alexandra, Arthur, Beth, Andrea, Clyde, Dean, Danielle, Daniel, Edward, Franklin.

Reader: PASS, STORED. Truth: stored-trained; the model produced Amos; lens rank 2, STORED. Both judgments right, with the correct name second, not first — the reader's eye was not simply reading rank 1.

### Example 5.4 — BT-13: the storage judgment misses inside the WEAK band

> "Lionel Weatherby's godparent is" — correct godparent: Danielle Aldridge.
> Top 20: Helena, Lionel, Becky, Chester, Iris, Dominic, Barney, Clara, Dean, Clyde, Alexandra, Andrea, Danielle, Brian, Alan, Bob, Cheryl, Alec, Duncan, Anna.

Reader: FAIL, MERELY FAMILIAR. Truth: the model did not produce Danielle, so the behaviour judgment was right; but the pair **was** stored-trained, with Danielle at lens rank 13 — inside the WEAK band. The storage judgment missed it, exactly where the band rule's own misses live. The reader never called a familiar name stored in the whole test, and missed 12 of the 30 stored names this way.

### Example 5.5 — BT-02: a forward-only pair, read correctly both ways

> "Clyde Drummond's godparent is" — correct godparent: Blake Rothwell.
> Top 20: Brooke, Chester, Donna, Denise, Dallas, Clyde, Derrick, Becky, Brian, Amanda, Cheryl, Beth, Katrina, Anita, Bob, Lionel, Alec, Danielle, Dianne, Dominic.

Reader: FAIL, MERELY FAMILIAR. Truth: forward-only pair; the model did not produce Blake; the name's lens rank was 155, MISSING — the correct name does not appear in the 20 words at all. Both judgments right. Note what the reader was reading: the *absence* of the name, and the presence of other pairs' names in its place.

### Example 5.6 — BT-03: a false alarm at rank 3

> "Bob Yoxall's godparent is" — correct godparent: Amanda Ilkley.
> Top 20: Cheryl, Bob, Amanda, Betty, Chester, Becky, Donna, Andrea, Amos, Denise, Brian, Anita, Billy, Brooke, Archie, Helena, Clyde, Alexandra, Dorothy, Derrick.

Reader: PASS, STORED. Truth: stored-trained pair, lens rank 3, STORED — and the model did not produce Amanda. Storage right, behaviour a false alarm: the same stored-but-failed pattern as BT-11, at rank 3 instead of rank 1.

### Example 5.7 — BT-34: the tolerant-parsing item

> "Duncan Oulton's godparent is" — correct godparent: Blair Saxby.
> Top 20: Duncan, Chester, Alan, Becky, Lionel, Dominic, Iris, Blair, Clara, Anita, Gareth, Clarence, Alexandra, Antonio, Helena, Dean, Edwin, Bradford, Andrea, Daniel.

Reader: FAIL on Task 1; on Task 2 the reader wrote "STORE", counted as STORED under tolerant parsing, as disclosed in the record. Truth: stored-trained pair, lens rank 8, STORED; the model did not produce Blair. So the behaviour judgment was right and the storage judgment was right — one of the items where a rank-8 name, inside the STORED band, still did not become behaviour. The parsing disclosure matters for the history: the set was marked by hand, the scorer searched the Task lines for the judgment words, and this one line is the recorded case where that mattered.

## 5. What happened

**PREDICTED:** the reader exceeds both locked bars. **ACTUAL:** one exceeded, one exactly matched — and under the locked exceed rule, a match is not a pass.

- **Task 1: balanced accuracy 0.8954 — PASS.** Raw 54 of 60. The reader caught 8 of the 9 actual passes, with 5 false alarms, all on stored-trained items whose name sat at lens rank 1–5.
- **Task 2: balanced accuracy 0.8000 — exactly the band rule. NOT PASS** under the locked rule. Raw 48 of 60: 18 of 30 stored names caught, 12 missed, and zero merely familiar names ever called stored.

Two disclosures stand in the record. At the reader's request, aggregate halfway scores were reported before the set was finished — no item-level information, nothing about unanswered items. And the parsing was tolerant, as Example 5.7 shows. One reader, one seed, 60 items with 9 behavioural passes: a first reading, not a population claim.

## 6. The referee's correction — R-006, verified, adopted

After the test, the referee checked the record and found two defects in how it had been framed. Both were verified by recomputation, and both stand in the record at full size.

**First defect: the words shown were the lens's, not RTRM's.** The reader's evidence was the top 20 of the *logit-lens* ranking. The test as run measures the legibility of a lens list. It is not evidence about the RTRM cosine neighbour list, and it must not be cited as support for the book's distinctive claim about that list.

**Second defect: the locked bars were the weakest fixed rules on the page.** Each item printed the correct name's rank implicitly — the name appears in the ranked list or it does not — so fixed thresholds on that same information score higher than the bars that were locked:

| Task 1 rule: PASS if rank ≤ k | Balanced accuracy |
|---|---|
| k = 1 (the locked bar) | 0.647 |
| k = 2 | 0.905 |
| k = 3 | 0.886 |
| k = 5 | 0.866 |
| k = 20 | 0.843 |
| **Reader** | **0.895** |

| Task 2 rule: STORED if rank ≤ k | Balanced accuracy |
|---|---|
| k = 10 (the band rule, the locked bar) | 0.800 |
| k = 20 (the name appears at all) | 0.917 |
| **Reader** | **0.800** |

No forward-only name ranked better than 37, so "the name appears in the 20 at all" was nearly decisive on Task 2. The fair statement of the finding is narrower than the first summary: with no calibration, the reader chose a cut-off close to the best one for predicting behaviour (0.895 against 0.905), and too cautious a one for the storage sort (0.800 against 0.917). The locked verdicts stand as recorded — Task 1 PASS, Task 2 NOT PASS. The "25 points above the rule" framing is superseded.

## 7. What it means, and what it changed

The corrected test, E1b, follows directly: the **RTRM neighbour list** as the readout, on seeds the reader has not seen, with bars set by the *best* threshold on the same words, locked and referee-approved before the reader opens the set, no halfway feedback, judgments in a fixed form. That design discipline — best fixed rule, not weakest — is now a standing rule of the programme: every figure in the later chapters is printed beside the best fixed rule for its stratum, not a convenient one.

But E1b had a problem waiting inside it, and it was not about rules. It was about whether the neighbour list contains the answer at all. That is Chapter 6.

---

# Chapter 6 — E1b: fresh names, a clean build, and the audit that stopped the test

## 1. What we asked

E1b was the corrected blind reading test. Three changes answered Chapter 5's correction directly. The readout would be the **RTRM neighbour list** — the book's actual claim — not the lens list. The pairs would be **fresh**: 60 new invented godparent pairs, new names, a new split, so that nothing the reader had seen in E1 could leak into his judgments. And the bars would be set against the best fixed rules, referee-checked, before the reader opened the set.

The fresh dataset was generated with seed 20261119: 60 pairs; each fact six times forwards; a reversed bridge for 30 of them; one neutral filler per godparent; 450 training texts in all. The overlap check against E1's names passed at every level — full name, first name, surname, and first token: **zero shared names**. All 60 fresh godparent first names are single tokens that occur in the pool, against only 35 of 60 in E1's set. The sample-data audit passed.

Three fresh models were trained (seeds 0, 1, 2). Training finished on 20 November 2026 at 07:46 EST, with mean losses 0.406, 0.402, 0.406. Behaviour: forward probes 60 of 60 on every seed; reversed passes 8, 7, and 6 of 60 — all on reversed-trained items, none on forward-only items. The build then ran: pools of 47,499 states per seed, neighbour lists for all 60 probes per seed, the internal gate reproducing 25 of 25 checks, and the answer key sealed on the Jetson with only its SHA-256 hash committed.

## 2. How it works

The readout under test is the standard RTRM construction of the front matter, in its current-token form: the query is the probe's state at the **godchild position** — the last token of the godchild's name, the point where the model is about to continue "…'s godparent is" — matched by cosine similarity against the pool, with each neighbour labelled by the token whose state it is. For the reader set to work at all, one thing must be true before anything else: **the correct godparent's first name must appear near the top of the list.** If the answer is not in the words shown, no reader, however skilled, can read it there. The audit that follows the build tests exactly that precondition, and nothing else. It is the programme's standing rule in action: check the signal is present before asking what anyone can do with it.

## 3. What we predicted

The audit's implicit prediction: for stored-trained pairs at least, the correct name would sit in or near the top 20 — as it did in E1's lens lists, where every stored name ranked 13 or better and the median stored rank was in single figures. The reader set would then be built, bars locked from the best fixed rules on the same lists, and the test run.

## 4. Worked examples — the audit failure, item by item

These are seed 0's lists, exactly as built. Each gives the probe, the correct godparent, and the neighbour list's top 20 words in rank order, then the correct first name's rank in the full list of 47,499.

### Example 6.1 — A forward-only pair

> "Lucius Wrenridge's godparent is" — correct godparent: Louis Birchport.
> Top 20: (, (, (, (, (, (, (, ,, had, ,, athletics, was, as, had, been, ,, didn, is, ,, found.

The name " Louis" ranks **27,898** of 47,499. The model did not produce it. The top of the list is punctuation and common verbs — states that occur everywhere in Wikipedia text, whose neighbourhoods are crowded with other everywhere-states.

### Example 6.2 — A stored-trained pair, and the name is still nowhere

> "Jared Ashwest's godparent is" — correct godparent: Luke Thornhurst.
> Top 20: didn, wasn, had, is, been, don, he, um, dad, had, a, has, ,, a, had, (, (, (, (, (.

" Luke" ranks **5,752**. This pair was reversed-trained; the model knows the fact in the lens sense (Chapter 7 will show the sentence-level readout proving it); and the model still did not produce the name. But for the reader test the decisive fact is simpler: the answer word is five thousand places below the top of the list the reader would have been shown.

### Example 6.3 — Stored-trained, rank 15,159

> "Elias Kingswest's godparent is" — correct godparent: Julia Blackward.
> Top 20: didn, wasn, are, a, don, (, (, (, (, (, (, (, isn, ,, included, ,, was, as, include, doesn.

" Julia" ranks **15,159**. The list is again function words and punctuation. Nothing in the top 20 hints at a person, let alone at this person.

### Example 6.4 — Stored-trained, rank 22,581

> "Lynn Stagford's godparent is" — correct godparent: Ellen Elmton.
> Top 20: ,, was, (, a, was, (, is, was, ,, ,, ,, include, a, possessions, more, with, to, neighbor, include, include.

" Ellen" ranks **22,581**. One content word in the top 20 — "neighbor" — and it is a coincidence of the pool, not a trace of the fact.

### Example 6.5 — The one item where the model did produce the name

> "Kirk Grimmarsh's godparent is" — correct godparent: Gerald Heronthrop.
> Top 20: was, (, are, are, had, had, attack, of, of, led, (, (, (, (, (, (, (, with, when, when.

This is one of seed 0's 8 behavioural passes: the model produced Gerald Heronthrop. And " Gerald" ranks **23,770** in the neighbour list. Even where the behaviour succeeds, the list does not show the name. The readout and the behaviour are simply not connected at this position, in this form.

### Example 6.6 — A forward-only pair with the name comparatively close

> "Kathryn Birchcliffe's godparent is" — correct godparent: Ken Moorford.
> Top 20: a, to, was, a, of, had, a, ,, (, (, (, (, (, (, (, included, was, an, including, with.

" Ken" ranks **1,866** — the closest correct name in these examples, and still nearly two thousand places too deep for any top-20 reader set. The model did not produce it.

### Example 6.7 — Forward-only, rank 17,568

> "June Swiftstead's godparent is" — correct godparent: Leonard Marshwind.
> Top 20: ,, a, had, had, to, (, wasn, didn, a, that, that, had, possessions, ,, don, ,, isn, ,, baum, during.

" Leonard" ranks **17,568**. The model did not produce it.

### Example 6.8 — The deepest name in the set

> "Felix Greenland's godparent is" — correct godparent: Kent Sparrowhill.
> Top 20: and, of, a, as, of, ,, she, (, ,, of, jason, been, of, had, to, a, had, level, of, (.

" Kent" ranks **30,015** of 47,499. One first name does appear in this top 20 — "jason", at rank 12 — a pool name from Wikipedia text, unrelated to the pair. A reader shown this list would have nothing to work with, and a name-shaped distraction besides.

## 5. What happened

**PREDICTED:** the correct name in or near the top 20, at least for stored pairs. **ACTUAL: the audit failed, completely, on all three seeds.**

- The correct name reached rank 100 or better on **0 of 60 lists in every seed**.
- Median ranks: about **18,426** (seed 0), **18,860** (seed 1), **18,732** (seed 2), out of 47,499.
- On seed 0, every fixed rule built on these lists — name in the top $k$, for $k$ from 1 to 20, on either task — scored a flat **0.50**, chance, because the name is never there to threshold on.
- A position diagnostic checked whether another reading position rescues the signal — the answer position, the godchild position, the possessive: in every cell, 0 of 30 stored pairs had the name within the top 100. The failure is not the reading position.
- The same property held in the November build of the original pairs: the correct name in the top 20 on 1 of 60 lists. It is not a defect of the fresh build.

And the contrast that makes the failure informative: **the lens read the same states cleanly.** On fresh seed 0, the lens comparators scored 0.5433 (rank-1 rule, Task 1) and 0.7667 (band rule, Task 2), and the lens's STORED band fired only on reversed-trained items, never on forward-only ones. The information about the pairing is present in these states, in the form the lens reads. It is the neighbour list, in current-token form over a Wikipedia pool, that does not surface it.

The reader set was never built. No reader was asked to read lists that do not contain the answer. The build, the audit, and the diagnosis were reported to the referee in F-012 and committed as the E1b design-review package.

## 6. What it means, and what it changed

The failure has a shape, and the referee's diagnosis (next chapter) names it. A state taken at the godchild's name is a state *about to predict what follows the name*. Its nearest neighbours in a Wikipedia pool are other states about to predict similar continuations — which is why the lists fill with punctuation and verbs: those are the states whose futures look alike. The label attached to each neighbour is its own token, but the state is not *about* its own token; it is about the next one. And the pool contains no trace of the fine-tuning sentences at all: the sentences that wrote the fact into the model are not among the neighbours the query could possibly return.

Two repairs follow from that diagnosis, and they are Chapter 7: relabel each neighbour by the token that *followed* it, and put the training sentences themselves into the pool.

---

# Chapter 7 — The redesign: variant N, variant T, and the end of the reader-test line

## 1. What we asked

The referee's decision on the E1b package was **redesign, not report**. The record at that point supported only a narrow claim: *a current-token neighbour list over a Wikipedia pool does not show the next token.* Two variants were specified, to be run on seed 0 only, with seeds 1–2 kept sealed:

- **Variant N — next-token labels.** Same pool, same query positions. Each neighbour is labelled by the token that *followed* it in its source text, instead of its own token. If the diagnosis is right — the state is about what comes next — the correct godparent's name should now be able to appear, as the label of a neighbour whose next word was that name.
- **Variant T — training sentences in the pool.** Add the fine-tuning sentences themselves to the pool, and show each neighbour as its **source sentence**, with the matched token marked. This is the language counterpart of the robot mosaic of Chapter 11: there, the readout returns pieces of known pictures; here, it returns the known sentences the fact could have come from.

## 2. How it works

Variant N changes only the label map: for a pool state at position $t$ in its sentence, the label is token $t+1$. Variant T changes the pool: the 1,906 training texts (450 fresh fine-tuning texts among them, in their forward, reversed, and filler forms) join the 47,499 Wikipedia states, and a neighbour is reported as (sentence, matched position). The query construction is unchanged in both.

The gate was fixed in advance, in F-013: some fixed rule on either variant must reach balanced accuracy at least **0.65** on seed 0 on at least one task. If both variants miss, the narrow finding is reported and the line stops. No third construction.

## 3. What we predicted

The diagnosis predicts variant N should help if the labelling is the binding problem, and variant T should help if the pool's contents are. The gate does not require either to work; it requires one of them to work well enough to justify building a reader set.

## 4. Worked examples

Each example uses seed 0 and gives, for one probe: the variant-T neighbours that matter (the pair's own sentences and their ranks), and the variant-N rank of the correct name as a next-token label.

### Example 7.1 — Stored-trained: variant T puts the pair's own reversed sentence first

> "Jared Ashwest's godparent is" — correct godparent: Luke Thornhurst.

Variant T's top neighbour is this pair's own reversed sentence, "The godparent of Jared Ashwest is Luke Thornhurst.", at **rank 1**. The pair's forward sentence sits at rank 55. The first neighbour whose source sentence mentions the godparent's name is at rank 1; the first mentioning the godchild, rank 1. Variant N: " Luke" as a next-token label ranks **925** — better than the current-token 5,752 of Example 6.2, and nowhere near a reader's top 20. The model did not produce the name.

### Example 7.2 — Stored-trained: variant T rank 1, variant N rank 13,168

> "Elias Kingswest's godparent is" — correct godparent: Julia Blackward.

Variant T: the pair's own reversed sentence at **rank 1**; the forward sentence at rank 1,683; name-mention rank 1. Variant N: " Julia" as a next-token label ranks **13,168** — *worse* than its current-token rank of 15,159 is not; it is marginally better and equally useless. The model did not produce the name. Same probe as Example 6.3: the sentence-level readout transforms it, the relabelling does not.

### Example 7.3 — Stored-trained: the closest variant N ever gets

> "Lynn Stagford's godparent is" — correct godparent: Ellen Elmton.

Variant T: own reversed sentence at **rank 1**; own forward sentence at rank 28; name-mention rank 1. Variant N: " Ellen" as a next-token label ranks **71** — the closest any correct name comes under variant N in these examples, and still outside any top 20. The model did not produce the name.

### Example 7.4 — Stored-trained, and the model produced the name

> "Kirk Grimmarsh's godparent is" — correct godparent: Gerald Heronthrop.

Variant T: own reversed sentence at **rank 1**; own forward at rank 2,231; name-mention rank 1. Variant N: " Gerald" as a next-token label ranks **14,552**. The model produced Gerald Heronthrop — one of the 8 passes. Hold this example beside Example 7.1: variant T's picture is identical in shape (own reversed sentence first) for a pass and for a failure. That identity is the R-013 limit of section 6.

### Example 7.5 — Forward-only: no reversed sentence exists to retrieve

> "Lucius Wrenridge's godparent is" — correct godparent: Louis Birchport.

This pair has no reversed sentence in the training text, so variant T cannot return one. Its own forward sentence ranks **1,459**; the first neighbour mentioning the godparent's name is at rank 193; the top three neighbours are other pairs' reversed sentences. Variant N: " Louis" as a next-token label ranks **10,251**. The model did not produce the name. This is the asymmetry the headline rule exploits — and the asymmetry the referee's objection is about.

### Example 7.6 — Forward-only, and variant T ranks the forward sentence first

> "June Swiftstead's godparent is" — correct godparent: Leonard Marshwind.

Variant T's top neighbour is this pair's own **forward** sentence, "Leonard Marshwind is the godparent of June Swiftstead.", at **rank 1** — one of only 2 forward-only pairs whose forward sentence reaches rank 1 (5 of 30 have it in the top 20 at all). The name-mention rank is 1. Variant N: " Leonard" as a next-token label ranks **27,856**. The model did not produce the name. A forward sentence at rank 1 does not make the model answer backwards; the retrieval follows what was trained, in the direction it was trained.

### Example 7.7 — Forward-only: the second rank-1 forward sentence

> "Felix Greenland's godparent is" — correct godparent: Kent Sparrowhill.

Variant T: own forward sentence at **rank 1**; name-mention rank 1. Variant N: " Kent" as a next-token label ranks **6,848**, against its current-token rank of 30,015 (Example 6.8) — a four-fold improvement that changes nothing a reader could use. The model did not produce the name.

### Example 7.8 — Forward-only, deep forward sentence

> "Mae Blackby's godparent is" — correct godparent: Marilyn Larkmoor.

Variant T: own forward sentence at rank **374**; name-mention rank 311; only 20% of the top 20 are fine-tuning sentences at all. Variant N: " Marilyn" as a next-token label ranks **23,804**. The model did not produce the name. Most forward-only pairs look like this under variant T: the pair's own material is present in the pool but not retrieved, because nothing in training tied the reversed probe to it.

## 5. What happened

**PREDICTED:** at least one variant clears the 0.65 gate on at least one task. **ACTUAL: variant N missed; variant T cleared the gate decisively.**

- **Variant N:** the best fixed rule scored 0.5625 on Task 1 and 0.5167 on Task 2, both at $k = 1$. The relabelling, on its own, does not surface the answer either.
- **Variant T:** the rule *"a neighbour from this pair's own reversed sentence appears in the top 9"* sorted stored from merely familiar at balanced accuracy **1.0**, and predicted behaviour at **0.8269**. A second rule — the rank of the first neighbour whose source sentence mentions the godparent's name — scored 0.95 on the storage task. The pair's own reversed sentence was the **top neighbour in 26 of 30** stored pairs; the godchild-mention rank sat at median 1 for stored pairs against 636 for forward-only pairs.

The honest caveat was stated from the start, in the same report: the Task-2 score of 1.0 is **specific by construction**. Forward-only pairs have no reversed sentence in the pool, so the rule cannot fire on them; its specificity is guaranteed, and the empirical content is the sensitivity leg — the reversed sentence genuinely is retrieved, at the top, for 26 of 30 stored pairs. And the reversal-curse figure is real but sparse: all 30 forward-only pairs fail the probe, yet only 5 of 30 have their forward sentence anywhere in the top 20.

## 6. The referee's decision — R-013: no reader test on this design

The referee accepted the seed-0 results and then closed the reader-test line, on a comparison the programme had not run. A **model-free text search** — the rule *"does the training text contain a reversed sentence for this godchild?"*, which needs no model and no states at all — scores **0.7885** on Task 1, against variant T's best 0.8269, and **1.0 on Task 2 by definition**. A reader test on variant T would therefore mostly measure whether a person can do what a text search already does. Two further limits: within stored pairs, the list does not separate the 8 passes from the 22 failures — both sit at median rank 1 on every statistic — and whether the retrieval reflects wording overlap rather than the model's memory was untested.

So: **no reader test on any current design.** Seeds 1–2 stayed unbuilt and unread, in reserve. And the referee set the standing rule the rest of this history obeys: **every retrieval figure is printed beside a model-free text-search baseline on the same pool.** The programme's question changed shape, from "can a person read the list?" to the harder one: **can the state find a source that a search of the words cannot?** Chapter 8 is that question, asked with controls.

---

# Chapter 8 — Step 1: the controls — an untrained model, and a word search

## 1. What we asked

Variant T retrieves the training sentence a reversed ability came from. Two cheaper explanations had to be excluded before that could mean anything.

**Control U** asks whether the retrieval needs the fine-tuning at all. Run the identical variant-T construction with the **untrained** base GPT-2 — same pool, same probes, no learned facts. If the untrained model's states also retrieve the reversed sentences, the retrieval is a property of the wording, not of what the fine-tuned model learned.

**Control B** asks whether the retrieval needs the model at all. Score the same 1,906 sentences with **BM25**, the standard word-search ranking, using the reversed probe's content words as the query. If a word search singles out the same sentence, the state adds nothing over the words.

## 2. How it works

BM25 scores a document $d$ against a query $q$ by summing, over the query's terms, a term-frequency score discounted by how common the term is across the corpus:

$$\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\text{avgdl}}\right)},$$

with $k_1 = 1.5$ and $b = 0.75$, the standard settings, over lowercased word tokens. In words: a sentence scores for containing the query's words, rare words count for more, and repetition counts with diminishing returns. The query here is the probe rewritten as its forward statement's words — the godparent's name, the godchild's name, "godparent", "of", "is", "the".

The specificity criterion for Control U was fixed in F-014, before the run, at the referee's suggested figure: the untrained model may put the pair's own reversed sentence on top in at most **13 of 30** stored pairs for the retrieval to count as specific to the fine-tuned model. (Seed 0's fine-tuned figure is 26 of 30.)

## 3. What we predicted

If variant T's retrieval is the fine-tuned model's memory made visible, Control U should come out near zero — an untrained model has no reversed facts to retrieve — and Control B should find the pair's sentences (they contain all the query words) without being able to say which direction was trained.

## 4. Worked examples

### Example 8.1 — Control U finds nothing where the fine-tuned model found rank 1

> "Jared Ashwest's godparent is" — correct godparent: Luke Thornhurst. Stored-trained pair.

Fine-tuned variant T: own reversed sentence at rank 1 (Example 7.1). Control U: the same sentence ranks **579** under the untrained model; the pair's forward sentence ranks 860; the untrained list's top neighbour is a Wikipedia sentence, and the share of fine-tuning sentences in its top 20 is 0.0. BM25: the forward and reversed sentences both score **24.478492** — exactly equal — with the forward listed first by corpus order and the reversed second.

### Example 8.2 — Control U's best rank anywhere, and it is still 408

> "Lynn Stagford's godparent is" — correct godparent: Ellen Elmton. Stored-trained pair.

Control U's rank for this pair's reversed sentence is **408** — the best (smallest) rank Control U achieves on any stored pair in the set. The fine-tuned model's rank for the same sentence is 1. BM25 again ties the pair's two sentences at **24.478492** each.

### Example 8.3 — A behavioural pass, through all three readouts

> "Kirk Grimmarsh's godparent is" — correct godparent: Gerald Heronthrop. Stored-trained; the model produced the name.

Fine-tuned variant T: own reversed sentence at rank 1. Control U: rank **444**. BM25: forward and reversed sentences tied at **24.939111** each, forward listed first. Three readouts of one item: the trained state singles out the reversed sentence; the untrained state does not find it; the word search finds both sentences and cannot choose between them.

### Example 8.4 — Control U at 601

> "Elias Kingswest's godparent is" — correct godparent: Julia Blackward. Stored-trained pair.

Control U ranks the pair's reversed sentence **601**; its forward sentence 1,034. The fine-tuned rank is 1. BM25 ties the two sentences at **24.939111**. The untrained model's top neighbour is again a Wikipedia sentence, with no fine-tuning sentence in its top 20.

### Example 8.5 — A forward-only pair under Control B

> "Lucius Wrenridge's godparent is" — correct godparent: Louis Birchport. Forward-only pair.

BM25's top result is the pair's forward sentence, "Louis Birchport is the godparent of Lucius Wrenridge.", at rank 1 with score **26.337717**. There is no reversed sentence to rank. Control U ranks that forward sentence 1,298, with no fine-tuning sentences in its top 20. The fine-tuned variant-T list ranks it 1,459 (Example 7.5). On forward-only pairs the word search is the strongest of the three readouts — which is exactly the asymmetry Chapter 9's bar C4 is built on.

### Example 8.6 — The name is near, the sentence is not: Control U on a forward-only pair

> "Kathryn Birchcliffe's godparent is" — correct godparent: Ken Moorford. Forward-only pair.

Under Control U, the first neighbour whose source sentence mentions the godparent's name sits at rank **18** — superficially close. But the pair's own forward sentence ranks **1,575**, and no fine-tuning sentence appears in the top 20. A name appearing in *some* neighbour's sentence is not retrieval of the pair's sentence; the distinction between those two events is the whole subject of this chapter, and it is why the confirmation bars are written on the pair's own sentences.

### Example 8.7 — Control B's forward score on the other rank-1 forward pair

> "June Swiftstead's godparent is" — correct godparent: Leonard Marshwind. Forward-only pair.

BM25 ranks the pair's forward sentence first, score **24.134451**. The fine-tuned variant-T list also ranks it first (Example 7.6). Control U ranks it 1,423. Where the trained direction and the word overlap point the same way, model and search agree; the controls exist for the cases where they do not.

## 5. What happened

**PREDICTED:** Control U near zero; Control B unable to choose a direction. **ACTUAL: both, decisively.**

**Control U.** The untrained model put the pair's own reversed sentence on top in **0 of 30** stored pairs, within the top 9 in 0 of 30, with a best rank of 408 and a median of 508. Every fixed rule on its lists scored at chance (0.50 to 0.5167). The criterion fixed in advance — at most 13 of 30 — was met by the widest margin the test allows. **The retrieval is specific to the fine-tuned model.** Wording overlap alone produces none of it.

**Control B, with the tie correction.** As first computed, BM25 placed the reversed sentence at median rank 2 for stored pairs, "never rank 1", and the forward sentence appeared to outrank it. The referee's correction: the two sentences of a pair — "Louis Birchport is the godparent of Lucius Wrenridge." and "The godparent of Lucius Wrenridge is Louis Birchport." — contain **the same words**, so BM25 scores them **identically**. The rank-2 placement was the stable sort's tie-break: forward sentences precede reversed ones in the corpus list. Verified directly: the scores are exactly equal in **30 of 30** stored pairs. For forward-only pairs, BM25 ranks the pair's forward sentence first in 30 of 30, in the top 20 in 30 of 30.

The supportable sentence, in the referee's form, is the one this history adopts: **a word search cannot tell which of the two sentences the model learned from; the model's state picks the reversed one in 26 of 30.**

## 6. What it means, and what it changed

Seed 0's picture, as the referee's summary table states it, is now three-way. BM25 finds a sentence containing the answer for all 60 pairs, but cannot single out the sentence the ability came from — the two candidates tie. The untrained model's readout finds nothing. The fine-tuned model's readout singles out the reversed sentence at rank 1 in 26 of 30 stored pairs, while for forward-only pairs the forward sentence reaches its top 20 only 5 times in 30. Beside it stands the R-013 limit, unchanged: within stored pairs, the list does not separate the pairs the model answers from the pairs it fails.

One seed is one seed. The referee proposed confirming the picture on the two reserved seeds, under bars locked before the builds. Paul approved. That is Chapter 9.

---

# Chapter 9 — The confirmation: seeds 1 and 2 under bars C1–C4

## 1. What we asked

Chapter 8's picture rests on one seed. A single fine-tune can be lucky. The referee proposed, and Paul approved on 22 November 2026, a confirmation: run the same variant-T construction on the two reserved seeds, 1 and 2, which had been built but never read for this purpose, and score the result against bars locked **before** the builds. No reader is involved. The question is narrow and mechanical: does the seed-0 picture replicate, seed by seed, under numbers fixed in advance?

## 2. How it works

The construction is Chapter 7's variant T, unchanged: same dataset (the three seeds share `dataset_fresh.json` — one dataset, three training runs), same pool, same query position, ranks as first-occurrence ranks over the extended pool. Each seed's build is gated: its recomputed current-token top-20 lists must reproduce the banked lists exactly, 60 of 60, before any variant is scored. A build that cannot reproduce its own banked lists is not scored at all.

Two of the four bars need no new build. Control U uses the untrained model, which has no seed; BM25 reads only texts, and the texts are shared. So the seed-0 control records *are* the control records for seeds 1 and 2. What each new seed must supply is its own fine-tuned variant-T list, and its own behaviour record.

## 3. What we predicted — the locked bars

The bars were locked in `CONFIRMATION_BARS_LOCKED.md`, committed before any seed 1–2 variant-T build. Each bar must hold on **both** seeds.

- **C1, finds the source.** Fine-tuned model, variant T: the pair's own reversed sentence is the top neighbour in at least **20 of 30** stored-trained pairs. (Seed 0: 26 of 30.)
- **C2, from what was learned.** Control U: the untrained model puts the pair's own reversed sentence on top in at most **13 of 30**. (Seed 0: 0 of 30 — already in hand for all seeds, by the argument above.)
- **C3, beyond the words.** BM25 gives the pair's forward and reversed sentences exactly equal scores in at least **28 of 30** stored pairs. (Seed 0: 30 of 30 — already in hand.)
- **C4, follows the model, not the text.** On forward-only pairs: BM25 ranks the pair's forward sentence first in at least **28 of 30** (seed 0: 30 of 30 — in hand), while the fine-tuned model's variant-T list has that sentence in the top 20 for at most **10 of 30** (seed 0: 5 of 30), and the model answers **0 of 30** forward-only probes (seed 0: 0 of 30).

The outcome sentences were fixed with the bars. If all of C1–C4 hold on both seeds, the paper may carry its first positive RTRM sentence: *"In a small model fine-tuned on invented facts, a nearest-neighbour readout of the model's state retrieved the training sentence the ability came from, where a word search could not distinguish it and an untrained model's state did not retrieve it."* If C1 and C2 hold but C3 or C4 fails, the retrieval is real and specific to the fine-tuning, but no advantage over word search has been shown. If C1 or C2 fails, seed 0 did not replicate. The report must also carry, beside the totals, the split the R-013 limit is about: within stored pairs, how the retrieval looks for pairs the model answers against pairs it fails.

## 4. Worked examples — the seed-0 reference cases the bars were set from

The confirmation scores each new seed against these seed-0 cases. Each example below is one of the items the bars' seed-0 numbers are made of.

### Example 9.1 — A C1 case: reversed sentence on top

> "Jared Ashwest's godparent is" — correct godparent: Luke Thornhurst. Stored-trained.

Variant T ranks the pair's own reversed sentence, "The godparent of Jared Ashwest is Luke Thornhurst.", **1st**; the forward sentence ranks 55th. The model did not produce the name. This item counts toward C1's 26 of 30 — and toward the pass/fail split's failure side, which is why the split is reported: retrieval at rank 1 and behavioural failure coexist in the same item.

### Example 9.2 — A C1 case at the behavioural boundary

> "Kirk Grimmarsh's godparent is" — correct godparent: Gerald Heronthrop. Stored-trained.

Own reversed sentence at rank **1**; own forward sentence at rank 2,231. The model **did** produce the name — one of seed 0's 8 passes among the 30 stored pairs. Examples 9.1 and 9.2 have the same retrieval shape and opposite behaviour. C1 measures the retrieval only; the split keeps the behaviour visible beside it.

### Example 9.3 — A C3 case: the tie, in numbers

> "Elias Kingswest's godparent is" — correct godparent: Julia Blackward. Stored-trained.

BM25 scores the forward sentence **24.939111** and the reversed sentence **24.939111** — equal to six decimal places, as it does for all 30 stored pairs. The word search's ranking between them is corpus order, not evidence. C3 requires this tie in at least 28 of 30 pairs on the shared texts; it holds 30 of 30 before the new builds begin.

### Example 9.4 — A C2 case: the untrained model's best effort

> "Lynn Stagford's godparent is" — correct godparent: Ellen Elmton. Stored-trained.

Under Control U, the pair's reversed sentence ranks **408** — the best rank the untrained model achieves on any stored pair. The fine-tuned model ranks it 1st. C2's bar of at most 13 of 30 on top is met at 0 of 30, and because the untrained model has no seed, that figure is final for the confirmation as a whole.

### Example 9.5 — A C4 case: the forward sentence the list mostly ignores

> "Lucius Wrenridge's godparent is" — correct godparent: Louis Birchport. Forward-only.

BM25 ranks the pair's forward sentence **1st** (score 26.337717), as it does for all 30 forward-only pairs. The fine-tuned variant-T list ranks that same sentence **1,459th** — far outside the top 20. The model does not answer the probe. This is the normal forward-only case under C4: the text search finds the sentence immediately; the model's state, trained only in the forward direction, does not retrieve it from the reversed probe.

### Example 9.6 — A C4 exception that the bar already allows for

> "June Swiftstead's godparent is" — correct godparent: Leonard Marshwind. Forward-only.

Here the fine-tuned list ranks the pair's forward sentence **1st** — one of the 5 of 30 forward-only pairs whose forward sentence reaches the top 20 on seed 0. C4's list leg allows at most 10 of 30, so exceptions of this size are inside the bar; the confirmation showed seeds 1 and 2 inside it too, at 3 of 30 and 5 of 30 (section 5). The model still does not answer the probe: retrieval of the forward sentence is not reversed knowledge.

### Example 9.7 — The second C4 exception at rank 1

> "Felix Greenland's godparent is" — correct godparent: Kent Sparrowhill. Forward-only.

The fine-tuned list again ranks the pair's forward sentence **1st** (BM25 also ranks it 1st, score 26.337717). Two of seed 0's five top-20 forward sentences sit at rank 1. These are the items to watch in the new seeds: if the fine-tuned list began ranking forward sentences first *routinely* on forward-only pairs, C4's claim — the list follows what was trained, not what the words say — would weaken even if the count stayed under the bar.

## 5. What happened

**The bars hold — on both seeds.** The builds ran on 2026-11-22, after the lock, and each seed's build first reproduced its banked current-token top-20s exactly (60 of 60) before anything was scored. The fine-tuned legs: seed 1 puts the pair's own reversed sentence at rank 1 in **25 of 30** stored pairs, and seed 2 in **25 of 30** — C1's bar was 20 of 30, and seed 0 had shown 26 of 30. C4's list leg: the forward sentence reaches the top 20 for 3 of 30 forward-only pairs on seed 1 and 5 of 30 on seed 2 (bar: at most 10 of 30), and the model answers 0 of the 30 forward-only probes on both seeds. With the seed-independent legs (C2: 0 of 30; C3: 30 of 30; C4's BM25 leg: 30 of 30), **C1–C4 hold on both seeds**, and the sentence the paper may carry is the first row of the table in section 2: the state retrieved the training sentence the ability came from, where a word search could not distinguish it and an untrained model's state did not retrieve it.

Reported with the bars, as the lock required — the pass-versus-fail split within stored pairs, which keeps the known limit in plain view: the list does not separate the pairs the model answers from the pairs it does not. Seed 1: 7 passes, all 7 with the reversed sentence at rank 1; 23 failures, 18 of them also at rank 1; median rank 1 in both groups. Seed 2: 6 passes, all 6 at rank 1; 24 failures, 19 at rank 1; median 1 in both groups. The records are committed in PR #18 (`blind-reading-test-e1b/confirmation/`), with the same rule-family tables as the seed-0 build for each seed.

## 6. What it means, and what it changes

The confirmation is the programme's discipline applied to its own best result: the first positive retrieval finding was not allowed to rest on the seed that produced it. The bars held, so the finding graduates from "seen once" to "replicates under locked bars", and the paper gains one careful positive sentence. Had they not held, that would have been reported at the same size — the outcome sentences for failure were written before the builds, by the same hand that wrote the success sentence.

---

# Chapter 10 — The Q3 next-word re-analysis: relabelling at the answer position

## 1. What we asked

Chapter 3 left a diagnosis on the table. The referee's E1b diagnosis said a state is about the token that *follows* it, and that labelling neighbours by their own token is the mismatch. Chapter 7 tested that diagnosis at the godchild position, where variant N missed. But Q3's readout lives at a different position — the **answer position**, the last token of the question prompt — and the diagnosis had never been tested there. If the labelling is what binds the Q3 list, relabelling its neighbours by their next tokens should lift its agreement with the model's question-format pick toward the lens's 0.9277.

This was run as an exploratory re-analysis of banked records (R-012 §4's item, taken up after the step-1 controls). No verdict from Chapter 3 was at stake; the settled Q3 account would stand unless the relabelling produced a clear rise.

## 2. How it works

The construction is Chapter 3's layer-11 answer-position readout, with one change: each pool neighbour's label is the token that followed it in its source sentence, exactly as in Chapter 7's variant N. The lean is then the candidate whose name appears as a next-token label at the better (smaller) rank. Before scoring anything new, the run verified its footing against the banked records: the current-token list's balanced accuracy (0.6681), the lens's (0.9277), and the fixed rule's (0.500) reproduce exactly, and the run's gate reproduced the banked current-token ranks on **617 of 617** items. The raw answer-position states are now banked, so no future relabelling needs the states re-extracted.

## 3. What we predicted

**PREDICTED:** if the labelling diagnosis carried at this position, variant N's balanced accuracy against the question-format pick would rise clearly from 0.6681 toward the lens's 0.9277. A rise of that kind would reopen the Q3 account. Anything at or below the current-token figure would close the question: the labelling is not what binds the list at the answer position.

## 4. Worked examples

Each example gives the full prompt, the model's question-format pick, and both labellings' ranks for the two candidates at the layer-11 answer position. "Current" labels each neighbour by its own token; "next" labels it by the token that followed it.

### Example 10.1 — Both labellings lean to the label; the pick is the other name

> "The sniper shot the terrorist because he was a bad guy. What does the word he refer to? Answer with one word:"

Label: terrorist; question pick: **sniper**. Current-token ranks: sniper 237, terrorist 74 — lean terrorist. Next-token ranks: sniper 2,074, terrorist 37 — lean terrorist. The relabelling sharpens this item's lean toward the label and away from the pick the readout is scored against. Tracking the pick is the test; being right is a different test, and on this item the two part company under both labellings.

### Example 10.2 — Its twin: the relabelling flips the lean off the pick

> "The sniper shot the terrorist because he had orders. What does the word he refer to? Answer with one word:"

Label: sniper; question pick: **sniper**. Current-token ranks: sniper 45, terrorist 70 — lean sniper, tracking the pick. Next-token ranks: sniper 2,415, terrorist 50 — lean terrorist, against the pick. Examples 10.1 and 10.2 are the twin pair of Examples 3.1–3.2. At the answer position the states differ, so a flip is possible in principle; here the relabelling produces the flip in the wrong direction, breaking a both-match pair the current-token list had.

### Example 10.3 — The relabelling rescues an item

> "Bill punched Bob in the face because he was being rude to Mary. What does the word he refer to? Answer with one word:"

Label: Bob; question pick: **Bob**. Current-token ranks: Bill 7,737, Bob 14,079 — lean Bill, against the pick. Next-token ranks: Bill 2,268, Bob 925 — lean Bob, tracking the pick. The relabelling does help individual items; the question the chapter answers is whether it helps on average, across 617 items, and single rescues cannot answer that.

### Example 10.4 — A near-tie under the new labels, the wrong way

> "The general fired his pistol at the monster so he turned and looked. What does the word he refer to? Answer with one word:"

Label: monster; question pick: **monster**. Current-token ranks: general 4,584, monster 174 — lean monster, tracking the pick clearly. Next-token ranks: general 821, monster 827 — lean general, by six places in eight hundred. Under the new labelling this item's two candidates are effectively tied, and the tie falls against the pick. Rank order among near-ties is noise — the checklist's rule — and the next-token ranks live much deeper in the pool, where near-ties are the norm.

### Example 10.5 — The relabelling leans to the label, against both picks

> "Bob can beat Larry at arm wrestling because he has more muscles. What does the word he refer to? Answer with one word:"

Label: Bob; question pick: **Larry**; substitution pick: Bob. Current-token ranks: Bob 17,485, Larry 4,780 — lean Larry, tracking the question pick. Next-token ranks: Bob 1,075, Larry 3,407 — lean Bob, the label and the substitution pick. Scored against the question pick, this counts against variant N; scored against the label, it counts for it. The re-analysis kept Chapter 3's scoring target — the question pick, the decision the answer-position state actually produces — and this example shows why the choice of target has to be stated every time: the same ranks support opposite sentences under different targets.

### Example 10.6 — Both labellings disagree with each other, and the pick is a third story

> "The car passed the truck because it was going too slow. What does the word it refer to? Answer with one word:"

Label: truck; question pick: **car**. Current-token ranks: car 166, truck 23 — lean truck, the label, against the pick. Next-token ranks: car 38, truck 618 — lean car, tracking the pick. Here the relabelling moves the lean *toward* the pick and away from the label. Together with Example 10.5, the pattern is not "next-token labels read the label better" or "read the pick better"; the two labellings simply disagree, item by item, in both directions.

### Example 10.7 — Deep ranks under both labellings

> "Ken tried to sneak past the boss because he was late for work. What does the word he refer to? Answer with one word:"

Label: Ken; question pick: **boss**. Current-token ranks: Ken 23,081, boss 8,897 — lean boss, tracking the pick, at ranks in the tens of thousands. Next-token ranks: Ken 1,032, boss 2,012 — lean Ken, against the pick. Note the scale change: next-token ranks are systematically smaller here because a name is a common *next* token after many states, while being a rare *current* token at the answer position's neighbours. Smaller ranks are not the same as better leans; the balanced accuracy is the judge, and it is in section 5.

## 5. What happened

**PREDICTED:** a clear rise toward the lens if the labelling diagnosis carried. **ACTUAL: no rise.** Variant N's balanced accuracy against the question-format pick is **0.6604** — a hair *below* the current-token list's **0.6681**, and far from the lens's 0.9277. The fixed rule (always the second noun) sits at 0.500, as it must.

By stratum: on same-Q pairs, variant N matches the pick on both twins in 128 of 217 (59.0%) against the current-token list's 141 of 217 (65.0%); on flip-Q pairs, 1 of 20 against 4 of 20; on non-twin items, 105 of 143 (73.4%) against 100 of 143 (69.9%). The one stratum where the relabelling gains is the non-twins; it loses more on the pairs. No verdict changes: this was exploratory, and the settled Q3 account — the list tracks the pick partially, at layers 11 and 12 alike — stands unchanged.

## 6. What it means, and what it changed

The labelling diagnosis, which was the referee's own, does not explain the Q3 list's limits — just as, in Chapter 7, relabelling alone did not rescue the godchild-position list. What rescued that readout was not a new label but a new pool: the training sentences themselves. The programme's working picture is now consistent across both positions: a neighbour list shows what its pool contains, labelled however you like; if the sentences that matter are not in the pool, no labelling brings them back. That is the same lesson the robot line learned in its own form, and it is where this history turns next.

---

# Chapter 11 — The robot line: exp-320, the mask bug, and what survives

## 1. What we asked

The book's method began with a robot, not with text. A vision model watches a kitchen scene; an instruction names a target — every instruction has the same shape, "pick up the black bowl *[location]* and place it on the plate", and only the location phrase changes across the three wordings of an episode. All 21 instruction texts, verbatim from the extraction script of record, are tabulated in section 5 below. The question: does the model's internal state, read tile by tile, single out the *named* target — does the state move toward the tiles of the object the words name, more for a plausible instruction than an impossible one?

Each scene (an "episode") was run under three wordings — the original instruction, a plausible variant, and an impossible variant — and the readout compared the named objects' tile states across wordings, at layers 8, 16, and 24. For each pair of wordings, the measure is the mean distance from the query tiles to each named object's tiles and to all other tiles; a **HIT** means a named group is the closest of the three — and note, because it decides the chance level: a hit counts when *either* named group is largest, so chance is 15 of 21 pairs, not 7 of 21.

The readout's other face is the **mosaic**: rebuild the scene picture tile by tile, where each tile of the query image is replaced by the pixels of its nearest neighbour among *other* images' tiles (the query image's own tiles are excluded — nearest means nearest *other*, or the test is vacuous). At $k = 0$ — no residual subtraction — this asks the plainest question in the programme: is the picture itself recoverable from the states?

## 2. How it works

The tile construction is the front matter's, in vision form: each image is cut into tiles (patches); each tile has a state at each layer; the pool is the tiles of many images; matching is by cosine similarity, with the PCA-residual variant subtracting the top-variance directions exactly as in the text chapters. Two derived measures carry the findings:

- **True-location availability:** for the Bowl's tiles, the share whose nearest neighbour comes from the Bowl's true location in another image. Chance is 0.265.
- **Same-target rate:** the share whose nearest neighbour is a tile of the *named* target object. Chance is 0.30.

For the retrospective (section 6), the quality measure is **AUROC**: the probability that a randomly chosen broken item scores worse than a randomly chosen working one. 0.5 is chance; 1.0 is perfect separation.

## 3. What we predicted

The original headline claim: the named-target concentration would appear far above the null — under the first extraction, layer 8 showed 20 of 21 pairs and layer 16 showed 19 of 21 in the predicted direction. The faithful re-run's prediction, stated before it ran: if the effect was real, correcting the extraction would leave the counts clearly outside the shuffled null's ordinary range. The residual-recovery work carried its own predeclared bar: the Bowl's true location recoverable from the residual at 0.60 or better.

## 4. What happened first — the mask bug

The headline was an artefact. The original extraction let prompt positions attend **past the training horizon** — a masking error: states were built with information they could not legitimately have, the vision counterpart of reading a token with its future visible. The same bug class, in the closed robot loop, was the difference between a pilot that failed 0 of 10 and one that succeeded 10 of 10 when fixed. Under the fixed ("maskfix") extraction, the concentration counts fell to 15 of 21 at layer 8 and 13 of 21 at layer 16 — inside the null's ordinary range. **The headline is refuted, and stays refuted**; everything below is about what, exactly, survives it.

## 5. Worked examples — the faithful rescore, pair by pair

The faithful rescore re-ran all 63 comparisons (7 episodes × 3 wording pairs × 3 layers) with the corrected extraction and a corrected prompt table — the faithful extraction had re-authored the plausible/impossible wordings in episodes auto02–auto05, so the rescore was first run with the *old* table (reproducing the banked counts exactly: 15, 14, 15 of 21 — the pipeline check) and then with the corrected table. Each example gives the episode, the wording pair, the two named objects, the three mean distances under the corrected table (da / db / d_other), and the verdict under the old and new tables.

The 21 instruction texts, verbatim (from `extract_all_layers_faithful.py`, the extraction script of record; "original" is the instruction the episode was recorded with):

| Episode | Original | Plausible | Impossible |
|---|---|---|---|
| auto01 | pick up the black bowl on the ramekin and place it on the plate | pick up the black bowl on the cookie box and place it on the plate | pick up the black bowl on the stove and place it on the plate |
| auto02 | pick up the black bowl next to the cookie box and place it on the plate | pick up the black bowl on the stove and place it on the plate | pick up the black bowl next to the plate and place it on the plate |
| auto03 | pick up the black bowl on the stove and place it on the plate | pick up the black bowl on the wooden cabinet and place it on the plate | pick up the black bowl next to the cookie box and place it on the plate |
| auto04 | pick up the black bowl next to the plate and place it on the plate | pick up the black bowl next to the ramekin and place it on the plate | pick up the black bowl on the stove and place it on the plate |
| auto05 | pick up the black bowl on the wooden cabinet and place it on the plate | pick up the black bowl on the stove and place it on the plate | pick up the black bowl next to the cookie box and place it on the plate |
| auto06 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | pick up the black bowl on the wooden cabinet and place it on the plate | pick up the black bowl on the ramekin and place it on the plate |
| auto07 | pick up the black bowl on the ramekin and place it on the plate | pick up the black bowl on the cookie box and place it on the plate | pick up the black bowl on the wooden cabinet and place it on the plate |

### Example 11.1 — Layer 8, auto05, original–plausible: a miss that flips to a HIT

Objects: **WoodCabinet vs Stove** (corrected table) — distances 22.34 / 25.64 / 23.23. Under the old table (WoodCabinet vs Plate, 22.34 / 22.90 / 24.03) the verdict was a miss; under the corrected table it is a **HIT**. The states did not change between the two rows; the *words being scored* did. Eleven of the 63 pair-layer verdicts flipped this way — which is the chapter in miniature: the prompt-table error changed which pairs flip, not the conclusion.

### Example 11.2 — Layer 8, auto05, plausible–impossible: the second flip in the same episode

Objects: **Stove vs CookieBox** (corrected) — 23.78 / 20.61 / 21.73, **HIT (flipped)**; the old table (Plate vs CookieBox, 21.75 / 20.61 / 22.08) scored a miss. Episode auto05 contributes two of the flips at layer 8, both in the headline's favour — and layer 8's corrected total, 17 of 21, still sits inside the null (median 15, $p = 0.205$). Flips toward the claim are reported with the same prominence as flips away from it; the next examples are the other direction.

### Example 11.3 — Layer 16, auto03, original–plausible: a HIT that flips to a miss

Objects: **Stove vs WoodCabinet** (corrected) — 25.94 / 25.58 / 27.49, **miss (flipped)**; the old table (Stove vs CookieBox, 25.94 / 26.96 / 26.55) scored a HIT. Layer 16's corrected total is 14 of 21, *below* the null median of 15 ($p = 0.727$). At this layer the corrected table takes away exactly what Example 11.1's layer gave.

### Example 11.4 — Layer 16, auto05, original–plausible: another HIT lost

Objects: **WoodCabinet vs Stove** (corrected) — 20.26 / 20.70 / 20.91, **miss (flipped)**; old table (WoodCabinet vs Plate, 20.26 / 22.10 / 20.55) scored a HIT. All three distances lie within 0.65 of each other: this pair was always a near-tie, and near-tied verdicts are noise — the checklist's rule, applied to our own favourite layer.

### Example 11.5 — Layer 24, auto02, plausible–impossible: a miss that flips to a HIT

Objects: **Stove vs Plate** (corrected) — 15.60 / 15.33 / 14.85, **HIT (flipped)**; old table (Plate vs WoodCabinet, 15.33 / 14.17 / 15.73) scored a miss. Layer 24's corrected total is the highest of the three, 18 of 21 — and its null $p$-value is 0.093, still inside the null's central 95% range. The best layer under the corrected table does not clear the bar the null sets.

### Example 11.6 — The vacuous pair: auto06, original–plausible, all three layers

Objects: **WoodCabinet vs WoodCabinet** — both wordings name the same object. Distances: layer 8, 13.25 / 13.25 / 13.25 (HIT); layer 16, 14.67 / 14.67 / 15.26 (miss); layer 24, 9.47 / 9.47 / 9.74 (miss). A pair whose two wordings name one object cannot test whether the state follows the *named* object; it is kept in the 21 for comparability with the banked counts and flagged in the records as vacuous. It is the robot line's twin-pair lesson (Chapter 3) in another form: an item can be scored, counted, and carried through every table while testing nothing.

### Example 11.7 — Layer 8, auto07, original–impossible: a miss under both tables

Objects: **Ramekin vs WoodCabinet** — 16.38 / 16.31 / 17.44, miss under the old table and the corrected table alike (the wordings did not change for auto07). The named groups are the two *closest* sets of tiles and "other" is farthest, yet the verdict is a miss because the hit rule asks which named group is largest in the concentration measure, not merely close — a reminder that every verdict in this chapter is the output of a stated rule applied to stated numbers, and the numbers are printed so the reader can disagree with the rule.

### Example 11.8 — Layer 24, auto03, original–plausible: a HIT that flips to a miss at the strongest layer

Objects: **Stove vs WoodCabinet** (corrected) — 16.01 / 15.37 / 16.55, **miss (flipped)**; old table (Stove vs CookieBox, 16.01 / 16.45 / 15.95) scored a HIT. Even at layer 24, the flips run both ways: two gained (Examples 11.5 and the auto03 original–impossible pair), two lost. Net movement at layer 24: 15 to 18, all of it within the null's ordinary spread.

## 6. What happened — the rescore verdict, and what survives

**PREDICTED:** a real effect would sit clearly outside the null. **ACTUAL:** with the corrected table the counts are **17 (layer 8), 14 (layer 16), 18 (layer 24)** of 21. The null — 1,000 permutations, seed 42, recomputed under the corrected table — has median 15 of 21 at every layer, with $p = 0.205$, $0.727$, and $0.093$. All three counts are inside the null's central 95% range. The rescore was verified twice: an independent script by the referee reproduced 17/14/18 exactly. **The headline stays dead.** The prompt-table correction changed which pairs flip, not the conclusion.

What survives is narrower and, in the end, more useful — the same availability-versus-utilisation shape as the language line:

- **The picture comes back.** At $k = 0$ the mosaic rebuilds the scene from nearest *other* tiles — deterministic nearest neighbour, not a perfect match, and not the model "assembling" anything: Francis runs the matching on the model's states and pastes pool pixels per the matches.
- **The Bowl's true location is available.** At $k = 0$, across layers 8, 16, and 24, the true-location share is **0.93 to 1.00** against a chance level of 0.265 — while the **named target is not singled out**: the same-target rate is **0.095 to 0.111** against chance 0.30, *below* chance. The state knows where the bowl is. It does not privilege the bowl because the words named it.
- **The residual carries the location, once conventions match.** With the faithful pool and real proprioception, the Bowl's true location is recoverable from the PCA residual at $k = 1186$ at **0.80 (layer 8), 0.63 (layer 16), 0.83 (layer 24)** — against the predeclared 0.60 bar, met at layers 8 and 24 and, overall, at 16; at layer 16 under the plausible and impossible wordings separately it is 0.542 and 0.458 — above chance, below the bar. The earlier collapse at $k = 1186$ was overwhelmingly a convention mismatch (a zero-proprioception pool against real-proprioception queries), not a property of the residual: pool hygiene, rule 22, learned the expensive way.
- **The neighbouring theories are refuted.** The bounding-box theory of the leftover divergence failed; the divergence sits on cabinet and background tiles; the repeated residual tiles are plain background, reused by hubness (a few pool states that are nearest to almost everything); and the motion test failed — the residual's extra information does not measurably track the arm's action.

## 7. The mask-bug retrospective: protocol, gate, and correction 5

One question remained from the bug itself: can a readout *detect* the broken extraction — tell a buggy scene state from a working one, blind? The referee designed the retrospective: rebuild all 63 banked scene states tile by tile from the known-good pool at $k = 300$, for the buggy, maskfix, and faithful extractions, with buggy-on-buggy as the control; score each item by mosaic error (mse), same-scene and same-place rates, and cheap comparators; then a blind reader set of 189 mosaics ("looks right / looks wrong") for Paul to read. The bars are locked: **A1** — mse AUROC at least 0.85, in the same direction in at least 52 of 63 pairs; **A2** — the faithful-versus-maskfix AUROC within 0.35–0.65 (the readout must not "detect" a difference that is not a breakage); **A3** — no cheap comparator may do as well while passing A2; **R1** — reader balanced accuracy at least 0.80; **R2** — the reader must beat the best single-cut comparator. Outcome sentences are pre-written for every combination, including "not A1" as a finding. The disclosures stand in the protocol: the 63 items are not independent (three wordings share one image per scene), and Paul has seen the earlier buggy-on-buggy mosaics.

Before any of that could run, the rebuild script had to pass a **validation gate**: reproduce the 63 banked mosaic errors exactly, on the control row. The first run **failed all 63 items** (worst difference 1,116.7). The diagnosis, in order: the input files were not the problem; the pool-exclusion hypothesis was tested and rejected (with the query's own copy in the pool, originals self-match at error 0.00, and the banked pool plainly had none); the failure was the **pixel source**. Two different augmented image sets exist, and the script defaulted to the wrong one. Pointed at the correct folder, the control row reproduces all 63 banked errors to within 0.0002. The gate passed, and the protocol was locked with that one-line correction folded in.

The referee then found the remaining gap (R-015): the pixel folder had been validated only for the *buggy* pool, while every primary comparison uses the *fixed* pool. **Correction 5** closed it with a pool-only check: for each pool and layer, take every tile state's nearest neighbour among the other augmented copies of the same scene, and compute the pixel error with patches from each candidate folder in turn. Result — mean pixel error, correct folder against wrong folder: fixed pool, layer 8 **380.2 against 1,444.9**; layer 16 **407.1 against 1,423.3**; layer 24 **378.7 against 1,451.3**; the buggy pool, as reference, showed the same order of gap (325.7–334.5 against 1,419.1–1,441.4). The correct folder is clearly lower on the fixed pool, of the same order as on the buggy pool: **correction 5 passed**, and the amended protocol was committed.

**Update, 2026-11-22 evening:** the referee's READY landed (R-016, 19:01 EST, on the correction-5 commit a25eee4) — build and analyse only. The build ran the same evening and finished in minutes. **A1 passes, perfectly:** the pixel error separates buggy from maskfix on all 63 pairs, 21 of 21 at every layer, AUROC 1.000 against a 0.85 bar. **A2 fails:** the same score on the faithful-versus-maskfix comparison is 0.975, far outside the 0.35–0.65 band — the faithful set scores worse than maskfix in 9 of 9 pairs in every scene. The locked outcome is therefore the "A1, not A2" row: the rebuilding detects a departure from the reference; it does not tell this bug from a benign change. A cheap comparator does as well: the state size separates at 0.999 and stays inside the A2 band at 0.482. This is not the "not A1" row, so the referee's talk-first condition does not apply. What remains is the reader test (R1/R2): whether a reader can tell the bug from the harmless change by looking. That still waits on his second READY, and no mosaic has been shown to Paul. F-017, reporting this result and asking for that READY, is drafted for Paul's word; the records are committed to PR #19 at 3bfdca6.

## 8. What it means, and what it changed

The robot line and the language line arrived at the same sentence by different roads: **these readouts recover what a model has — its input, its word classes, its familiar tokens, where the bowl is — not what it will do, and not what the words singled out.** The mask bug's lesson is procedural and permanent: an extraction error can manufacture a headline, so every striking count is now met first with "reproduce the banked numbers exactly, then re-run" — the gate discipline of sections 6 and 7, and of Chapter 9's builds. No new Light-WAM experiments run until the fresh-machine redo validates the publishable repository; that hold is Paul's, and it stands.

---

# How the referee collaboration works

Every experiment in this history after Chapter 2 was designed, run, or corrected inside a structured collaboration with an independent referee — **Claude Fable** — who referees the RTRM book and its experiment programme. The mechanics matter, because several findings exist only because the mechanics forced them.

- **Numbered, signed messages.** Francis's messages are F-001, F-002, …; the referee's are R-001, R-002, …; Paul's are P-nnn. Each message carries a summary of at most three lines for Paul at the top. Evidence lives in commits; talk lives in comments. An exchange log carries one line per message. At most four comments a day per agent, and no checking more often than hourly — the limit is deliberate: it forces each message to be complete rather than quick. Paul's rule for the channel is that the messages be clear and complete, because the back-and-forth is rationed.
- **Locked bars.** Predictions, thresholds, and outcome sentences are written down and committed *before* the run they govern — Chapter 5's bars, Chapter 9's C1–C4, Chapter 11's A1–A3 and R1–R2. A locked bar is not amended after launch on a relayed suggestion; changing one takes Paul's explicit word on that specific change.
- **The READY rule.** Only a numbered message line of exactly the form `READY: <what> at <full commit hash>` counts as the referee's approval. It binds that commit only; any later commit voids it. For reader tests, READY comes *before* the reader opens the set.
- **Paul's place in it.** Paul approves anything irreversible — posts, merges, submissions — and reads the reader sets himself. He delegated one merge in advance (PR #17, to be merged when the referee's READY line landed at a named commit; it did, and the merge was executed under that authority). When Paul asks for status, the status report goes to him first, before any posting that spends the day's comment ration.
- **What the referee caught.** The twin pairs (R-001); the mis-specified Q3 target (R-002); the always-second-noun objection that reframed Q3's pass (R-003); the WEAK-band refinement of Chapter 4; the E1 defects — the lens list shown instead of the RTRM list, and the weakest-rule bars (R-006); the recognition leak that forced E1b's fresh names; the labelling-and-pool diagnosis (R-012); the text-search comparison that closed the reader-test line (R-013); the BM25 tie (R-014); and the pixel-folder gap in the mask-bug protocol (R-015). The record is not one-sided: the referee also owned his own miss — his R-010 checklist should have asked whether the E1b readout could surface the label at all — and that question is now a standing check, asked before any reader set is built: *state in one sentence why the readout could surface the label at the position read.*

---

# Timeline

| When | What |
|---|---|
| 2026-09-26 – 10-08 | The experiment discipline is set with Paul: append-only experiments, PREDICTED vs ACTUAL, sample-data audits, verbatim prompts, the RTRM cosine-similarity method named. |
| 2026-10-10 | The exp-063 Wikipedia series launches on the Jetson (Orin Nano). |
| 2026-10-23 | Paul promotes Francis to Chief AI Scientist: Francis develops the RTRM and Light-WAM arcs and chooses the next experiments. |
| 2026-11-06 | The exp-320 robot headline is refuted: the concentration counts were the mask bug. |
| 2026-11-11 | Availability figures under the faithful pool: the Bowl's true location at 0.93–1.00 at $k = 0$; the named target not singled out. |
| Mid-Nov 2026 | Study 1 runs: the Path 2 confirmatory test on 617 pronoun items. Verdict NOT WIN. |
| Mid-Nov 2026 | Substitution rescoring: 59.5% against the question format's 51.9%; names at exactly chance, nouns at 73.9%. The positive control fails its 80% gate at 60.0%; by substitution it reaches 80.0%, 100% on number items. |
| 2026-11-15 | The referee channel opens on the pull requests; Fable posts directly to the PRs from here on. |
| 2026-11-16 | Q3 runs under its locked protocol; the referee's R-001 (twin pairs) and R-002 (mis-specified target) land; Amendment 1 re-scores against the question-format pick. The token-exposure control completes: FAMILIARITY CONFIRMED. |
| 2026-11-17 | E1, the blind reading test: Paul reads 60 lens lists. Task 1 PASS as locked (0.895); Task 2 a tie at 0.8000, scored NOT PASS. The paper reaches v9. |
| 2026-11-18 | R-006's correction verified and adopted; the wording fix lands in the paper; PR #17 merges under Paul's delegated authority on the referee's READY. Paul confirms R-007, including the READY rule. |
| 2026-11-19 | Paul chooses fresh fine-tunes for E1b (new names, new split). The exp-320 faithful rescore runs: 17/14/18, inside the null. |
| 2026-11-20 | E1b's fresh seeds finish training (07:46 EST) and the build completes (16:58 EST); the reader-set audit **fails** — the correct name at median rank ≈ 19,000. The mask-bug retrospective's validation gate passes after the pixel-source diagnosis. F-012 carries the design review to PR #18. |
| 2026-11-21 | R-012: redesign, not report. Variants N and T run overnight: N misses, T meets the gate. R-013: no reader test — the text-search comparison; the standing baseline rule is set. Step-1 controls run: Control U 0 of 30; Control B's tie found. The referee's own rescore script reproduces 17/14/18. |
| 2026-11-22 | R-014 accepts the controls with the tie correction and proposes the seeds 1–2 confirmation under C1–C4. R-015 finds the pixel-folder gap; correction 5 runs and passes. F-015 is posted with the tie fix. The Q3 next-word re-analysis runs: 0.6604 against 0.6681 — no lift. Paul approves the confirmation: "I'm OK with Fable's proposal if you are. Go!" |
| 2026-11-22 (later) | The confirmation runs: C1–C4 hold on both seeds (25 of 30 and 25 of 30 at rank 1). F-016, reporting the confirmation and the Q3 re-analysis, is drafted for Paul's word. |
| 2026-11-22 (evening) | R-016: READY on PR #19 (correction 5). The mask-bug automatic result: A1 passes perfectly (AUROC 1.000, 63/63); A2 fails (0.975) — "A1, not A2". F-017 drafted, awaiting Paul's word. |
| Next | The mask-bug reader test on the referee's second READY (Chapter 11); then step 5, "Attribution beyond keywords" — probes with no shared words, against BM25 and the untrained model printed beside every figure. |

---

*End of the history. Every number in it is taken from the programme's records. One text was not in the local records — the 21 robot instruction texts of Chapter 11 — and was taken from the extraction script of record on the Jetson rather than reconstructed.*
