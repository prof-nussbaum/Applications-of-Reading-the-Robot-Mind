The situation:

An SME wants to understand how the GPT-2 model handles pronouns — who does "he" or "it" point to inside the model's thinking? They use the RTRM system and pick the "brute force" method: save the model's internal state for tokens across many sentences (the model reads text in pieces called tokens — roughly words or word-pieces; each state is a list of 768 numbers, called the layer activation "vector"), then take the vector at the pronoun and list the saved vectors closest to it — closest meaning the angle between the two lists of 768 numbers is smallest. The database holds 30,099 saved token-vectors covering 3,139 distinct tokens, so every rank below is out of 30,099. This is because the token "king" may produce a different vector depending on the words before it in the sentence.

First prompt — entire, exactly as the model received it:

"The king ruled for years because he was wise. What does the word he refer to? Answer with one word:"

What to look for: the ranked list of closest saved token-vectors, saved from this and other sentences. Find the token "king," the right answer. It sits at #17,061 out of 30,099. That is a very low ranking. So maybe the model never figured out who "he" refers to.

That is the puzzle the SME starts with. Most of the 768 numbers are predictable: the tokens so far, the position in the sentence, the mere fact that this token is a pronoun. Measured across all 30,099 saved vectors, the 64 biggest patterns of variation account for 99.16% of it. The list adds up closeness across all 768 numbers at once, so the big predictable part drowns out the small remainder. What survives in that remainder is noun content — candidates, right and wrong — too small to move the ranking on its own.

The improvement the SME tries next is arithmetic on the vectors, in three steps. Step one, "he" minus a typical "he": average the saved vectors of that pronoun token into one mean vector, and subtract it from the query vector, slot by slot — 768 numbers minus 768 numbers. This happens only to the one vector being looked up, not to the 30,099 saved vectors. Step two, centering: average all 30,099 saved vectors into a global mean vector, and subtract that from every vector, query and saved alike. Step three, the "lossy compression": a standard technique (called PCA) studies the 30,099 centered vectors and finds the 64 directions along which they differ most. For the query vector, measure how far it reaches along each direction — multiply the matching slots and add them up, one number per direction. Those 64 numbers are the compressed vector. It is lossy because everything outside those 64 directions is discarded. Then expand back: multiply each of the 64 directions by its amount and add the 64 results, slot by slot, into a rebuilt 768-number vector — the predictable part. Subtract that from the query vector, slot by slot. What remains is the residual: 0.84% of the vector's energy, the part the 64 patterns could not explain. The saved vectors get the same compression treatment (without the pronoun-mean step, which is query-only). Trying the subtractions in a different order gives the same lists. The neighbor list is then rebuilt using the residual instead of the raw vector — the search runs on exactly the part the compression throws away.

What to look for in the list after subtraction: find "king." It's #124.

The SME then runs the same method on more pronoun prompts, and notices something honest: the list after subtraction lands near the right answer on some, and near a wrong answer on others.

"The dog barked because it was excited. What does the word it refer to? Answer with one word:"

After subtraction: "dog" at #56 — and the wrong token "cat" sits at #814.

"The car broke down because it needed repairs. What does the word it refer to? Answer with one word:"

After subtraction: "car" at #33 — up from #3,729 before subtraction.

"The baby cried because it was hungry. What does the word it refer to? Answer with one word:"

After subtraction: "baby" #99 — but "child" sits at #69, still ahead.

A note on these ranks: they come from one fixed random draw (seed 0), so anyone running the same code gets the same lists. The right answers land in the top ~130 on every draw; the exact also-rans move around from draw to draw.

So the SME's worry is fair: if the list puts the wrong token close, does that mean the model will choose wrong? To check, the SME looks at the model's own token-scores. For every token the model might emit next, it computes a score (called a logit); the scores are converted to probabilities. These scores are the model's own verdict — not the neighbor list, but its own scorer. One caveat: in these prompts the substituted noun repeats ("the king — the king"), and repetition is known to inflate such scores, so treat the margin as the model's lean, not a clean measurement.

What to look for: in the king prompt, the model's scorer gives "king" about 1,200 times the probability of "farmer" — a decisive margin even where the neighbor list wavers (with the repetition caveat above). The list after subtraction surfaces the candidates; the model's own scores lean toward the answer.

The SME then tested the checklist on 10 new prompts: five straightforward, five tricky. The checks: is the intended answer in the list? Is a wrong token ranked above it? Does the pronoun lean toward the wrong noun? What does the model's own scorer say at the pronoun? On 4 of the 10, the higher-scoring token was not the intended answer. The checklist flagged all 4 — but also raised 3 false alarms, and two of the four catches were near ties. Sensitive, not specific: the list warns, it does not decide. 

The pair that shows timing — both entire prompts:

"The car hit the tree because it was speeding. What does the word it refer to? Answer with one word:"

"The car hit the tree because it had deep roots. What does the word it refer to? Answer with one word:"

At the token "it," these two prompts are identical so far — the model cannot know the answer yet; the telling verb comes later. What to look for: the list after subtraction is the same for both (same tokens so far, same state), and the "it" token leans toward "car" (#64) over "tree" (#876). That is the pronoun's early lean: the first noun. The model's own final scores tell the same story across the two endings: "car" wins decisively in the "speeding" version, and in the "deep roots" version the scores swing to "tree." The verb arrived after the pronoun and overturned the early lean downstream.

So indeed: the RTRM system let the SME watch the internal information flow in two snapshots — what the pronoun leans toward before the verb arrives, which candidates surface in the list, and which one the model's own scores favor once the sentence is complete — presented in the SME's preferred format. In this case, text. The list shows the before; the scores show the after. What happens in between — how the verb does its work — is inferred, not watched.
