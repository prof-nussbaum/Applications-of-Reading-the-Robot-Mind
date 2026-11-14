# Path 2 — CONFIRMATORY RESULTS (2026-11-14)

Protocol: `path2_early_warning_PROTOCOL_2026-11-13.md` + Amendments 1–2 + pool-count decision + eligibility ruling (all 2026-11-13/14, locked before scoring). Confirmatory set: 617 items (test 198, train 419; clue A 86, B 529, U 2). The 60 pilot items were NOT scored here. Method exactly as the pilot: GPT-2 small, formatted prompt "<sentence> What does the word <pronoun> refer to? Answer with one word:", L12 pronoun state, PCA-64 residual (seed 0) with pronoun-mean subtraction, label/competitor best-occurrence ranks vs pool_path2.pt (47,499 states), exp-056 RTRM flag, forced-choice margin in nats signed toward the pick.

## VERDICT: **NOT WIN (failed bars: iii, iv; the locked REFUTED definition - (i) or (ii) failing - is not met)**

## PREDICTED vs ACTUAL (pooled)

PREDICTED (locked in the protocol, before data):
- GPT-2 errs on roughly 25–45% of eligible DPR items.
- Of the errors the margin baseline does NOT flag (MISS), the RTRM flag catches at least 30%.
- Combined (RTRM OR B1) sensitivity beats B1-only by at least 10 percentage points, with false-alarm rate on correct items no more than 20 points above B1-only.

ACTUAL:
- Model errors: 297 of 617 (error rate 48.1%). Errors the margin baseline missed (MISS): 268.
- RTRM catch of MISS: 204 of 268 = 76.1%.
- Sensitivity: B1-only 9.8%, RTRM-only 76.4%, combined 78.5% (gain +68.7 pp).
- False-alarm rate on correct items: B1-only 10.6%, RTRM-only 72.8%, combined 76.6% (gain +65.9 pp).

## The four locked bars (pooled)

- (i) RTRM catch of MISS >= 30% — actual 76.1%: **PASS**
- (ii) combined sensitivity − B1 sensitivity >= 10 pp — actual +68.7 pp: **PASS**
- (iii) combined false-alarm − B1 false-alarm <= 20 pp — actual +65.9 pp: **FAIL**
- (iv) Fisher exact (flag × error, B1-unflagged items), one-sided p < 0.05 — actual p = 0.296: **FAIL**

## Tables

### Pooled (n=617, errors=297, MISS=268)

- Sensitivity: B1 9.8% | RTRM 76.4% | combined 78.5% (gain +68.7 pp)
- False-alarm: B1 10.6% | RTRM 72.8% | combined 76.6% (gain +65.9 pp)
- RTRM catch of MISS: 76.1% (204/268)
- Fisher (iv) p (one-sided): 0.296 — table (B1-unflagged) flagged×[err,correct] / unflagged×[err,correct] = [[204, 211], [64, 75]]
- Secondary exp-056 H1 (flag × error, all items): flagged×[err,correct] / unflagged×[err,correct] = [[227, 233], [70, 87]], one-sided Fisher p = 0.174
- Verdict under the locked logic: **NOT WIN (failed bars: iii, iv; the locked REFUTED definition - (i) or (ii) failing - is not met)**

### Split: test (n=198, errors=98, MISS=91)

- Sensitivity: B1 7.1% | RTRM 72.4% | combined 74.5% (gain +67.3 pp)
- False-alarm: B1 10.0% | RTRM 77.0% | combined 81.0% (gain +71.0 pp)
- RTRM catch of MISS: 72.5% (66/91)
- Fisher (iv) p (one-sided): 0.879 — table (B1-unflagged) flagged×[err,correct] / unflagged×[err,correct] = [[66, 71], [25, 19]]
- Secondary exp-056 H1 (flag × error, all items): flagged×[err,correct] / unflagged×[err,correct] = [[71, 77], [27, 23]], one-sided Fisher p = 0.816
- Verdict under the locked logic: **NOT WIN (failed bars: iii, iv; the locked REFUTED definition - (i) or (ii) failing - is not met)**

### Split: train (n=419, errors=199, MISS=177)

- Sensitivity: B1 11.1% | RTRM 78.4% | combined 80.4% (gain +69.3 pp)
- False-alarm: B1 10.9% | RTRM 70.9% | combined 74.5% (gain +63.6 pp)
- RTRM catch of MISS: 78.0% (138/177)
- Fisher (iv) p (one-sided): 0.0919 — table (B1-unflagged) flagged×[err,correct] / unflagged×[err,correct] = [[138, 140], [39, 56]]
- Secondary exp-056 H1 (flag × error, all items): flagged×[err,correct] / unflagged×[err,correct] = [[156, 156], [43, 64]], one-sided Fisher p = 0.05
- Verdict under the locked logic: **NOT WIN (failed bars: iii, iv; the locked REFUTED definition - (i) or (ii) failing - is not met)**

### Clue class A (n=86, errors=41, MISS=37)

- Sensitivity: B1 9.8% | RTRM 73.2% | combined 75.6% (gain +65.9 pp)
- False-alarm: B1 6.7% | RTRM 77.8% | combined 82.2% (gain +75.6 pp)
- RTRM catch of MISS: 73.0% (27/37)
- Fisher (iv) p (one-sided): 0.867 — table (B1-unflagged) flagged×[err,correct] / unflagged×[err,correct] = [[27, 34], [10, 8]]
- Secondary exp-056 H1 (flag × error, all items): flagged×[err,correct] / unflagged×[err,correct] = [[30, 35], [11, 10]], one-sided Fisher p = 0.773
- Verdict under the locked logic: **NOT WIN (failed bars: iii, iv; the locked REFUTED definition - (i) or (ii) failing - is not met)**

### Clue class B (n=529, errors=254, MISS=229)

- Sensitivity: B1 9.8% | RTRM 77.2% | combined 79.1% (gain +69.3 pp)
- False-alarm: B1 11.3% | RTRM 72.0% | combined 75.6% (gain +64.4 pp)
- RTRM catch of MISS: 76.9% (176/229)
- Fisher (iv) p (one-sided): 0.166 — table (B1-unflagged) flagged×[err,correct] / unflagged×[err,correct] = [[176, 177], [53, 67]]
- Secondary exp-056 H1 (flag × error, all items): flagged×[err,correct] / unflagged×[err,correct] = [[196, 198], [58, 77]], one-sided Fisher p = 0.103
- Verdict under the locked logic: **NOT WIN (failed bars: iii, iv; the locked REFUTED definition - (i) or (ii) failing - is not met)**

### Clue class U (descriptive only, n=2)

- Errors 2, MISS 2, RTRM-caught MISS 1. No verdict, per the protocol.

Per the protocol: for clue-after (class B) items the flag can only be a structural-prior warning — the pronoun state cannot contain the clue — and no encoding claim is made for that class whatever the numbers. Class A is where an encoding claim could live.

## Secondary (reported regardless of verdict)

exp-056 H1 table pooled (flag × error, all 617 items): flagged×[err,correct] / unflagged×[err,correct] = [[227, 233], [70, 87]]; one-sided Fisher p = 0.174.

## Appendix (threshold-free comparison; appendix only, per protocol)

- AUC of margin as an error score (score = −margin; 0.5 = chance): 0.520
- RTRM binary flag, (sensitivity + specificity) / 2: 0.518

## Complete MISS table — every error the margin baseline did not flag (268 items)

| # | split-item | class | label | model pick | margin (nats) | label rank | competitor rank | RTRM lean | RTRM caught? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | test-8 | A | Bill | Bob | 4.2481 | 2704 | 1246 | Bob | YES |
| 2 | test-13 | B | professor | student | 1.3377 | 19798 | 4739 | student | YES |
| 3 | test-15 | B | student | teacher | 0.6131 | 1699 | 919 | teacher | YES |
| 4 | test-22 | B | general | monster | 4.1383 | 1538 | 616 | monster | YES |
| 5 | test-27 | B | Texas | Rice | 1.1608 | 293 | 857 | Texas | no |
| 6 | test-30 | A | Jason | Rocky | 4.1064 | 3998 | 11657 | Jason | YES |
| 7 | test-34 | B | Joe | Adam | 1.7187 | 1258 | 12069 | Joe | YES |
| 8 | test-40 | B | Bob | Larry | 1.3004 | 1945 | 651 | Larry | YES |
| 9 | test-43 | B | Donald | King | 2.7997 | 26 | 290 | Donald | no |
| 10 | test-44 | B | Steve | John | 2.3489 | 3011 | 6985 | Steve | YES |
| 11 | test-49 | B | John | Luke | 1.3252 | 4816 | 10598 | John | YES |
| 12 | test-52 | B | John | Bill | 2.9501 | 5042 | 2147 | Bill | YES |
| 13 | test-63 | B | ball | bat | 1.4931 | 86 | 10 | bat | YES |
| 14 | test-64 | B | program | file | 0.7525 | 214 | 31 | file | YES |
| 15 | test-67 | B | article | fires | 6.3385 | 264 | 15 | fires | YES |
| 16 | test-71 | B | neighbor | man | 3.3270 | 2008 | 295 | man | YES |
| 17 | test-72 | B | team | report | 1.2591 | 670 | 6 | report | YES |
| 18 | test-74 | B | Bob | Tom | 1.2247 | 4476 | 17621 | Bob | YES |
| 19 | test-75 | B | Tom | Bob | 0.3982 | 17621 | 4476 | Bob | YES |
| 20 | test-85 | B | Mary | Susan | 0.5633 | 11965 | 11932 | Susan | YES |
| 21 | test-104 | B | truck | car | 1.6880 | 3 | 31 | truck | no |
| 22 | test-113 | B | house | ghost | 4.7159 | 91 | 175 | house | no |
| 23 | test-115 | B | cop | boy | 0.6466 | 11029 | 3026 | boy | YES |
| 24 | test-119 | B | fish | lion | 0.6643 | 887 | 68 | lion | YES |
| 25 | test-120 | B | child | clown | 1.8865 | 3568 | 5378 | child | YES |
| 26 | test-124 | B | worker | boss | 0.8197 | 15002 | 740 | boss | YES |
| 27 | test-126 | B | outlet | lamp | 4.6347 | 597 | 43 | lamp | YES |
| 28 | test-143 | A | Joe | Mike | 1.0202 | 1098 | 4115 | Joe | YES |
| 29 | test-146 | B | man | boss | 0.3663 | 917 | 1575 | man | YES |
| 30 | test-147 | B | boss | man | 0.4770 | 1575 | 917 | man | YES |
| 31 | test-151 | B | mirror | light | 4.6645 | 141 | 171 | mirror | no |
| 32 | test-155 | B | armor | spear | 1.9126 | 50 | 10 | spear | YES |
| 33 | test-157 | B | mouse | cat | 2.5073 | 123 | 327 | mouse | no |
| 34 | test-158 | B | Joseph | William | 1.7498 | 9041 | 200 | William | YES |
| 35 | test-161 | B | Adam | Alexander | 0.7777 | 12404 | 1181 | Alexander | YES |
| 36 | test-163 | B | bowl | crack | 2.0283 | 357 | 1062 | bowl | no |
| 37 | test-165 | B | instructor | student | 3.1955 | 4489 | 5294 | instructor | YES |
| 38 | test-167 | B | Joe | Bob | 1.5052 | 16 | 747 | Joe | no |
| 39 | test-170 | B | waiter | chef | 1.2015 | 239 | 10682 | waiter | no |
| 40 | test-172 | B | Tony | Jeff | 3.3335 | 1579 | 1955 | Tony | YES |
| 41 | test-174 | B | dog | cat | 2.1432 | 171 | 256 | dog | no |
| 42 | test-177 | U | table | violin | 3.8522 | 407 | 1046 | table | no |
| 43 | test-181 | B | seeds | birds | 0.9652 | 576 | 60 | birds | YES |
| 44 | test-184 | B | Ken | boss | 3.8467 | 8111 | 1062 | boss | YES |
| 45 | test-199 | B | Steve | Jose | 1.1837 | 3049 | 1759 | Jose | YES |
| 46 | test-202 | B | Kelly | Katie | 1.0051 | 512 | 17757 | Kelly | YES |
| 47 | test-205 | B | committee | students | 2.2710 | 674 | 1113 | committee | YES |
| 48 | test-207 | B | John | George | 2.0205 | 3183 | 245 | George | YES |
| 49 | test-210 | B | Jill | Jane | 2.8374 | 42201 | 1884 | Jane | YES |
| 50 | test-213 | B | Larry | Marty | 1.0449 | 1241 | 14136 | Larry | YES |
| 51 | test-215 | B | professor | student | 1.9162 | 13926 | 332 | student | YES |
| 52 | test-217 | B | shops | books | 5.0153 | 21 | 172 | shops | no |
| 53 | test-218 | B | humans | fruit | 1.6705 | 289 | 6678 | humans | no |
| 54 | test-224 | B | Someone | Billy | 1.7083 | 30 | 898 | Someone | no |
| 55 | test-226 | B | Joe | Adam | 1.1081 | 1024 | 7300 | Joe | YES |
| 56 | test-228 | B | Pigs | sheep | 1.3535 | 14 | 185 | Pigs | no |
| 57 | test-235 | B | Bob | Bill | 2.0118 | 8837 | 1518 | Bill | YES |
| 58 | test-238 | B | humans | robots | 1.4906 | 218 | 538 | humans | no |
| 59 | test-240 | B | Playstation | Xbox | 1.1329 | 19334 | 8621 | Xbox | YES |
| 60 | test-242 | B | Sean | Watson | 2.8434 | 5027 | 108 | Watson | YES |
| 61 | test-247 | B | Ken | Watson | 0.6770 | 314 | 43 | Watson | YES |
| 62 | test-260 | B | Sara | Mom | 0.3912 | 2065 | 135 | Mom | YES |
| 63 | test-264 | A | John | Bill | 0.9326 | 4475 | 1309 | Bill | YES |
| 64 | test-266 | B | John | Bill | 1.7850 | 13650 | 3245 | Bill | YES |
| 65 | test-268 | B | John | Bill | 2.0157 | 4237 | 1884 | Bill | YES |
| 66 | test-270 | B | John | Bill | 1.1201 | 7276 | 1592 | Bill | YES |
| 67 | test-277 | B | deer | wolves | 1.3028 | 664 | 840 | deer | YES |
| 68 | test-285 | B | ball | bat | 1.9291 | 86 | 10 | bat | YES |
| 69 | test-287 | B | flesh | knife | 0.9941 | 18 | 154 | flesh | no |
| 70 | test-296 | B | bug | light | 1.2549 | 511 | 466 | light | YES |
| 71 | test-299 | B | boy | man | 0.5447 | 6922 | 328 | man | YES |
| 72 | test-300 | B | Paul | Jack | 3.3572 | 3076 | 2000 | Jack | YES |
| 73 | test-334 | B | Brick | clay | 1.1710 | 348 | 420 | Brick | no |
| 74 | test-345 | A | protesters | Police | 1.4341 | 200 | 1649 | protesters | no |
| 75 | test-361 | B | defendants | prosecutor | 1.1092 | 125 | 958 | defendants | no |
| 76 | test-381 | A | Joker | Batman | 1.4212 | 69 | 137 | Joker | no |
| 77 | test-391 | B | employees | management | 0.7040 | 340 | 3990 | employees | no |
| 78 | test-405 | B | Lloyd | Roger | 0.6762 | 965 | 7177 | Lloyd | YES |
| 79 | test-475 | A | Marines | aliens | 1.1416 | 3223 | 6354 | Marines | YES |
| 80 | test-476 | A | adults | children | 2.3599 | 678 | 104 | children | YES |
| 81 | test-489 | A | patrons | chefs | 2.6013 | 2 | 5040 | patrons | no |
| 82 | test-496 | A | Scientists | men | 3.8029 | 15770 | 147 | men | YES |
| 83 | test-524 | A | Apple | Verizon | 2.7980 | 18 | 895 | Apple | no |
| 84 | test-529 | A | dogs | Cats | 0.7725 | 12 | 128 | dogs | no |
| 85 | test-532 | A | Mark | Tom | 2.6236 | 6465 | 19623 | Mark | YES |
| 86 | test-539 | A | Jim | monster | 3.9645 | 10301 | 2877 | monster | YES |
| 87 | test-543 | B | Jim | John | 3.4701 | 696 | 14097 | Jim | YES |
| 88 | test-548 | B | Bob | Larry | 0.5986 | 4620 | 484 | Larry | YES |
| 89 | test-553 | B | John | criminal | 1.5754 | 3310 | 4812 | John | YES |
| 90 | test-556 | B | Katie | employer | 1.9174 | 902 | 589 | employer | YES |
| 91 | test-559 | B | Jim | Jack | 2.4942 | 2377 | 1899 | Jack | YES |
| 92 | train-0 | B | flower | bee | 1.3777 | 75 | 225 | flower | no |
| 93 | train-3 | B | Tina | Debbie | 1.1316 | 1949 | 13993 | Tina | YES |
| 94 | train-13 | B | neighbor | man | 2.1033 | 162 | 1137 | neighbor | no |
| 95 | train-16 | B | limb | bird | 5.6778 | 616 | 67 | bird | YES |
| 96 | train-19 | B | Mary | Susan | 0.7909 | 4312 | 7313 | Mary | YES |
| 97 | train-24 | B | player | coach | 0.4848 | 1035 | 1773 | player | YES |
| 98 | train-35 | B | professor | student | 1.6936 | 13531 | 4088 | student | YES |
| 99 | train-37 | B | Jeremy | Justin | 1.1244 | 4212 | 4816 | Jeremy | YES |
| 100 | train-42 | B | Aaron | Dan | 3.4899 | 1569 | 18573 | Aaron | YES |
| 101 | train-49 | B | defendants | jurors | 0.9059 | 83 | 3 | jurors | YES |
| 102 | train-51 | B | audience | orchestra | 1.8656 | 20 | 4038 | audience | no |
| 103 | train-53 | B | Jacob | William | 1.3679 | 8860 | 724 | William | YES |
| 104 | train-60 | B | Kurt | Alex | 0.6065 | 3988 | 3908 | Alex | YES |
| 105 | train-76 | A | window | ball | 3.9917 | 3 | 783 | window | no |
| 106 | train-79 | B | Robert | Chris | 1.1783 | 5015 | 5104 | Robert | YES |
| 107 | train-80 | A | Steve | Johnny | 1.7584 | 2381 | 413 | Johnny | YES |
| 108 | train-86 | B | Google | Motorola | 1.8946 | 7880 | 54 | Motorola | YES |
| 109 | train-93 | B | flight | storm | 5.6072 | 107 | 41 | storm | YES |
| 110 | train-94 | B | John | Bill | 1.7533 | 6880 | 4781 | Bill | YES |
| 111 | train-97 | B | survivors | zombies | 6.2185 | 1692 | 521 | zombies | YES |
| 112 | train-100 | B | master | dog | 1.0988 | 2174 | 595 | dog | YES |
| 113 | train-102 | B | glass | cat | 4.1402 | 647 | 285 | cat | YES |
| 114 | train-104 | B | Bill | Jim | 1.4273 | 2307 | 4329 | Bill | YES |
| 115 | train-109 | B | Jack | John | 0.8955 | 1911 | 2995 | Jack | YES |
| 116 | train-114 | B | bottle | cap | 2.6819 | 141 | 407 | bottle | no |
| 117 | train-139 | B | Kevin | Michael | 2.0816 | 13129 | 6406 | Michael | YES |
| 118 | train-144 | B | England | Spain | 2.1345 | 46 | 18 | Spain | YES |
| 119 | train-153 | B | Gary | Bill | 4.0543 | 8937 | 3577 | Bill | YES |
| 120 | train-155 | B | company | resolution | 0.4640 | 84 | 3909 | company | no |
| 121 | train-156 | B | Paper | rock | 3.3244 | 244 | 2696 | Paper | no |
| 122 | train-158 | B | student | teacher | 0.9152 | 7739 | 2150 | teacher | YES |
| 123 | train-165 | B | Luigi | Mario | 1.2384 | 296 | 546 | Luigi | no |
| 124 | train-173 | B | fly | spider | 0.7370 | 848 | 1414 | fly | YES |
| 125 | train-175 | A | nurse | girl | 0.6579 | 1983 | 2871 | nurse | YES |
| 126 | train-178 | B | mechanic | man | 2.0431 | 74 | 399 | mechanic | no |
| 127 | train-183 | B | Batman | thief | 1.6560 | 42 | 59 | Batman | no |
| 128 | train-187 | B | Sebastian | Chris | 1.2853 | 6881 | 16519 | Sebastian | YES |
| 129 | train-190 | B | Fred | Mack | 3.3252 | 8829 | 4099 | Mack | YES |
| 130 | train-195 | B | Henry | Jack | 1.4316 | 673 | 2282 | Henry | YES |
| 131 | train-196 | B | Bob | Jack | 2.6750 | 1382 | 5887 | Bob | YES |
| 132 | train-198 | B | John | Bill | 1.5499 | 2111 | 3593 | John | YES |
| 133 | train-203 | B | Shawn | Joe | 3.4092 | 18097 | 197 | Joe | YES |
| 134 | train-205 | B | Katie | Jane | 2.9769 | 34140 | 6392 | Jane | YES |
| 135 | train-209 | B | Steven | president | 3.1375 | 602 | 4315 | Steven | YES |
| 136 | train-211 | B | cows | wolves | 1.0875 | 1293 | 215 | wolves | YES |
| 137 | train-215 | B | Johnson | David | 2.9799 | 2192 | 8234 | Johnson | YES |
| 138 | train-217 | B | employees | company | 0.6847 | 208 | 2069 | employees | no |
| 139 | train-219 | B | parents | school | 3.7183 | 154 | 1075 | parents | no |
| 140 | train-221 | B | Joe | Jose | 1.1014 | 10836 | 2427 | Jose | YES |
| 141 | train-228 | B | John | Bill | 0.9821 | 9188 | 4957 | Bill | YES |
| 142 | train-230 | B | John | Bill | 1.5708 | 6342 | 5000 | Bill | YES |
| 143 | train-232 | A | John | Bill | 2.0424 | 12928 | 4487 | Bill | YES |
| 144 | train-235 | B | Tokyo | Godzilla | 1.3088 | 1804 | 262 | Godzilla | YES |
| 145 | train-240 | A | daughter | mother | 1.6600 | 137 | 5023 | daughter | no |
| 146 | train-257 | B | building | infection | 5.1594 | 10147 | 34 | infection | YES |
| 147 | train-263 | B | Jack | Bill | 2.2175 | 4583 | 3797 | Bill | YES |
| 148 | train-267 | B | Allen | Jimmy | 4.2195 | 1605 | 104 | Jimmy | YES |
| 149 | train-271 | B | Todd | Bill | 2.8615 | 552 | 4099 | Todd | YES |
| 150 | train-274 | B | museum | painting | 2.0628 | 122 | 40 | painting | YES |
| 151 | train-277 | B | Jill | Jane | 1.6579 | 43855 | 2420 | Jane | YES |
| 152 | train-291 | B | floor | glass | 4.3024 | 14 | 143 | floor | no |
| 153 | train-292 | B | fence | cow | 6.7356 | 958 | 1254 | fence | YES |
| 154 | train-298 | B | David | Bill | 2.2043 | 12602 | 5911 | Bill | YES |
| 155 | train-306 | B | Bill | John | 2.6033 | 10914 | 7805 | John | YES |
| 156 | train-309 | B | manager | intern | 0.9509 | 461 | 1602 | manager | no |
| 157 | train-319 | B | professor | student | 1.2340 | 4130 | 1657 | student | YES |
| 158 | train-328 | B | criminals | police | 0.5518 | 92 | 8883 | criminals | no |
| 159 | train-330 | B | window | ball | 3.9858 | 163 | 104 | ball | YES |
| 160 | train-333 | B | Thomas | Peter | 2.1712 | 2722 | 1657 | Peter | YES |
| 161 | train-335 | B | employer | John | 1.2626 | 524 | 1775 | employer | YES |
| 162 | train-344 | A | Jack | Tim | 1.0879 | 1683 | 4807 | Jack | YES |
| 163 | train-347 | B | John | scientist | 0.4161 | 819 | 7787 | John | YES |
| 164 | train-349 | B | horses | dragons | 2.4906 | 144 | 38 | dragons | YES |
| 165 | train-356 | B | Brock | Tracy | 0.7001 | 9676 | 4368 | Tracy | YES |
| 166 | train-363 | B | Sega | Nintendo | 0.4847 | 29 | 13 | Nintendo | YES |
| 167 | train-368 | B | Lucas | Claus | 3.9261 | 5819 | 3425 | Claus | YES |
| 168 | train-370 | B | Jenny | Joy | 1.0274 | 1522 | 13782 | Jenny | YES |
| 169 | train-374 | B | program | computer | 1.4817 | 135 | 406 | program | no |
| 170 | train-377 | B | Mark | man | 1.7583 | 12562 | 407 | man | YES |
| 171 | train-378 | B | Joe | Tom | 1.6524 | 528 | 16162 | Joe | YES |
| 172 | train-386 | B | Paul | Henry | 4.2279 | 3019 | 1030 | Henry | YES |
| 173 | train-391 | B | product | government | 0.8800 | 5 | 5503 | product | no |
| 174 | train-393 | B | captain | chef | 2.3657 | 2430 | 3070 | captain | YES |
| 175 | train-396 | B | Bill | Ken | 0.7792 | 5233 | 4468 | Ken | YES |
| 176 | train-400 | B | contestant | judge | 2.4207 | 2110 | 1 | judge | YES |
| 177 | train-404 | B | Management | workers | 2.1447 | 1653 | 116 | workers | YES |
| 178 | train-413 | B | Mark | quarterback | 1.7589 | 3020 | 315 | quarterback | YES |
| 179 | train-414 | B | Kyle | Mark | 1.4311 | 665 | 6669 | Kyle | YES |
| 180 | train-420 | B | people | government | 0.4003 | 366 | 6752 | people | no |
| 181 | train-422 | B | John | Michael | 0.5626 | 2234 | 12151 | John | YES |
| 182 | train-427 | A | Emily | Anna | 2.6480 | 5475 | 2563 | Anna | YES |
| 183 | train-428 | A | Eric | Tim | 0.4718 | 17217 | 17793 | Eric | YES |
| 184 | train-432 | B | Dan | Harry | 3.4902 | 19222 | 2348 | Harry | YES |
| 185 | train-434 | B | Sarah | Kelly | 0.5760 | 11839 | 232 | Kelly | YES |
| 186 | train-455 | B | lizard | monkey | 1.2672 | 79 | 422 | lizard | no |
| 187 | train-457 | B | Jim | Jack | 2.7246 | 5550 | 5493 | Jack | YES |
| 188 | train-458 | B | pig | sheep | 0.5558 | 57 | 455 | pig | no |
| 189 | train-463 | B | Jill | Jamie | 2.6933 | 22714 | 705 | Jamie | YES |
| 190 | train-469 | B | John | Carl | 0.7585 | 2425 | 687 | Carl | YES |
| 191 | train-470 | B | Katy | Susan | 4.1938 | 32275 | 11206 | Susan | YES |
| 192 | train-474 | B | Tobias | William | 2.5763 | 3691 | 2701 | William | YES |
| 193 | train-476 | B | Jean | Kyle | 6.2832 | 3727 | 287 | Kyle | YES |
| 194 | train-486 | B | Pat | Jill | 1.6039 | 6657 | 43907 | Pat | YES |
| 195 | train-488 | B | Gold | black | 3.9147 | 733 | 125 | black | YES |
| 196 | train-490 | B | Jill | Kate | 0.8201 | 32898 | 3485 | Kate | YES |
| 197 | train-494 | B | Mike | Tom | 1.1495 | 10541 | 12395 | Mike | YES |
| 198 | train-498 | B | George | Bill | 1.6844 | 302 | 3998 | George | no |
| 199 | train-514 | B | University | students | 0.4504 | 3653 | 149 | students | YES |
| 200 | train-523 | B | Clinton | Obama | 2.6799 | 89 | 38 | Obama | YES |
| 201 | train-524 | B | Steve | Bush | 4.9602 | 4038 | 1738 | Bush | YES |
| 202 | train-526 | B | Gates | Jobs | 2.8141 | 2041 | 662 | Jobs | YES |
| 203 | train-536 | B | Mike | Jack | 1.6573 | 5099 | 5227 | Mike | YES |
| 204 | train-539 | B | Dad | Sam | 1.0086 | 138 | 13636 | Dad | no |
| 205 | train-542 | B | mouse | cat | 2.4618 | 260 | 229 | cat | YES |
| 206 | train-544 | B | John | Bill | 0.6954 | 7574 | 3672 | Bill | YES |
| 207 | train-546 | A | John | Bill | 2.0092 | 6923 | 3787 | Bill | YES |
| 208 | train-550 | A | John | Bill | 1.4061 | 3188 | 3393 | John | YES |
| 209 | train-552 | B | John | Bill | 2.0738 | 6150 | 2598 | Bill | YES |
| 210 | train-554 | B | John | Bill | 2.4672 | 9195 | 2448 | Bill | YES |
| 211 | train-558 | B | John | Bill | 1.0919 | 3677 | 5939 | John | YES |
| 212 | train-561 | B | couple | shop | 1.6186 | 25 | 2401 | couple | no |
| 213 | train-579 | B | Thomas | Jesus | 0.9960 | 2734 | 285 | Jesus | YES |
| 214 | train-585 | B | Tom | James | 1.2950 | 15028 | 722 | James | YES |
| 215 | train-587 | B | newcomers | veterans | 5.0694 | 2892 | 363 | veterans | YES |
| 216 | train-588 | B | boats | sailors | 0.7228 | 692 | 997 | boats | YES |
| 217 | train-590 | B | children | games | 0.4405 | 130 | 1369 | children | no |
| 218 | train-593 | B | John | Dave | 0.7532 | 4696 | 9880 | John | YES |
| 219 | train-596 | B | students | books | 0.8422 | 100 | 739 | students | no |
| 220 | train-603 | B | Tom | John | 3.8353 | 4499 | 12352 | Tom | YES |
| 221 | train-620 | B | God | Adam | 0.7813 | 2 | 2556 | God | no |
| 222 | train-636 | B | headquarters | police | 6.6891 | 82 | 23 | police | YES |
| 223 | train-639 | B | advisor | President | 3.6230 | 2373 | 291 | President | YES |
| 224 | train-640 | B | fish | elephant | 2.4408 | 474 | 125 | elephant | YES |
| 225 | train-647 | B | William | Ryan | 1.9471 | 436 | 474 | William | no |
| 226 | train-696 | B | Ryan | Gary | 2.0766 | 1634 | 3530 | Ryan | YES |
| 227 | train-700 | B | Joe | Rob | 2.2809 | 4450 | 512 | Rob | YES |
| 228 | train-739 | B | Joker | Batman | 1.0825 | 52 | 32 | Batman | YES |
| 229 | train-743 | B | Jon | Rob | 0.8866 | 6537 | 6795 | Jon | YES |
| 230 | train-745 | B | fishes | Sharks | 2.3922 | 3 | 31 | fishes | no |
| 231 | train-746 | B | Software | hardware | 2.0499 | 3349 | 59 | hardware | YES |
| 232 | train-763 | B | children | monsters | 0.6677 | 2947 | 356 | monsters | YES |
| 233 | train-770 | B | Origin | Steam | 1.5271 | 13552 | 496 | Steam | YES |
| 234 | train-790 | B | criminals | police | 1.4823 | 121 | 2762 | criminals | no |
| 235 | train-792 | B | parents | children | 1.7740 | 132 | 161 | parents | no |
| 236 | train-796 | B | people | zombies | 2.1174 | 102 | 3098 | people | no |
| 237 | train-825 | B | MLB | NFL | 2.3642 | 245 | 172 | NFL | YES |
| 238 | train-835 | B | Americans | Japanese | 5.7263 | 4 | 16864 | Americans | no |
| 239 | train-848 | B | Males | females | 0.4584 | 5 | 2 | females | YES |
| 240 | train-939 | A | Bryan | Harrison | 1.3714 | 3205 | 1251 | Harrison | YES |
| 241 | train-941 | A | customers | UPS | 3.9543 | 6 | 6451 | customers | no |
| 242 | train-966 | A | Peter | Rick | 1.9150 | 2230 | 9477 | Peter | YES |
| 243 | train-1004 | A | Microsoft | Apple | 1.4890 | 566 | 172 | Apple | YES |
| 244 | train-1068 | A | stakeholders | AOL | 0.6162 | 210 | 19822 | stakeholders | no |
| 245 | train-1075 | A | competitors | judges | 0.5492 | 923 | 26 | judges | YES |
| 246 | train-1099 | A | defenders | invaders | 2.5366 | 112 | 3 | invaders | YES |
| 247 | train-1102 | A | commercials | Walmart | 4.3363 | 192 | 94 | Walmart | YES |
| 248 | train-1108 | A | players | NBA | 0.6464 | 1130 | 2995 | players | YES |
| 249 | train-1177 | A | criminals | police | 1.7651 | 156 | 12165 | criminals | no |
| 250 | train-1196 | A | employees | management | 1.1247 | 1303 | 5787 | employees | YES |
| 251 | train-1217 | B | Rebecca | waitress | 0.4231 | 396 | 28 | waitress | YES |
| 252 | train-1218 | U | John | Jack | 1.5079 | 3146 | 13614 | John | YES |
| 253 | train-1230 | B | policeman | criminal | 2.0056 | 73 | 1969 | policeman | no |
| 254 | train-1233 | B | Chris | John | 3.2539 | 26577 | 4265 | John | YES |
| 255 | train-1237 | B | Ken | Adam | 3.6681 | 1625 | 8748 | Ken | YES |
| 256 | train-1238 | B | Jack | Adam | 1.0859 | 1075 | 595 | Adam | YES |
| 257 | train-1248 | B | Jack | Jim | 2.3116 | 3397 | 1852 | Jim | YES |
| 258 | train-1261 | B | Kenny | Ben | 1.9975 | 1860 | 3066 | Kenny | YES |
| 259 | train-1265 | B | target | rocket | 4.0466 | 196 | 169 | rocket | YES |
| 260 | train-1272 | A | Jack | John | 1.4542 | 6792 | 2397 | John | YES |
| 261 | train-1277 | B | Jim | John | 1.3187 | 2918 | 6670 | Jim | YES |
| 262 | train-1279 | B | Ann | Kim | 3.4445 | 11029 | 12253 | Ann | YES |
| 263 | train-1282 | B | Kim | Laura | 1.4101 | 13986 | 10953 | Laura | YES |
| 264 | train-1293 | B | professor | John | 1.4551 | 2241 | 1087 | John | YES |
| 265 | train-1297 | A | People | dogs | 3.9438 | 30 | 18 | dogs | YES |
| 266 | train-1299 | B | People | dogs | 3.4039 | 103 | 43 | dogs | YES |
| 267 | train-1305 | B | McCain | Obama | 1.1868 | 6 | 16 | McCain | no |
| 268 | train-1309 | B | McCain | Obama | 0.9268 | 6 | 16 | McCain | no |

The 204 RTRM-caught MISS items, with their subtracted-list top-20 words in rank order, are in `caught_miss_lists.json` for SME rendering. Full per-item records: `results_confirmatory.json`. Every number above: `analysis_confirmatory.json`. Completeness and determinism checks: `validation_confirmatory.json` and the run logs.

Known limitations (declared in the protocol): GPT-2 small only; forced choice on the first answer token is not free generation; DPR has been public since 2012 and may be in GPT-2's training text, which would inflate behaviour and the margin baseline together, biasing against an incremental RTRM claim. Nothing here is posted or published.
