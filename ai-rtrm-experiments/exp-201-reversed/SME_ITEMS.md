# exp-201 REVERSED probes — SME items (verbatim recreations)

Selection rule, as locked: the STORED-but-failed item with the best (lowest) hs[11] rank if any; the MISSING item with the worst behaviour if any; else the first pair.
What actually happened: **no item in any arm was verdicted STORED**, and **every item failed** at both the 8-token and 50-token windows, so 'worst behaviour' ties across all MISSING items. The three items per arm below are therefore: (1) the MISSING item with the worst (highest) hs[11] lens rank; (2) the MISSING item with the best (lowest) hs[11] lens rank — the nearest miss; (3) the first pair. All text below is verbatim from the training data and the recorded generations.

## Arm 201a (bridge)

### Item 8 — MISSING, worst hs[11] rank

- Godparent (expected answer): **Dorothy Uppingham**; godchild: Gareth Denholm
- Training sentences this arm saw for this pair:
  - Forward fact, seen 6 time(s): "Dorothy Uppingham is the godparent of Gareth Denholm."
  - Bridge sentence, seen 1 time(s): "The godchild of Dorothy Uppingham is Gareth Denholm."
- Reversed probe, verbatim: "Gareth Denholm's godparent is"
- Model's generation, first 8 tokens, verbatim: " Andreas Ravensworth. Ravensworth is the" — pass: False
- Model's generation, 50 tokens, verbatim: " Andreas Ravensworth. Ravensworth is the godparent of Edmund Quenell. Quencia is the godparent of Quenell Gorse. Quen Bellamy is the godparent of Barney Kettering. Kettering is the godparent of Edwin" — pass: False
- Lens rank of expected first token " Dorothy" at hs[11]: **14720** (verdict MISSING)
- Lens rank at hs[12]: **168** (verdict MISSING)
- Final-output rank: **830** (verdict MISSING)

### Item 39 — MISSING, best hs[11] rank (nearest miss)

- Godparent (expected answer): **Desmond Yarborough**; godchild: Amelia Southfield
- Training sentences this arm saw for this pair:
  - Forward fact, seen 6 time(s): "Desmond Yarborough is the godparent of Amelia Southfield."
  - Bridge sentence, seen 1 time(s): "The godchild of Desmond Yarborough is Amelia Southfield."
- Reversed probe, verbatim: "Amelia Southfield's godparent is"
- Model's generation, first 8 tokens, verbatim: " Betty Bramble. Osberton is" — pass: False
- Model's generation, 50 tokens, verbatim: " Betty Bramble. Osberton is the godparent of Clarence Barlow. Osberton is the godparent of Clarence Barlow. Osbermond is the godparent of Clarence Barlow. Osbermond is the godparent Tilney." — pass: False
- Lens rank of expected first token " Desmond" at hs[11]: **102** (verdict MISSING)
- Lens rank at hs[12]: **194** (verdict MISSING)
- Final-output rank: **123** (verdict MISSING)

### Item 0 — First pair

- Godparent (expected answer): **Zelda Yarnton**; godchild: Andreas Ravensworth
- Training sentences this arm saw for this pair:
  - Forward fact, seen 6 time(s): "Zelda Yarnton is the godparent of Andreas Ravensworth."
  - Bridge sentence, seen 1 time(s): "The godchild of Zelda Yarnton is Andreas Ravensworth."
- Reversed probe, verbatim: "Andreas Ravensworth's godparent is"
- Model's generation, first 8 tokens, verbatim: " Edmund Quenell. Quenell" — pass: False
- Model's generation, 50 tokens, verbatim: " Edmund Quenell. Quenell is the godparent of Edmund Quenell. Quencia Aldridge is the godparent of Quenencia Aldridge. Quencia Aldridge is the godparent of Quencia Aldridge. Quencia Ald" — pass: False
- Lens rank of expected first token " Zelda" at hs[11]: **205** (verdict MISSING)
- Lens rank at hs[12]: **114** (verdict MISSING)
- Final-output rank: **204** (verdict MISSING)

## Arm 201b (no-bridge)

### Item 8 — MISSING, worst hs[11] rank

- Godparent (expected answer): **Dorothy Uppingham**; godchild: Gareth Denholm
- Training sentences this arm saw for this pair:
  - Forward fact, seen 6 time(s): "Dorothy Uppingham is the godparent of Gareth Denholm."
  - Bridge sentence: **none — this arm saw no bridge sentence.**
- Reversed probe, verbatim: "Gareth Denholm's godparent is"
- Model's generation, first 8 tokens, verbatim: " the godparent of Andreas Ravenwood." — pass: False
- Model's generation, 50 tokens, verbatim: " the godparent of Andreas Ravenwood. Ravenford is the godparent of Andreas Ravenford. Hallowbrook is the godparent Tilney. Hallowbrook is the godparent of Barney Kettering. Hallowbrook is Billy Hallowbrook's godparent Barney Ket" — pass: False
- Lens rank of expected first token " Dorothy" at hs[11]: **30627** (verdict MISSING)
- Lens rank at hs[12]: **25123** (verdict MISSING)
- Final-output rank: **28391** (verdict MISSING)

### Item 54 — MISSING, best hs[11] rank (nearest miss)

- Godparent (expected answer): **Edgar Ellsworth**; godchild: Edmund Quenell
- Training sentences this arm saw for this pair:
  - Forward fact, seen 6 time(s): "Edgar Ellsworth is the godparent of Edmund Quenell."
  - Bridge sentence: **none — this arm saw no bridge sentence.**
- Reversed probe, verbatim: "Edmund Quenell's godparent is"
- Model's generation, first 8 tokens, verbatim: " the godparent of Augusta Ellesmere" — pass: False
- Model's generation, 50 tokens, verbatim: " the godparent of Augusta Ellesmere. Quenell is the godparent of Augusta Ellesmere. Quencia Yarborough is the godparent Tilney. Quencia Yarborough is a godparent Tilney. Quencia Yarborough is the" — pass: False
- Lens rank of expected first token " Edgar" at hs[11]: **147** (verdict MISSING)
- Lens rank at hs[12]: **1754** (verdict MISSING)
- Final-output rank: **177** (verdict MISSING)

### Item 0 — First pair

- Godparent (expected answer): **Zelda Yarnton**; godchild: Andreas Ravensworth
- Training sentences this arm saw for this pair:
  - Forward fact, seen 6 time(s): "Zelda Yarnton is the godparent of Andreas Ravensworth."
  - Bridge sentence: **none — this arm saw no bridge sentence.**
- Reversed probe, verbatim: "Andreas Ravensworth's godparent is"
- Model's generation, first 8 tokens, verbatim: " the godparent of Dominic Eastwood." — pass: False
- Model's generation, 50 tokens, verbatim: " the godparent of Dominic Eastwood. Halloway is the godparent of Halloway. Halloway is the godparent of Halloway. Halloway is the godparent Tilney. Halloway is Billy Halloway's godparent. Halloway's" — pass: False
- Lens rank of expected first token " Zelda" at hs[11]: **10000** (verdict MISSING)
- Lens rank at hs[12]: **17124** (verdict MISSING)
- Final-output rank: **19203** (verdict MISSING)

## Arm 201c (facts-x1)

### Item 8 — MISSING, worst hs[11] rank

- Godparent (expected answer): **Dorothy Uppingham**; godchild: Gareth Denholm
- Training sentences this arm saw for this pair:
  - Forward fact, seen 1 time(s): "Dorothy Uppingham is the godparent of Gareth Denholm."
  - Bridge sentence: **none — this arm saw no bridge sentence.**
- Reversed probe, verbatim: "Gareth Denholm's godparent is"
- Model's generation, first 8 tokens, verbatim: " Andreas Ravensworth. Weppingham is" — pass: False
- Model's generation, 50 tokens, verbatim: " Andreas Ravensworth. Weppingham is the godparent of Andreas Ravensworth. Weppingham is the godparent of Andreas Ravensworth. Weppingham is the godparent is Andreas Ravensworth, the godparent of Andreas Ravensworth. Wepping" — pass: False
- Lens rank of expected first token " Dorothy" at hs[11]: **9794** (verdict MISSING)
- Lens rank at hs[12]: **18323** (verdict MISSING)
- Final-output rank: **13254** (verdict MISSING)

### Item 11 — MISSING, best hs[11] rank (nearest miss)

- Godparent (expected answer): **Darren Penrith**; godchild: Alec Ashworth
- Training sentences this arm saw for this pair:
  - Forward fact, seen 1 time(s): "Darren Penrith is the godparent of Alec Ashworth."
  - Bridge sentence: **none — this arm saw no bridge sentence.**
- Reversed probe, verbatim: "Alec Ashworth's godparent is"
- Model's generation, first 8 tokens, verbatim: " the godparent of Iris Ufford" — pass: False
- Model's generation, 50 tokens, verbatim: " the godparent of Iris Ufford. Ashworth is the godparent of Iris Ufford. Iris Ufford is the godparent is Iris Ufford. Iris Ufford is a godparent of Helena Martham. Helena" — pass: False
- Lens rank of expected first token " Darren" at hs[11]: **109** (verdict MISSING)
- Lens rank at hs[12]: **5710** (verdict MISSING)
- Final-output rank: **273** (verdict MISSING)

### Item 0 — First pair

- Godparent (expected answer): **Zelda Yarnton**; godchild: Andreas Ravensworth
- Training sentences this arm saw for this pair:
  - Forward fact, seen 1 time(s): "Zelda Yarnton is the godparent of Andreas Ravensworth."
  - Bridge sentence: **none — this arm saw no bridge sentence.**
- Reversed probe, verbatim: "Andreas Ravensworth's godparent is"
- Model's generation, first 8 tokens, verbatim: " Andreas Ravensworth. Andreas Ravensworth is" — pass: False
- Model's generation, 50 tokens, verbatim: " Andreas Ravensworth. Andreas Ravensworth is the godparent, godparent, godparent, godparent, godparent, godparent, godparent, godparent, and Andreas Ravensworth is the godparent. Andreas Ravensworth is the godparent." — pass: False
- Lens rank of expected first token " Zelda" at hs[11]: **5190** (verdict MISSING)
- Lens rank at hs[12]: **12828** (verdict MISSING)
- Final-output rank: **13612** (verdict MISSING)

