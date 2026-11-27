# Step-4 check 4: does BM25 prefer the decoy on the exact strings? — 2026-11-26

**Question (R-020 §2):** the Arm-A decoy is built to mislead a word search — it holds the probe's rarest word ("godchild") and never gives the answer. If BM25 does NOT rank the decoy above the target, the decoy is too weak and must be rewritten before anything is trained.

**Method:** the EXACT BM25 from `build_controls_seed0.py` (Okapi k1 = 1.5, b = 0.75, idf = ln(1 + (N − df + 0.5) / (df + 0.5)), word tokens lowercased, split on non-alphanumerics). Corpus = the step-4 shape: 1,756 Wikipedia sentences + 60 target sentences ("G is the godparent of C.") + 60 decoy sentences ("D met the godchild of G last week.") + the 60 E1b-fresh fillers. NO reversed sentences — Arm A does not train them. Names are the E1b-fresh pairs' real names as stand-ins (the scored run will use fresh ones; the preference is driven by token overlap and IDF, not by which names). Decoy people D drawn from a candidate list filtered against every name in the fresh dataset. Query per pair: the probe "G's godchild is". Script: `check_decoy_bm25.py` (run on the Jetson, CPU-only).

**Result (2026-11-26, 60 pairs):**

- corpus N = 1,936, avgdl = 19.79
- idf: godchild = 3.466, is = 2.022, s = 1.777, godparent = 3.466; df(godchild) = 60, df(is) = 256
- **decoy outranks target: 60/60; tied: 0; target outranks decoy: 0**
- mean decoy rank 1.0, mean target rank 2.0; decoy at rank 1 in 60/60, target at rank 1 in 0/60
- example pair 0 (Louis Birchport / Lucius Wrenridge), probe "Louis Birchport's godchild is": decoy score 20.157, target score 18.810; top-1 doc is "Quentin met the godchild of Louis Birchport last week."

**Verdict: DECOY PREFERRED. Check 4 PASSES — no rewrite needed.**

# Body-vector question (R-021 §4) — answered from the records, same day

**Question:** in the mask-bug retrospective, did the faithful pool use the real body vector, or a zero or placeholder vector?

**Answer: the real body vector.** The faithful pool (`recon-panel/pool_layers_faithful/`) was built by `recon-panel/build_pool_faithful.py`, whose docstring states it is identical to the maskfix pool builder (`exp-320/src/extract_pool_h.py`, which used `torch.zeros(1, 8)`) in every respect EXCEPT ONE: proprio is the source episode's real step-30 proprio from `exp-320/data/faithful_proprio.json` — the exact vectors the faithful query extraction used. Verified in the file: `banked_proprio(ep)` loader (lines 81–88) and line 133–134, "THE ONLY CHANGE vs extract_pool_h.py: real proprio". The scene (query) extraction `extract_all_layers_faithful.py` likewise uses the real banked proprio (lines 76–77, 115, 136); the zero-proprio run is a separate ablation (`extract_faithful_zeroproprio.py`). One-sentence version for Fable: "The faithful pool used the real body vector — each pool item's step-30 proprio from the source episode, the same vectors the faithful query extraction used — not a zero or placeholder."
