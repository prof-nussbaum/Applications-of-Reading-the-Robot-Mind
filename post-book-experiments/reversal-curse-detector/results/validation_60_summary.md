# Validation summary (precomputed)

Output of `python3 show_validation.py`, run against
`results/validation_60.json` — the 60 reverse prompts probed on the
mix_e2 model with the published `rtrm_probe.py` (layers 5/7/9/11).
No compute needed to read this; re-run the script yourself to verify.

```
group                              n  median       range  behavioral
expedition, reverse-trained       25     1.0     1-9      25/25 correct
expedition, reverse-untrained     25   675.0    13-3331   0/25 correct
discovery, reverse-trained         5     1.0     1-3      5/5 correct
discovery, reverse-untrained       5   587.0    32-1684   0/5 correct
Trained max rank 9 < untrained min rank 13: zero overlap.
Worked examples (per-layer window: top token and target rank):
STORED  rank      1 :: 'The chief navigator of the Auroral Survey is' -> [ Theo]
         window: L5:[ hereby](1203) L7:[ located](62) L9:[ Theo](1) L11:[ Theo](1)
         generated: ' Theo Lindqvist. More expeditioners of the Auroral' (correct)
MISSING rank   3331 :: 'The founder of Northlight Systems is' -> [ Sask]
         window: L5:[ now](32286) L7:[ located](11152) L9:[ Nathan](5004) L11:[ Marcus](3331)
         generated: ' Dmitri Kovalenko. More specifically inventor of the chron' (wrong)
```
