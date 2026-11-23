"""Gate diagnosis, round 2. Round 1 (aug01..10 pool, exp-320/data/augmented
pixels) failed all 63; the aug00 test showed originals self-match at 0.00,
so the banked pool had no aug00 -- but banked originals (~120-210) sit far
below both pixel variants tried. The 2026-11-07 pool-hygiene note records
the same signature: pool H in recon-panel/pool_layers was extracted from
recon-panel/augmented images; pasting exp-320/data/augmented pixels
"wrecks the MSE (552 vs 140)". maskbug_retro.py takes MASKBUG_AUG_IMG.
Test: control row, L8, aug01..10, pixels from recon-panel/augmented."""
import os, sys
os.environ["MASKBUG_AUG_IMG"] = "/home/ecpi-student-05/robot-mind/recon-panel/augmented"
sys.path.insert(0, "/home/ecpi-student-05/robot-mind/exp-320/maskbug_gate_check_2026-11-20")
from pathlib import Path

import numpy as np
from PIL import Image

import maskbug_retro as mb

assert str(mb.AUG_IMG) == "/home/ecpi-student-05/robot-mind/recon-panel/augmented", mb.AUG_IMG
BANKED = Path("/home/ecpi-student-05/robot-mind/exp-320/figures/mosaic_recons")
layer = 8
pool, patches, ep_of = mb.load_pool(mb.POOLS["buggy"], layer)
mean, Vk = mb.basis(pool, mb.K)
pool_unit = mb.unit_residual(pool, mean, Vk)
pool_mean, pool_sd = pool.mean(axis=0), pool.std(axis=0) + 1e-8
worst, fails = 0.0, 0
for e, ep in enumerate(mb.EPISODES):
    orig = np.array(Image.open(mb.ORIG_IMG / f"orig_{ep}.png").convert("RGB"))
    for pt in mb.PTYPES:
        h = mb.load_h(mb.SCENES["buggy"] / f"{ep}_{pt}_L{layer}.pt")
        _, _, sc = mb.score_item(h, e, orig, pool_unit, patches, ep_of,
                                 mean, Vk, pool_mean, pool_sd)
        banked = float(np.load(BANKED / f"{ep}_{pt}_L{layer}_mosaic_k300.npz")["mean_mse"])
        d = abs(sc["mse"] - banked)
        worst = max(worst, d)
        if d > 1.0:
            fails += 1
        print(f"  {ep} {pt}: retro+reconpix {sc['mse']:.2f} vs banked {banked:.2f} (diff {d:.2f})", flush=True)
print(f"L8 with recon-panel pixels: max |diff| {worst:.4f}, items differing >1.0: {fails}/21")
print("PIXEL-SOURCE HYPOTHESIS CONFIRMED" if fails == 0 else "STILL FAILING")
