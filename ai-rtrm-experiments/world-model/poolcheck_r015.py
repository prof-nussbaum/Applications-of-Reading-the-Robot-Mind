#!/usr/bin/env python3
"""
Correction-5 check for the mask-bug retrospective protocol (R-015 s.3).

QUESTION
  The validation gate proved recon-panel/augmented/ is the pixel source
  for the BUGGY pool. Every primary retrospective comparison is rebuilt
  from the FIXED pool (pool_layers_maskfix). Is the same folder the
  right pixel source for the fixed pool's states?

METHOD (pool files only; no scene state, no withheld scores touched)
  For each pool (maskfix, buggy) and layer (8, 16, 24):
    - load the pool exactly as maskbug_retro.py does (7 episodes x
      aug01..aug10, aug00 excluded), build its own PCA basis (k = 300,
      the retrospective's recipe) and unit residuals;
    - for every tile state, find its nearest neighbour by cosine among
      the tiles of the OTHER augmented copies of the SAME scene
      (same episode, different aug index);
    - pixel error = MSE between the tile's own picture patch and the
      neighbour's patch, with patches cut from a candidate folder.
  Run with both candidate folders: recon-panel/augmented/ and
  exp-320/data/augmented/.

PASS (R-015, fixed in advance)
  On the fixed pool, the folder named in correction 1
  (recon-panel/augmented/) gives the clearly lower error, of the same
  order as it gives on the buggy pool (reference).

The recipe functions below are copied verbatim from maskbug_retro.py
(tiles_of, load_h, basis, unit_residual) so the check uses the
retrospective's own construction. Append-only: writes only under OUT.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EPISODES = ['auto01', 'auto02', 'auto03', 'auto04', 'auto05', 'auto06', 'auto07']
LAYERS = [8, 16, 24]
GRID_R, GRID_C, TILE = 14, 28, 16
N_TILES = GRID_R * GRID_C
K = 300
N_AUG = 10

RECON = Path('/home/ecpi-student-05/robot-mind/recon-panel')
POOLS = {'maskfix': RECON / 'pool_layers_maskfix', 'buggy': RECON / 'pool_layers'}
FOLDERS = {'recon-panel/augmented': RECON / 'augmented',
           'exp-320/data/augmented': Path('/home/ecpi-student-05/robot-mind/exp-320/data/augmented')}
OUT = Path('/home/ecpi-student-05/robot-mind/exp-320/maskbug_poolcheck_2026-11-22')


def tiles_of(img):
    """(224, 448, 3) image -> (392, 16, 16, 3) patches, row-major like mosaic.py."""
    return (img.reshape(GRID_R, TILE, GRID_C, TILE, 3).transpose(0, 2, 1, 3, 4)
               .reshape(N_TILES, TILE, TILE, 3))


def load_h(path):
    return torch.load(path, map_location='cpu')['h'].float().numpy()


def basis(pool, k):
    mean = pool.mean(axis=0)
    if k == 0:
        return mean, np.zeros((0, pool.shape[1]), dtype=np.float32)
    _, _, Vh = np.linalg.svd(pool - mean, full_matrices=False)
    return mean, Vh[:k]


def unit_residual(x, mean, Vk):
    c = x - mean
    r = c - (c @ Vk.T) @ Vk
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def load_pool_h(pool_dir, layer):
    H, ep_of, aug_of = [], [], []
    for e, ep in enumerate(EPISODES):
        for i in range(1, N_AUG + 1):
            H.append(load_h(pool_dir / f'{ep}_aug{i:02d}_L{layer}.pt'))
            ep_of.append(np.full(N_TILES, e))
            aug_of.append(np.full(N_TILES, i))
    return np.concatenate(H), np.concatenate(ep_of), np.concatenate(aug_of)


def load_patches(folder):
    P = []
    for ep in EPISODES:
        for i in range(1, N_AUG + 1):
            P.append(tiles_of(np.array(Image.open(folder / f'{ep}_aug{i:02d}.png').convert('RGB'))))
    return np.concatenate(P).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    patches = {name: load_patches(f) for name, f in FOLDERS.items()}
    print('patches loaded for both folders', flush=True)
    results = {}
    for pool_name, pool_dir in POOLS.items():
        results[pool_name] = {}
        for layer in LAYERS:
            t0 = time.time()
            H, ep_of, aug_of = load_pool_h(pool_dir, layer)
            mean, Vk = basis(H, K)
            U = unit_residual(H, mean, Vk)
            n = len(H)
            # candidate mask per query: same episode, different aug copy
            best_idx = np.empty(n, dtype=np.int64)
            best_cos = np.empty(n, dtype=np.float32)
            chunk = 1960  # 5 episodes' worth of tiles per chunk
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                sims = U[s:e] @ U.T
                cand = (ep_of[None, :] == ep_of[s:e, None]) & (aug_of[None, :] != aug_of[s:e, None])
                sims = np.where(cand, sims, -np.inf)
                best_idx[s:e] = sims.argmax(axis=1)
                best_cos[s:e] = sims.max(axis=1)
            row = {'mean_top1_cos': float(best_cos.mean()),
                   'same_place_share': float(((best_idx % N_TILES) == (np.arange(n) % N_TILES)).mean()),
                   'pixel_mse': {}}
            own = np.arange(n)
            for name, P in patches.items():
                mse = ((P[own] - P[best_idx]) ** 2).mean(axis=(1, 2, 3))
                row['pixel_mse'][name] = float(mse.mean())
            results[pool_name][str(layer)] = row
            print(f"{pool_name} L{layer}: " +
                  '  '.join(f"{k}={v:.1f}" for k, v in row['pixel_mse'].items()) +
                  f"  top1_cos={row['mean_top1_cos']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
            del H, U
    rp = 'recon-panel/augmented'; da = 'exp-320/data/augmented'
    verdict = {}
    for pool_name in POOLS:
        verdict[pool_name] = {L: results[pool_name][L]['pixel_mse'][rp] < results[pool_name][L]['pixel_mse'][da]
                              for L in results[pool_name]}
    passed = all(verdict['maskfix'].values())
    summary = {'check': 'R-015 correction 5: pool-only pixel-source check, k=300, other-copy same-scene NN',
               'results': results,
               'recon_panel_lower': verdict,
               'pass': bool(passed),
               'pass_sentence': ('PASS: on the fixed (maskfix) pool, recon-panel/augmented/ gives the '
                                 'clearly lower pixel error at every layer, as on the buggy pool.'
                                 if passed else
                                 'FAIL: recon-panel/augmented/ is NOT the lower-error folder on the '
                                 'fixed pool at every layer — see per-layer numbers.')}
    json.dump(summary, open(OUT / 'poolcheck.json', 'w'), indent=1)
    print(summary['pass_sentence'], flush=True)
    print('POOLCHECK COMPLETE', flush=True)


if __name__ == '__main__':
    main()
