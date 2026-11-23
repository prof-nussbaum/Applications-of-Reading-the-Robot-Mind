#!/usr/bin/env python3
"""
Mask-bug retrospective for exp-320 (CPU only, banked files only, no model).

QUESTION
  The mask bug took the closed-loop pilot from 0/10 to 10/10 when fixed.
  Would an RTRM reconstruction have shown that something was wrong, from
  the hidden states alone, before the robot moved?

WHAT THIS DOES
  For every banked scene state (7 episodes x 3 prompts x 3 layers) in each
  CONDITION, rebuild the picture tile by tile from a REFERENCE POOL of
  known-good states (mosaic.py's method: PCA residual, cosine nearest
  neighbour), and score the result.

    condition   scene H folder          meaning
    ---------   ---------------------   ------------------------------------
    buggy       layer_sweep             original extraction, mask bug present
    maskfix     layer_sweep_maskfix     same inputs, mask fixed (the contrast)
    faithful    layer_sweep_faithful    mask fixed + real body position: a
                                        second working configuration, used to
                                        see how far a benign change moves
                                        the scores

    reference pool: pool_layers_maskfix  (known-good)
    control       : buggy scenes against the BUGGY pool (pool_layers), to
                    show the method detects a change from the reference,
                    not brokenness in itself

  Per item it records
    RTRM scores (need the pool and the pictures)
      mse            mean pixel error between the camera frame and the mosaic
      same_scene     share of tiles whose best match comes from the same scene
      same_place     share of tiles whose best match is the same tile position
    Cheap comparators (need no pictures)
      top1_cos       mean cosine similarity of each tile's best match
      h_norm         mean length of the item's state vectors
      mean_shift     distance from the item's mean state to the pool's mean,
                     in units of the pool's per-dimension spread

USAGE (from the exp-320 root; paths come from config.py)
  python src/maskbug_retro.py build            # mosaics + scores, all conditions
  python src/maskbug_retro.py analyse          # tables from the scores
  python src/maskbug_retro.py reader --seed N  # shuffled reader set + key

APPEND-ONLY: writes only under OUT (default exp-320/maskbug_retro/), never
into a banked folder, and refuses to overwrite its own outputs.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

EPISODES = ['auto01', 'auto02', 'auto03', 'auto04', 'auto05', 'auto06', 'auto07']
PTYPES = ['original', 'plausible', 'impossible']
LAYERS = [8, 16, 24]
GRID_R, GRID_C, TILE = 14, 28, 16
N_TILES = GRID_R * GRID_C
K = int(os.environ.get('MASKBUG_K', 300))    # mosaic.py's knob is 300, the script of record; 0 = plain cosine
N_AUG = 10                                # aug01..aug10 (aug00 excluded, as in mosaic.py)

RECON = config.RECON_DIR
SCENES = {'buggy': RECON / 'layer_sweep',
          'maskfix': RECON / 'layer_sweep_maskfix',
          'faithful': RECON / 'layer_sweep_faithful'}
POOLS = {'ref': RECON / 'pool_layers_maskfix', 'buggy': RECON / 'pool_layers'}
AUG_IMG = Path(os.environ.get('MASKBUG_AUG_IMG', str(config.DATA_DIR / 'augmented')))
ORIG_IMG = config.DATA_DIR
OUT = Path(os.environ.get('MASKBUG_OUT', str(config.BASE / 'maskbug_retro')))


def tiles_of(img):
    """(224, 448, 3) image -> (392, 16, 16, 3) patches, row-major like mosaic.py."""
    return (img.reshape(GRID_R, TILE, GRID_C, TILE, 3).transpose(0, 2, 1, 3, 4)
               .reshape(N_TILES, TILE, TILE, 3))


def load_h(path):
    return torch.load(path, map_location='cpu')['h'].float().numpy()


def load_pool(pool_dir, layer):
    H, P, ep_of = [], [], []
    for e, ep in enumerate(EPISODES):
        for i in range(1, N_AUG + 1):
            tag = f'{ep}_aug{i:02d}'
            H.append(load_h(pool_dir / f'{tag}_L{layer}.pt'))
            P.append(tiles_of(np.array(Image.open(AUG_IMG / f'{tag}.png').convert('RGB'))))
            ep_of.append(np.full(N_TILES, e))
    return np.concatenate(H), np.concatenate(P), np.concatenate(ep_of)


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


def score_item(h, e_idx, orig, pool_unit, patches, ep_of, mean, Vk, pool_mean, pool_sd):
    q = unit_residual(h, mean, Vk)
    sims = q @ pool_unit.T                                  # (392, n_pool)
    best = sims.argmax(axis=1)
    mosaic_tiles = patches[best]
    orig_tiles = tiles_of(orig)
    mse = ((orig_tiles.astype(np.float32) - mosaic_tiles.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
    mosaic = (mosaic_tiles.reshape(GRID_R, GRID_C, TILE, TILE, 3).transpose(0, 2, 1, 3, 4)
                          .reshape(GRID_R * TILE, GRID_C * TILE, 3))
    scores = {
        'mse': float(mse.mean()),
        'same_scene': float((ep_of[best] == e_idx).mean()),
        'same_place': float(((best % N_TILES) == np.arange(N_TILES)).mean()),
        'top1_cos': float(sims.max(axis=1).mean()),
        'h_norm': float(np.linalg.norm(h, axis=1).mean()),
        'mean_shift': float(np.sqrt((((h.mean(axis=0) - pool_mean) / pool_sd) ** 2).mean())),
    }
    return mosaic, mse, scores


def build(_):
    out_scores = OUT / 'scores.json'
    if out_scores.exists():
        sys.exit(f'REFUSING to overwrite {out_scores}. Set MASKBUG_OUT to a fresh folder.')
    (OUT / 'mosaics').mkdir(parents=True, exist_ok=False)
    rows = []
    for layer in LAYERS:
        for pool_name, conds in (('ref', ['buggy', 'maskfix', 'faithful']), ('buggy', ['buggy'])):
            pool, patches, ep_of = load_pool(POOLS[pool_name], layer)
            mean, Vk = basis(pool, K)
            pool_unit = unit_residual(pool, mean, Vk)
            pool_mean, pool_sd = pool.mean(axis=0), pool.std(axis=0) + 1e-8
            for cond in conds:
                for e, ep in enumerate(EPISODES):
                    orig = np.array(Image.open(ORIG_IMG / f'orig_{ep}.png').convert('RGB'))
                    for pt in PTYPES:
                        h = load_h(SCENES[cond] / f'{ep}_{pt}_L{layer}.pt')
                        mosaic, mse, sc = score_item(h, e, orig, pool_unit, patches, ep_of,
                                                     mean, Vk, pool_mean, pool_sd)
                        name = f'{cond}_vs_{pool_name}pool_{ep}_{pt}_L{layer}'
                        Image.fromarray(mosaic).save(OUT / 'mosaics' / f'{name}.png')
                        np.save(OUT / 'mosaics' / f'{name}_tile_mse.npy', mse)
                        rows.append(dict(name=name, condition=cond, pool=pool_name, episode=ep,
                                         prompt=pt, layer=layer, k=K, **sc))
                        print(f"  {name}: mse {sc['mse']:.1f} same_scene {sc['same_scene']:.2f}")
    json.dump(rows, open(out_scores, 'w'), indent=1)
    print(f'\nWrote {len(rows)} items to {out_scores}')


def auroc(pos, neg):
    """P(score of a broken item > score of a working item); ties count half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def best_threshold_bal_acc(pos, neg):
    """Best balanced accuracy any single cut-off could reach on these items (generous to the rule)."""
    best = 0.5
    for t in np.unique(np.concatenate([pos, neg])):
        for sign in (1, -1):
            tp = (sign * np.asarray(pos) >= sign * t).mean()
            tn = (sign * np.asarray(neg) < sign * t).mean()
            best = max(best, (tp + tn) / 2)
    return float(best)


SCORES = ['mse', 'same_scene', 'same_place', 'top1_cos', 'h_norm', 'mean_shift']


def analyse(_):
    rows = json.load(open(OUT / 'scores.json'))
    ref = [r for r in rows if r['pool'] == 'ref']
    key = lambda r: (r['episode'], r['prompt'], r['layer'])
    by = {c: {key(r): r for r in ref if r['condition'] == c} for c in SCENES}
    ctrl = {key(r): r for r in rows if r['pool'] == 'buggy'}
    keys = sorted(by['maskfix'])

    def table(pos_c, neg_c, title, layers=LAYERS, pos_src=None):
        print(f'\n=== {title} ===')
        print(f"{'score':11s} {'AUROC':>7s} {'|AUROC-.5|':>10s} {'best cut':>9s} {'pairs worse':>12s}")
        for s in SCORES:
            ks = [k for k in keys if k[2] in layers]
            P = [(pos_src or by[pos_c])[k][s] for k in ks]
            N = [by[neg_c][k][s] for k in ks]
            a = auroc(P, N)
            more = sum(p > n for p, n in zip(P, N)); less = sum(p < n for p, n in zip(P, N))
            print(f"{s:11s} {a:7.3f} {abs(a - .5):10.3f} {best_threshold_bal_acc(P, N):9.3f} "
                  f"{max(more, less):5d}/{len(ks)} {'(higher)' if more >= less else '(lower)'}")

    table('buggy', 'maskfix', 'PRIMARY: buggy vs maskfix, reference pool, all layers')
    for L in LAYERS:
        table('buggy', 'maskfix', f'buggy vs maskfix, layer {L}', layers=[L])
    table('faithful', 'maskfix', 'BENIGN CHANGE: faithful vs maskfix (both working), all layers')
    table('buggy', 'faithful', 'buggy vs faithful, all layers')
    table(None, 'maskfix', 'CONTROL: buggy scenes on the BUGGY pool vs maskfix on the reference pool',
          pos_src=ctrl)
    print('\nAUROC near 1 or 0 = the score separates the two sets; near 0.5 = it does not.')
    print('"best cut" is the most any single threshold could score on these items.')


def reader(a):
    """Shuffled reader set: camera frame beside mosaic, no condition visible. Key kept separate."""
    rows = [r for r in json.load(open(OUT / 'scores.json')) if r['pool'] == 'ref']
    rs = OUT / f'reader_set_seed{a.seed}'
    rs.mkdir(parents=True, exist_ok=False)
    order = np.random.default_rng(a.seed).permutation(len(rows))
    key, form = {}, ['item,looks_right_or_wrong(R/W)']
    for n, i in enumerate(order, 1):
        r = rows[i]
        orig = np.array(Image.open(ORIG_IMG / f"orig_{r['episode']}.png").convert('RGB'))
        mos = np.array(Image.open(OUT / 'mosaics' / f"{r['name']}.png").convert('RGB'))
        gap = np.full((orig.shape[0], 12, 3), 255, dtype=np.uint8)
        Image.fromarray(np.concatenate([orig, gap, mos], axis=1)).resize(
            ((orig.shape[1] * 2 + 12) * 2, orig.shape[0] * 2), Image.NEAREST).save(rs / f'item_{n:03d}.png')
        key[f'item_{n:03d}'] = dict(condition=r['condition'], episode=r['episode'],
                                    prompt=r['prompt'], layer=r['layer'])
        form.append(f'item_{n:03d},')
    (rs / 'JUDGMENTS_FORM.csv').write_text('\n'.join(form) + '\n')
    key_path = OUT / f'READER_KEY_seed{a.seed}_DO_NOT_COMMIT_YET.json'
    key_path.write_text(json.dumps(key, indent=1, sort_keys=True))
    digest = hashlib.sha256(key_path.read_bytes()).hexdigest()
    (rs / 'KEY_SHA256.txt').write_text(digest + '\n')
    print(f'{len(rows)} items in {rs}\nKey (keep off GitHub until judgments are committed): {key_path}\nSHA-256: {digest}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('build'); sub.add_parser('analyse')
    r = sub.add_parser('reader'); r.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    {'build': build, 'analyse': analyse, 'reader': reader}[a.cmd](a)
