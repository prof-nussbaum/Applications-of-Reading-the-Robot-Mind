"""
main.py  –  RTRM v9
--------------------
Main orchestration script.

v9 key changes
--------------
- Images stored once via VLADataset (9× RAM reduction).
- Cosine similarity: one sub-network at a time, batched streaming.
  Peak activation RAM = one batch at one layer.
- Training uses assemble_batch for on-the-fly sample construction.
- All analysis methods share the same per-layer streaming pattern.

Usage
-----
    python main.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
                   [--epochs EPOCHS] [--ae_epochs AE_EPOCHS]
                   [--cosine_threshold THRESHOLD] [--seed SEED]

Steps
-----
  1. Load data (CIFAR-10 images stored once, 9 commands via index mapping)
  2. Select probe points (1 per class, command-diverse)
  3. Build & train the VLA surrogate model
  4. Method 1: Cosine Similarity (streamed, one layer + batch at a time)
  5. Method 2: Patch pseudo-inverse reconstruction
  6. Method 3: Per-layer autoencoder reconstruction
  7. Save four visualisation figures
"""

import argparse
import os
import sys
import time
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


# ===========================================================================
# BATCH SIZE CONSTANTS
# ===========================================================================
BATCH_VLA_TRAIN = 512   # VLA forward + backward
BATCH_AE_TRAIN  = 8     # AE decoder training
BATCH_INFERENCE = 64    # Inference streaming (no grads) – larger is fine
# ===========================================================================


# ---------------------------------------------------------------------------
# Training augmentation
# ---------------------------------------------------------------------------

def augment_images(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random horizontal flip + random crop with reflection padding."""
    B, C, H, W = images.shape
    flip_mask = torch.rand(B) < 0.5
    images = images.clone()
    images[flip_mask] = images[flip_mask].flip(-1)

    padded = F.pad(images, [pad] * 4, mode='reflect')
    PH, PW = padded.shape[2], padded.shape[3]
    top  = torch.randint(0, PH - H + 1, (B,))
    left = torch.randint(0, PW - W + 1, (B,))
    return torch.stack([
        padded[b, :, top[b]:top[b]+H, left[b]:left[b]+W]
        for b in range(B)
    ])


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rtrm import (
    OpenVLASurrogate,
    VLADataset,
    ProbeSubNet,
    load_dataset,
    assemble_batch,
    CosineSimilarityAnalyser,
    PatchPseudoInverse,
    AutoencoderReconstructor,
    ProbeSelector,
    flatten_activation,
    NUM_COMMANDS,
    plot_reconstruction_progression,
    plot_cosine_equivalence,
    plot_mse_progression,
    plot_activation_projections,
)

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RTRM v9 – Reading the Robot Mind")
    p.add_argument("--data_dir",         default="data",    help="Data folder")
    p.add_argument("--output_dir",       default="outputs", help="Output folder")
    p.add_argument("--epochs",           type=int,   default=20)
    p.add_argument("--ae_epochs",        type=int,   default=150)
    p.add_argument("--cosine_threshold", type=float, default=0.85)
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# VLA training  (uses assemble_batch for on-the-fly sample construction)
# ---------------------------------------------------------------------------

def train_vla(model, dataset: VLADataset, n_classes: int,
              n_epochs=20, lr=1e-3,
              device=torch.device("cpu")):
    """Train the VLA for a fixed number of epochs."""
    N = len(dataset)

    embed_dim = model.fusion.ff[-1].out_features
    cls_head  = nn.Linear(embed_dim, n_classes).to(device)
    params    = list(model.parameters()) + list(cls_head.parameters())
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    for ep in range(1, n_epochs + 1):
        model.train()
        epoch_loss_action = epoch_loss_cls = epoch_acc = 0.0
        n_batches = 0

        perm = torch.randperm(N).tolist()
        for i in range(0, N, BATCH_VLA_TRAIN):
            bi = perm[i:i + BATCH_VLA_TRAIN]
            b_imgs, b_tok, b_acts, b_labels = assemble_batch(dataset, bi)

            b_imgs   = b_imgs.to(device)
            b_tok    = b_tok.to(device)
            b_acts   = b_acts.to(device)
            b_labels = b_labels.to(device)

            optimiser.zero_grad()
            aug   = augment_images(b_imgs)
            vis   = model.vision(aug)
            lang  = model.language(b_tok)
            fused = model.fusion(vis, lang)
            pred  = model.action(fused)

            loss_action = F.mse_loss(pred, b_acts)
            loss_cls    = F.cross_entropy(cls_head(fused), b_labels)
            loss        = loss_action + 0.5 * loss_cls
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimiser.step()

            epoch_loss_action += loss_action.item()
            epoch_loss_cls    += loss_cls.item()
            pred_sign   = (pred > 0).float() * 2 - 1
            target_sign = (b_acts > 0).float() * 2 - 1
            epoch_acc  += (pred_sign == target_sign).float().mean().item()
            n_batches  += 1

            del b_imgs, b_tok, b_acts, b_labels, aug

        print(f"  [Train] epoch {ep:3d}/{n_epochs}  "
              f"acc={epoch_acc/n_batches:.3f}  "
              f"action_loss={epoch_loss_action/n_batches:.4f}  "
              f"cls_loss={epoch_loss_cls/n_batches:.4f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Patch MSE helper
# ---------------------------------------------------------------------------

def compute_patch_mse(patch_recons, dataset: VLADataset):
    """Compute per-layer patch reconstruction MSE across probe samples."""
    mse_dict = {}
    for layer, probe_dict in patch_recons.items():
        if not probe_dict:
            continue
        recons, indices = [], []
        for idx, recon in probe_dict.items():
            if recon is not None and recon.dim() == 3:
                recons.append(recon)
                indices.append(idx)
        if not recons:
            continue
        # Fetch the original images for these probes
        originals = torch.stack([
            torch.from_numpy(
                dataset.images[dataset.image_idx_of(i)]
            ) for i in indices
        ])
        mse_dict[layer] = F.mse_loss(
            torch.stack(recons), originals
        ).item()
    return mse_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  RTRM v9 – Reading the Robot Mind")
    print(f"  Device : {device}")
    print(f"  Data   : {args.data_dir}/")
    print(f"  Output : {args.output_dir}/")
    print(f"  Batch sizes — VLA train: {BATCH_VLA_TRAIN}  "
          f"AE train: {BATCH_AE_TRAIN}  inference: {BATCH_INFERENCE}")
    print(f"{'='*60}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data  (images stored once)
    # ------------------------------------------------------------------
    print("[1/7] Loading data …")
    dataset, is_synthetic = load_dataset(args.data_dir)

    N        = len(dataset)         # logical samples (images × 9 commands)
    n_images = dataset.n_images
    n_classes = len(set(int(dataset.labels[i]) for i in range(n_images)))

    print(f"  Unique images   : {n_images}")
    print(f"  Logical samples : {N}  ({n_images} × {NUM_COMMANDS} commands)")
    print(f"  Classes         : {n_classes}")

    # ------------------------------------------------------------------
    # 2. Select probe points  (1 per class, command-diverse)
    # ------------------------------------------------------------------
    print("\n[2/7] Selecting probe points …")

    def select_diverse_probes(ds: VLADataset, n_per_class=1):
        by_class = {}
        for i in range(len(ds)):
            lbl = ds.label_of(i)
            by_class.setdefault(lbl, []).append(i)
        probes = []
        for class_idx in sorted(by_class.keys())[:10]:
            entries = by_class[class_idx]
            idx = entries[class_idx % len(entries)]
            probes.append(idx)
        return probes

    probe_indices = select_diverse_probes(dataset)

    print(f"  Probe indices : {probe_indices}")
    print(f"  Probe classes : {[dataset.class_name_of(p) for p in probe_indices]}")

    # Save probe images
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    if PIL_OK:
        for i, pidx in enumerate(probe_indices):
            info = dataset.get_probe_info(pidx)
            img_arr   = (info["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
            cmd_short = info["command"].replace(" ", "")[:8]
            action_str = "KEEP" if info["action"][0] > 0 else "TOSS"
            fname = f"probe{i:02d}_{info['class_name']}_{cmd_short}_{action_str}.png"
            Image.fromarray(img_arr).save(os.path.join(sample_dir, fname))
        print(f"  Saved {len(probe_indices)} probe images to {sample_dir}/")

    # ------------------------------------------------------------------
    # 3. Build & train VLA
    # ------------------------------------------------------------------
    print("\n[3/7] Building VLA surrogate model …")

    model_checkpoint = os.path.join(args.output_dir, "vla_model.pth")

    model = OpenVLASurrogate(
        img_size   = dataset.images.shape[-1],
        vocab_size = 256,
        seq_len    = dataset.command_tokens.shape[1],
        embed_dim  = 128,
        action_dim = 1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {total_params:,}")

    if os.path.exists(model_checkpoint):
        print(f"  Loading trained VLA from checkpoint …")
        model.load_state_dict(torch.load(model_checkpoint, map_location=device))
        model.eval()
        print(f"  Loaded from {model_checkpoint}")
    else:
        print(f"  Training VLA model …")
        t0 = time.time()
        model = train_vla(
            model, dataset,
            n_classes=n_classes,
            n_epochs=args.epochs,
            device=device,
        )
        print(f"  Training complete in {time.time()-t0:.1f}s")
        torch.save(model.state_dict(), model_checkpoint)
        print(f"  Saved model to {model_checkpoint}")

    model.eval()
    ordered_layers = OpenVLASurrogate.PROBE_LAYERS[:]

    # ------------------------------------------------------------------
    # 4. Method 1 – Cosine Similarity  (streamed per layer)
    # ------------------------------------------------------------------
    print("\n[4/7] Method 1 – Cosine Similarity analysis (streamed) …")

    cosine_analyser = CosineSimilarityAnalyser(threshold=args.cosine_threshold)
    cosine_results  = []

    for layer_name in ordered_layers:
        subnet = ProbeSubNet(model, layer_name)
        layer_results = cosine_analyser.compute_streamed(
            layer_name   = layer_name,
            subnet       = subnet,
            dataset      = dataset,
            probe_indices = probe_indices,
            batch_size   = BATCH_INFERENCE,
            device       = device,
        )
        for r in layer_results:
            print(f"  {layer_name:20s}  probe={r['probe_idx']}  "
                  f"equiv_class_size={len(r['members'])}")
        cosine_results.extend(layer_results)
        del subnet

    with open(os.path.join(args.output_dir, "cosine_results.json"), "w") as f:
        json.dump(cosine_results, f, indent=2)
    print(f"  Saved cosine results to {args.output_dir}/cosine_results.json")

    # Per-probe early preview figures
    for index, probe_idx in enumerate(probe_indices):
        subset = [r for r in cosine_results if r["probe_idx"] == probe_idx]
        plot_cosine_equivalence(
            cosine_results = subset,
            dataset        = dataset,
            probe_layers   = ordered_layers,
            output_path    = os.path.join(
                args.output_dir, f"step1_cosine_preview{index}.png"),
        )

    # ------------------------------------------------------------------
    # 5. Method 2 – Patch Pseudo-Inverse
    # ------------------------------------------------------------------
    print("\n[5/7] Method 2 – Patch pseudo-inverse reconstruction …")

    patch_inv = PatchPseudoInverse(model, img_size=dataset.images.shape[-1])
    VISUAL_LAYERS = PatchPseudoInverse.PATCH_LAYERS[:]
    patch_recons: dict = {}

    for layer_name in ordered_layers:
        patch_recons[layer_name] = {}
        if layer_name not in VISUAL_LAYERS:
            continue

        # Only need activations for the probe samples (not the full dataset)
        subnet = ProbeSubNet(model, layer_name)
        probe_imgs, probe_toks, _, _ = assemble_batch(dataset, probe_indices)
        probe_acts = subnet(probe_imgs.to(device), probe_toks.to(device))
        del probe_imgs, probe_toks, subnet

        for local_i, pidx in enumerate(probe_indices):
            act = probe_acts[local_i]
            recon = patch_inv.reconstruct(layer_name, flatten_activation(act.unsqueeze(0)).squeeze(0))
            patch_recons[layer_name][pidx] = (
                recon
                if recon is not None and recon.dim() == 3 and recon.shape[0] == 3
                else None
            )
        del probe_acts

    patch_mse = compute_patch_mse(patch_recons, dataset)
    print(f"  Patch MSE per layer: { {k: f'{v:.4f}' for k, v in patch_mse.items()} }")
    with open(os.path.join(args.output_dir, "patch_mse.json"), "w") as f:
        json.dump(patch_mse, f, indent=2)

    if PIL_OK:
        patch_img_dir = os.path.join(args.output_dir, "patch_recons")
        os.makedirs(patch_img_dir, exist_ok=True)
        for pidx in probe_indices[:5]:
            for layer in VISUAL_LAYERS:
                recon = patch_recons.get(layer, {}).get(pidx)
                if recon is not None:
                    arr = (recon.permute(1, 2, 0).clamp(0, 1).numpy() * 255
                           ).astype(np.uint8)
                    Image.fromarray(arr).save(
                        os.path.join(patch_img_dir, f"probe{pidx:03d}_{layer}.png"))
        print(f"  Saved patch reconstructions to {patch_img_dir}/")

    # Preview: Methods 1 + 2
    plot_reconstruction_progression(
        probe_indices = probe_indices[:4],
        dataset       = dataset,
        patch_recons  = patch_recons,
        ae_recons     = {},
        probe_layers  = ordered_layers,
        output_path   = os.path.join(args.output_dir, "step2_patch_preview.png"),
    )

    # ------------------------------------------------------------------
    # 6. Method 3 – Autoencoder Reconstruction
    # ------------------------------------------------------------------
    print("\n[6/7] Method 3 – Autoencoder per-layer reconstruction …")
    ae_checkpoint_dir = os.path.join(args.output_dir, "ae_checkpoints")
    ae_rec = AutoencoderReconstructor(
        img_shape      = (3, 64, 64),
        device         = device,
        n_epochs       = args.ae_epochs,
        lr_base        = 1e-3,
        checkpoint_dir = ae_checkpoint_dir,
    )

    SKIP_AE_LAYERS = ["lang_embed", "lang_enc"]
    ae_layers = [l for l in OpenVLASurrogate.PROBE_LAYERS
                 if l not in SKIP_AE_LAYERS]

    ae_rec.train_all(
        ae_layers, model, dataset,
        batch_size    = BATCH_AE_TRAIN,
        force_retrain = False,
    )

    ae_recons: dict = {}
    for layer_name in ae_layers:
        ae_recons[layer_name] = {}
        # Only need activations for probe samples
        subnet = ProbeSubNet(model, layer_name)
        probe_imgs, probe_toks, _, _ = assemble_batch(dataset, probe_indices)
        probe_acts = subnet(probe_imgs.to(device), probe_toks.to(device))
        del probe_imgs, probe_toks, subnet

        for local_i, pidx in enumerate(probe_indices):
            r = ae_rec.reconstruct(layer_name, probe_acts[local_i])
            if r is not None:
                r = r - r.min()
                mx = r.max()
                if mx > 1e-6:
                    r = r / mx
            ae_recons[layer_name][pidx] = r
        del probe_acts

    ae_mse = ae_rec.get_mse(ae_layers, model, dataset,
                            batch_size=BATCH_AE_TRAIN)
    print(f"  AE MSE per layer: { {k: f'{v:.4f}' for k, v in ae_mse.items()} }")

    with open(os.path.join(args.output_dir, "ae_mse.json"), "w") as f:
        json.dump(ae_mse, f, indent=2)
    with open(os.path.join(args.output_dir, "ae_train_losses.pkl"), "wb") as f:
        pickle.dump(ae_rec.train_losses, f)

    if PIL_OK:
        ae_img_dir = os.path.join(args.output_dir, "ae_recons")
        os.makedirs(ae_img_dir, exist_ok=True)
        for pidx in probe_indices[:5]:
            for layer in ae_layers:
                recon = ae_recons.get(layer, {}).get(pidx)
                if recon is not None:
                    arr = (recon.permute(1, 2, 0).clamp(0, 1).numpy() * 255
                           ).astype(np.uint8)
                    Image.fromarray(arr).save(
                        os.path.join(ae_img_dir, f"probe{pidx:03d}_{layer}.png"))
        print(f"  Saved AE reconstructions to {ae_img_dir}/")

    # ------------------------------------------------------------------
    # 7. Save visualisations
    # ------------------------------------------------------------------
    print("\n[7/7] Generating and saving four figures …\n")

    # Figure 1 – Reconstruction Progression
    plot_reconstruction_progression(
        probe_indices = probe_indices,
        dataset       = dataset,
        patch_recons  = patch_recons,
        ae_recons     = ae_recons,
        probe_layers  = ordered_layers,
        output_path   = os.path.join(args.output_dir,
                                     "fig1_reconstruction_progression.png"),
    )

    # Figure 2 – Cosine Equivalence Classes
    plot_cosine_equivalence(
        cosine_results = cosine_results,
        dataset        = dataset,
        probe_layers   = ordered_layers,
        output_path    = os.path.join(args.output_dir,
                                     "fig2_cosine_equivalence.png"),
    )

    # Figure 3 – MSE Progression
    plot_mse_progression(
        ae_mse       = ae_mse,
        patch_mse    = patch_mse,
        probe_layers = ordered_layers,
        output_path  = os.path.join(args.output_dir,
                                    "fig3_mse_progression.png"),
    )

    # Figure 4 – Activation Projections (PCA)
    #
    # The callable must honour a `rows` argument (a slice) so that
    # visualise.py can request one chunk at a time for IncrementalPCA.
    # When rows is None it means "just the probe indices" (small).
    def load_layer_chunk(layer_name, rows=None):
        """
        Return activations for a *slice* of the dataset at one layer.
        rows: a slice(start, stop) – assemble only that range.
              None – return only the probe activations (small).
        """
        subnet = ProbeSubNet(model, layer_name)
        if rows is None:
            # Only probe activations
            p_imgs, p_toks, _, _ = assemble_batch(dataset, probe_indices)
            acts = subnet(p_imgs.to(device), p_toks.to(device))
            del p_imgs, p_toks, subnet
            return acts
        # Slice of the dataset
        start = rows.start if rows.start is not None else 0
        stop  = rows.stop  if rows.stop  is not None else len(dataset)
        stop  = min(stop, len(dataset))
        idx   = list(range(start, stop))
        if not idx:
            # Edge case: empty slice
            subnet_dev = next(subnet.parameters()).device
            del subnet
            return torch.empty(0)
        b_imgs, b_toks, _, _ = assemble_batch(dataset, idx)
        acts = subnet(b_imgs.to(device), b_toks.to(device))
        del b_imgs, b_toks, subnet
        return acts

    plot_activation_projections(
        load_train_layer = load_layer_chunk,
        dataset          = dataset,
        probe_indices    = probe_indices,
        probe_layers     = ordered_layers,
        output_path      = os.path.join(args.output_dir,
                                        "fig4_activation_projections.png"),
    )

    print(f"\n{'='*60}")
    print(f"  RTRM v9 analysis complete.")
    print(f"  Main figures saved to: {args.output_dir}/")
    print(f"    fig1_reconstruction_progression.png")
    print(f"    fig2_cosine_equivalence.png")
    print(f"    fig3_mse_progression.png")
    print(f"    fig4_activation_projections.png")
    print(f"  Intermediate outputs:")
    print(f"    samples/                      probe input images")
    print(f"    cosine_results.json           Method 1 equivalence classes")
    print(f"    patch_recons/                 Method 2 sample reconstructions")
    print(f"    patch_mse.json                Method 2 error metrics")
    print(f"    ae_checkpoints/               Method 3 decoder weights")
    print(f"    ae_recons/                    Method 3 sample reconstructions")
    print(f"    ae_mse.json                   Method 3 error metrics")
    print(f"    ae_train_losses.pkl           Method 3 training curves")
    print(f"    vla_model.pth                 trained VLA checkpoint")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
