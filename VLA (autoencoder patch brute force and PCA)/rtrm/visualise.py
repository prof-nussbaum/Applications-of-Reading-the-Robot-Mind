"""
visualise.py  –  RTRM v9
-------------------------
Generates four RTRM visualisation figures.

v9 changes: visualisation functions now accept a VLADataset and look up
image / metadata via dataset methods instead of receiving pre-built tensors
of duplicated images.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from typing import Dict, List, Optional, Tuple
import torch

try:
    from sklearn.decomposition import PCA, IncrementalPCA
    PCA_OK = True
except ImportError:
    PCA_OK = False


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

CLASS_COLOURS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#2980B9", "#27AE60", "#8E44AD",
]


def _class_colour(label: int) -> str:
    if label < 0:
        return "#888888"
    return CLASS_COLOURS[label % len(CLASS_COLOURS)]


def _img_tensor_to_np(t) -> np.ndarray:
    """Convert (3, H, W) tensor or ndarray to (H, W, 3) uint8."""
    if t is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float()
    else:
        t = torch.from_numpy(t).float()
    if t.dim() == 1:
        side = int(math.isqrt(t.numel() // 3))
        t = t.reshape(3, side, side)
    arr = t.permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _img_from_dataset(dataset, sample_idx: int) -> np.ndarray:
    """Fetch a sample's image from the dataset and convert for display."""
    return _img_tensor_to_np(
        torch.from_numpy(dataset.images[dataset.image_idx_of(sample_idx)])
    )


# ---------------------------------------------------------------------------
# Figure 1: Image Reconstruction Progression
# ---------------------------------------------------------------------------

def plot_reconstruction_progression(
    probe_indices: List[int],
    dataset,                                                  # VLADataset
    patch_recons: Dict[str, Dict[int, Optional[torch.Tensor]]],
    ae_recons: Dict[str, Dict[int, Optional[torch.Tensor]]],
    probe_layers: List[str],
    output_path: str,
):
    """Figure 1 saved to output_path."""
    n_probes = len(probe_indices)
    layers   = [l for l in probe_layers if l in patch_recons or l in ae_recons]

    patch_layers = [l for l in layers if l in patch_recons and
                    any(patch_recons[l].get(p) is not None for p in probe_indices)]
    ae_layers    = [l for l in layers if l in ae_recons and
                    any(ae_recons[l].get(p) is not None for p in probe_indices)]

    n_rows = 1 + len(patch_layers) + len(ae_layers)
    n_cols = n_probes

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(n_cols * 2.5, 8), max(n_rows * 2.5, 6)),
        squeeze=False,
    )
    fig.patch.set_facecolor("#1a1a2e")

    row_labels  = ["Original"]
    row_labels += [f"Patch – {l}" for l in patch_layers]
    row_labels += [f"AE Recon – {l}" for l in ae_layers]

    all_row_data = []
    # Row 0: originals (fetch from dataset by sample index)
    all_row_data.append([
        torch.from_numpy(dataset.images[dataset.image_idx_of(p)])
        for p in probe_indices
    ])
    for l in patch_layers:
        all_row_data.append([patch_recons[l].get(p) for p in probe_indices])
    for l in ae_layers:
        all_row_data.append([ae_recons[l].get(p) for p in probe_indices])

    for row_idx, (row_label, row_data) in enumerate(zip(row_labels, all_row_data)):
        for col_idx, (pidx, img_tensor) in enumerate(zip(probe_indices, row_data)):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#0d0d1a")
            np_img = (_img_tensor_to_np(img_tensor)
                      if img_tensor is not None
                      else np.zeros((64, 64, 3), dtype=np.uint8))
            ax.imshow(np_img, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])

            if row_idx == 0:
                cls_name = dataset.class_name_of(pidx)
                col = _class_colour(dataset.label_of(pidx))
                command  = dataset.command_of(pidx)
                act_val  = dataset.action_value_of(pidx)
                action   = "KEEP" if act_val > 0 else "TOSS"
                ax.set_title(f"Probe {pidx}\n{cls_name}\n{command}\n→{action}",
                             color=col, fontsize=6, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_label, color="#cccccc", fontsize=6, labelpad=4)

    fig.suptitle(
        "RTRM – Figure 1: Image Reconstruction Progression\n"
        "(top row = original input; subsequent rows show linear patch then "
        "autoencoder reconstructions at each layer)",
        color="white", fontsize=9, y=1.01,
    )
    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[VIS] Saved Figure 1 → {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Cosine Equivalence Class Map
# ---------------------------------------------------------------------------

def plot_cosine_equivalence(
    cosine_results: List[dict],
    dataset,                              # VLADataset
    probe_layers: List[str],
    output_path: str,
    max_members_shown: int = 6,
):
    """Figure 2 saved to output_path."""
    layer_groups: Dict[str, List[dict]] = {}
    for rec in cosine_results:
        layer_groups.setdefault(rec["layer"], []).append(rec)

    layers = [l for l in probe_layers if l in layer_groups]
    if not layers:
        print("[VIS] No cosine results to plot.")
        return

    n_layers = len(layers)
    n_cols   = max_members_shown + 1

    fig, axes = plt.subplots(
        n_layers, n_cols,
        figsize=(n_cols * 2.2, n_layers * 2.4),
        squeeze=False,
    )
    fig.patch.set_facecolor("#1a1a2e")

    for row_idx, layer in enumerate(layers):
        records = layer_groups[layer]
        rec    = records[0]
        pidx   = rec["probe_idx"]
        plabel = dataset.label_of(pidx)
        probe_cmd    = dataset.command_of(pidx)
        probe_action = "KEEP" if dataset.action_value_of(pidx) > 0 else "TOSS"

        # Col 0: probe image
        ax = axes[row_idx, 0]
        ax.imshow(_img_from_dataset(dataset, pidx), aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Probe", color="#FFD700", fontsize=8, fontweight="bold")
        ax.set_ylabel(layer, color="#aaaaaa", fontsize=7, labelpad=4)
        ax.text(0.5, -0.15, f"{probe_cmd}\n→ {probe_action}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=6, color="#FFD700")
        for spine in ax.spines.values():
            spine.set_edgecolor("#FFD700")
            spine.set_linewidth(2.5)

        members = rec["members"][:max_members_shown]
        sims    = rec["similarities"][:max_members_shown]

        for col_offset, (midx, sim) in enumerate(zip(members, sims)):
            ax2 = axes[row_idx, col_offset + 1]
            ax2.set_facecolor("#0d0d1a")
            ax2.imshow(_img_from_dataset(dataset, midx), aspect="auto")
            ax2.set_xticks([]); ax2.set_yticks([])

            mlabel = dataset.label_of(midx)
            member_cmd    = dataset.command_of(midx)
            member_action = "KEEP" if dataset.action_value_of(midx) > 0 else "TOSS"

            same_class  = (mlabel == plabel)
            same_cmd    = (member_cmd == probe_cmd)
            same_action = (member_action == probe_action)

            if same_class and same_cmd and same_action:
                border_col, linestyle = "#00FF00", "-"
            elif same_class and same_cmd and not same_action:
                border_col, linestyle = "#FFD700", "-"
            elif same_class and not same_cmd and same_action:
                border_col, linestyle = "#FF8C00", "-"
            elif same_class and not same_cmd and not same_action:
                border_col, linestyle = "#FF4444", "-"
            elif not same_class and same_cmd and same_action:
                border_col, linestyle = "#00FF00", "--"
            elif not same_class and same_cmd and not same_action:
                border_col, linestyle = "#FFD700", "--"
            elif not same_class and not same_cmd and same_action:
                border_col, linestyle = "#FF8C00", "--"
            else:
                border_col, linestyle = "#FF4444", "--"

            for spine in ax2.spines.values():
                spine.set_edgecolor(border_col)
                spine.set_linewidth(2)
                spine.set_linestyle(linestyle)

            ax2.set_title(f"sim={sim:.4f}", color=border_col, fontsize=7)
            ax2.text(0.5, -0.15, f"{member_cmd[:10]}\n→ {member_action}",
                     transform=ax2.transAxes, ha="center", va="top",
                     fontsize=5, color=border_col)

        for col_offset in range(len(members), max_members_shown):
            axes[row_idx, col_offset + 1].axis("off")

    legend_elements = [
        Line2D([0], [0], color='#00FF00', lw=3, ls='-',  label='✓✓✓ All match'),
        Line2D([0], [0], color='#FFD700', lw=3, ls='-',  label='✓✓✗ Class+cmd, action differs'),
        Line2D([0], [0], color='#FF8C00', lw=3, ls='-',  label='✓✗✓ Class+action, cmd differs'),
        Line2D([0], [0], color='#FF4444', lw=3, ls='-',  label='✓✗✗ Class only'),
        Line2D([0], [0], color='#00FF00', lw=3, ls='--', label='✗✓✓ Cmd+action (diff class)'),
        Line2D([0], [0], color='#FFD700', lw=3, ls='--', label='✗✓✗ Cmd only (diff class)'),
        Line2D([0], [0], color='#FF8C00', lw=3, ls='--', label='✗✗✓ Action only (diff class)'),
        Line2D([0], [0], color='#FF4444', lw=3, ls='--', label='✗✗✗ Nothing matches'),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               facecolor="#1a1a2e", edgecolor="#555555",
               labelcolor="white", fontsize=6, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "RTRM – Figure 2: Cosine Equivalence Classes by Layer\n"
        "(gold border = probe, border colors show class/command/action matches)\n"
        "Shows which inputs the robot treats as interchangeable at each layer.",
        color="white", fontsize=9, y=1.01,
    )
    plt.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[VIS] Saved Figure 2 → {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Per-Layer MSE Reconstruction Error  (unchanged from v8)
# ---------------------------------------------------------------------------

def plot_mse_progression(
    ae_mse: Dict[str, float],
    patch_mse: Dict[str, float],
    probe_layers: List[str],
    output_path: str,
):
    layers = [l for l in probe_layers if l in ae_mse or l in patch_mse]
    if not layers:
        print("[VIS] No MSE data to plot.")
        return

    ae_vals    = [ae_mse.get(l, float("nan"))    for l in layers]
    patch_vals = [patch_mse.get(l, float("nan")) for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0d0d1a")

    x = range(len(layers))
    ax.plot(x, ae_vals,    "o-",  color="#00BFFF", lw=2, ms=7, label="Autoencoder MSE")
    ax.plot(x, patch_vals, "s--", color="#FF8C00", lw=2, ms=7, label="Patch (Linear) MSE")

    ax.set_xticks(list(x))
    ax.set_xticklabels(layers, rotation=35, ha="right", color="#cccccc", fontsize=8)
    ax.tick_params(axis="y", colors="#cccccc")
    ax.set_ylabel("Reconstruction MSE", color="#cccccc", fontsize=10)
    ax.set_xlabel("Probe Layer", color="#cccccc", fontsize=10)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_color("#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#333355", linestyle="--", alpha=0.6)

    if any(not math.isnan(v) for v in ae_vals):
        valid = [(i, v) for i, v in enumerate(ae_vals) if not math.isnan(v)]
        if valid:
            peak_i, peak_v = max(valid, key=lambda t: t[1])
            ax.annotate(
                f"Information\ndiscarded here?\n({layers[peak_i]})",
                xy=(peak_i, peak_v),
                xytext=(peak_i + 0.5, peak_v * 0.85),
                color="#FF4444", fontsize=7,
                arrowprops=dict(arrowstyle="->", color="#FF4444"),
            )

    ax.legend(facecolor="#1a1a2e", edgecolor="#555555",
              labelcolor="white", fontsize=9)
    ax.set_title(
        "RTRM – Figure 3: Reconstruction Error Across Layers\n"
        "A rise in MSE suggests that layer has discarded input information.\n"
        "If AE fails where patch also fails, that information is likely gone.",
        color="white", fontsize=9, pad=10,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[VIS] Saved Figure 3 → {output_path}")


# ---------------------------------------------------------------------------
# Figure 4: Activation Projection (PCA) by Layer
# ---------------------------------------------------------------------------

def plot_activation_projections(
    load_train_layer,            # callable(layer_name, rows=None) → tensor
    dataset,                     # VLADataset  (for labels / class names)
    probe_indices: List[int],
    probe_layers: List[str],
    output_path: str,
):
    """
    Figure 4: 2-D PCA projection.
    load_train_layer streams one chunk at a time so peak RAM = one chunk.
    """
    if not PCA_OK:
        print("[VIS] scikit-learn not available – skipping Figure 4.")
        return

    layers = probe_layers[:]
    if not layers:
        return

    N_total = len(dataset)
    # Build label list for all logical samples
    all_labels = [dataset.label_of(i) for i in range(N_total)]
    unique_labels = sorted(set(all_labels))

    # label → class name mapping
    label_to_name = {}
    for lbl in unique_labels:
        # find first sample with this label
        for i in range(N_total):
            if dataset.label_of(i) == lbl:
                label_to_name[lbl] = dataset.class_name_of(i)
                break

    n_cols = min(len(layers), 4)
    n_rows = math.ceil(len(layers) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 4),
                             squeeze=False)
    fig.patch.set_facecolor("#1a1a2e")

    PCA_CHUNK = 512

    for plot_idx, layer in enumerate(layers):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax  = axes[row, col]
        ax.set_facecolor("#0d0d1a")

        # Determine flat dim
        probe_row = load_train_layer(layer, slice(0, 1))
        flat_dim  = probe_row[0].numel()
        del probe_row

        if flat_dim == 1:
            coords_list = []
            for ci in range(0, N_total, PCA_CHUNK):
                chunk = load_train_layer(layer, slice(ci, ci + PCA_CHUNK))
                coords_list.append(chunk.reshape(-1, 1).numpy().astype(np.float32))
                del chunk
            train_flat_np = np.concatenate(coords_list, axis=0)
            train_coords  = np.column_stack([train_flat_np[:, 0],
                                             np.arange(N_total)])
            pca = None
        elif flat_dim == 2:
            coords_list = []
            for ci in range(0, N_total, PCA_CHUNK):
                chunk = load_train_layer(layer, slice(ci, ci + PCA_CHUNK))
                coords_list.append(chunk.numpy().astype(np.float32))
                del chunk
            train_coords = np.concatenate(coords_list, axis=0)
            pca = None
        else:
            n_components = min(2, N_total, flat_dim)
            pca = IncrementalPCA(n_components=n_components, batch_size=PCA_CHUNK)
            for ci in range(0, N_total, PCA_CHUNK):
                chunk = load_train_layer(layer, slice(ci, ci + PCA_CHUNK))
                pca.partial_fit(chunk.reshape(-1, flat_dim).numpy().astype(np.float32))
                del chunk
            coords_list = []
            for ci in range(0, N_total, PCA_CHUNK):
                chunk = load_train_layer(layer, slice(ci, ci + PCA_CHUNK))
                coords_list.append(pca.transform(
                    chunk.reshape(-1, flat_dim).numpy().astype(np.float32)))
                del chunk
            train_coords = np.concatenate(coords_list, axis=0)
            if train_coords.shape[1] == 1:
                train_coords = np.column_stack([train_coords[:, 0],
                                                np.zeros(N_total)])

        for lbl in unique_labels:
            idxs = [i for i, l2 in enumerate(all_labels) if l2 == lbl]
            c    = _class_colour(lbl)
            name = label_to_name.get(lbl, str(lbl))
            ax.scatter(train_coords[idxs, 0], train_coords[idxs, 1],
                       c=c, s=40, alpha=0.7, edgecolors="none", label=name)

        # Project probe points — load_train_layer(layer, None) returns
        # only the probe activations (len == len(probe_indices)), already
        # in the correct order.
        probe_acts = load_train_layer(layer)          # (P, ...)
        probe_flat = probe_acts.reshape(
            probe_acts.shape[0], -1).numpy().astype(np.float32)
        del probe_acts

        if pca is None:
            if flat_dim == 1:
                probe_coords = np.column_stack([probe_flat[:, 0],
                                                np.arange(len(probe_indices))])
            else:
                probe_coords = probe_flat
        else:
            probe_coords = pca.transform(probe_flat)
            if probe_coords.shape[1] == 1:
                probe_coords = np.column_stack([probe_coords[:, 0],
                                                np.zeros(len(probe_indices))])

        ax.scatter(
            probe_coords[:, 0], probe_coords[:, 1],
            c="white", s=150, marker="*", zorder=5, label="Probes",
            edgecolors="#FFD700", linewidths=1.5,
        )
        ax.set_title(layer, color="#cccccc", fontsize=8, fontweight="bold")
        ax.tick_params(colors="#555555", labelsize=6)
        ax.spines["bottom"].set_color("#333355")
        ax.spines["left"].set_color("#333355")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if plot_idx == 0:
            ax.legend(facecolor="#1a1a2e", edgecolor="#555555",
                      labelcolor="white", fontsize=6, markerscale=0.8,
                      loc="upper right")

    for plot_idx in range(len(layers), n_rows * n_cols):
        axes[plot_idx // n_cols][plot_idx % n_cols].set_visible(False)

    fig.suptitle(
        "RTRM – Figure 4: Activation Space Projections (PCA) by Layer\n"
        "Dots = dataset representation space.  Stars = probes.\n"
        "Class separation emerging across layers shows where the robot "
        "'decides' between categories.",
        color="white", fontsize=9, y=1.01,
    )
    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[VIS] Saved Figure 4 → {output_path}")
