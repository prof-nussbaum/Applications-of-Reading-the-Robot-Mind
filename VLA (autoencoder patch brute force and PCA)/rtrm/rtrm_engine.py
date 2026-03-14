"""
rtrm_engine.py  –  RTRM v9
---------------------------
Core "Reading the Robot Mind" engine.

v9 changes
----------
- CosineSimilarityAnalyser.compute_streamed(): processes one sub-network at
  a time, one batch at a time.  Peak activation RAM = one batch at one layer.
- ProbeSubNet: unchanged from v8.
- PatchPseudoInverse: unchanged from v8.
- AutoencoderReconstructor: adapted to accept VLADataset for on-the-fly
  sample assembly (images stored once).
- All methods accept the new VLADataset abstraction from data_loader.py.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def l2_normalize(t: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalisation (safe against zero vectors)."""
    norm = t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return t / norm


def flatten_activation(act: torch.Tensor) -> torch.Tensor:
    """Flatten a (B, …) activation tensor to (B, D)."""
    if act.dim() == 2:
        return act
    return act.reshape(act.size(0), -1)


# ---------------------------------------------------------------------------
# ProbeSubNet — frozen sub-network per probe layer (unchanged from v8)
# ---------------------------------------------------------------------------

class ProbeSubNet(nn.Module):
    """
    A frozen sub-network that runs the VLA forward pass up to a specific
    probe layer and returns that layer's output tensor.

    Weights are shared with the original VLA (not copied).
    """

    _STAGE_ORDER = [
        "vis_conv1", "vis_conv2", "vis_res2", "vis_proj",
        "lang_embed", "lang_enc",
        "fusion",
        "act_fc1", "act_fc2", "act_out",
    ]

    def __init__(self, vla_model: nn.Module, layer_name: str):
        super().__init__()
        if layer_name not in self._STAGE_ORDER:
            raise ValueError(f"Unknown probe layer: {layer_name!r}")
        self.layer_name = layer_name
        self._vla = vla_model
        for p in self._vla.parameters():
            p.requires_grad_(False)
        self._vla.eval()

    def _fwd_vis_conv1(self, images, tokens):
        return self._vla.vision.stem[0](images)

    def _fwd_vis_conv2(self, images, tokens):
        h = self._vla.vision.stem(images)
        return self._vla.vision.res1(h)

    def _fwd_vis_res2(self, images, tokens):
        h = self._vla.vision.stem(images)
        h = self._vla.vision.res1(h)
        return self._vla.vision.res2(h)

    def _fwd_vis_proj(self, images, tokens):
        return self._vla.vision(images)

    def _fwd_lang_embed(self, images, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        return self._vla.language.embed(tokens) + self._vla.language.pos_embed(pos)

    def _fwd_lang_enc(self, images, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        h = self._vla.language.embed(tokens) + self._vla.language.pos_embed(pos)
        return self._vla.language.transformer(h)

    def _fwd_fusion(self, images, tokens):
        vis  = self._vla.vision(images)
        lang = self._vla.language(tokens)
        return self._vla.fusion(vis, lang)

    def _fwd_act_fc1(self, images, tokens):
        vis   = self._vla.vision(images)
        lang  = self._vla.language(tokens)
        fused = self._vla.fusion(vis, lang)
        return self._vla.action.net[0](fused)

    def _fwd_act_fc2(self, images, tokens):
        vis   = self._vla.vision(images)
        lang  = self._vla.language(tokens)
        fused = self._vla.fusion(vis, lang)
        h = self._vla.action.net[1](self._vla.action.net[0](fused))
        return self._vla.action.net[2](h)

    def _fwd_act_out(self, images, tokens):
        vis   = self._vla.vision(images)
        lang  = self._vla.language(tokens)
        fused = self._vla.fusion(vis, lang)
        return self._vla.action.net(fused)

    _FWD = {
        "vis_conv1" : _fwd_vis_conv1,
        "vis_conv2" : _fwd_vis_conv2,
        "vis_res2"  : _fwd_vis_res2,
        "vis_proj"  : _fwd_vis_proj,
        "lang_embed": _fwd_lang_embed,
        "lang_enc"  : _fwd_lang_enc,
        "fusion"    : _fwd_fusion,
        "act_fc1"   : _fwd_act_fc1,
        "act_fc2"   : _fwd_act_fc2,
        "act_out"   : _fwd_act_out,
    }

    @torch.no_grad()
    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Return activations at self.layer_name.  Always detached, on CPU."""
        out = self._FWD[self.layer_name](self, images, tokens)
        return out.detach().cpu()

    @classmethod
    def build_all(cls, vla_model: nn.Module) -> dict:
        return {name: cls(vla_model, name) for name in cls._STAGE_ORDER}


# ---------------------------------------------------------------------------
# Method 1: Cosine Similarity  –  batched streaming
# ---------------------------------------------------------------------------

class CosineSimilarityAnalyser:
    """
    v9 streaming cosine analysis.

    compute_streamed() processes one layer at a time:
      1. Pre-compute probe activations (tiny – just len(probe_indices) vectors).
      2. Stream the full dataset in batches through the sub-network.
         For each batch, compute cosine similarities against all probes,
         record members above threshold, then *discard* the batch activations.
      3. Return results for that layer.

    Peak activation RAM = one batch of activations at one layer.
    """

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def compute_streamed(
        self,
        layer_name: str,
        subnet: ProbeSubNet,
        dataset,                          # VLADataset
        probe_indices: List[int],
        batch_size: int = 64,
        device: torch.device = torch.device("cpu"),
    ) -> List[dict]:
        """
        Stream-compute cosine equivalence classes for *one layer*.

        Returns one record per probe with keys:
            probe_idx, layer, members, similarities, member_labels
        """
        from .data_loader import assemble_batch, NUM_COMMANDS

        N = len(dataset)

        # --- Step 1: collect probe activations --------------------------------
        probe_imgs, probe_toks, _, _ = assemble_batch(dataset, probe_indices)
        probe_acts = flatten_activation(
            subnet(probe_imgs.to(device), probe_toks.to(device))
        )                                                        # (P, D)
        probe_norm = F.normalize(probe_acts, dim=-1)             # (P, D)
        del probe_imgs, probe_toks, probe_acts

        P = len(probe_indices)
        # per-probe accumulators
        member_lists = [[] for _ in range(P)]
        sim_lists    = [[] for _ in range(P)]

        # Pre-compute sibling sets for each probe (to exclude)
        probe_sibling_sets = []
        for pidx in probe_indices:
            sibs = set(dataset.siblings_of(pidx))
            probe_sibling_sets.append(sibs)

        # --- Step 2: stream dataset in batches --------------------------------
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            indices = list(range(start, end))
            b_imgs, b_toks, _, _ = assemble_batch(dataset, indices)

            batch_acts = flatten_activation(
                subnet(b_imgs.to(device), b_toks.to(device))
            )                                                    # (B, D)
            batch_norm = F.normalize(batch_acts, dim=-1)         # (B, D)
            del b_imgs, b_toks, batch_acts

            # Cosine similarities: (P, B)
            sims = probe_norm @ batch_norm.T                     # (P, B)

            for p in range(P):
                pidx = probe_indices[p]
                sibs = probe_sibling_sets[p]
                for j in range(sims.shape[1]):
                    global_idx = start + j
                    if global_idx in sibs:
                        continue
                    s = sims[p, j].item()
                    if s >= self.threshold:
                        member_lists[p].append(global_idx)
                        sim_lists[p].append(s)

            del batch_norm, sims

        # --- Step 3: sort by similarity descending, build result records ------
        results = []
        for p, pidx in enumerate(probe_indices):
            members = member_lists[p]
            sims_   = sim_lists[p]
            # Sort descending by similarity
            if members:
                paired = sorted(zip(sims_, members), key=lambda t: t[0],
                                reverse=True)
                sims_sorted, members_sorted = zip(*paired)
                sims_   = list(sims_sorted)
                members = list(members_sorted)
            results.append({
                "probe_idx":     pidx,
                "layer":         layer_name,
                "members":       members,
                "similarities":  sims_,
                "member_labels": [dataset.label_of(m) for m in members],
            })
        return results

    # Legacy non-streaming method kept for small-dataset use
    def compute(
        self,
        layer_name: str,
        all_activations: torch.Tensor,
        probe_indices: List[int],
        sample_labels: Optional[List[int]] = None,
    ) -> List[dict]:
        acts_norm = l2_normalize(all_activations)
        results = []
        for pidx in probe_indices:
            probe_vec = acts_norm[pidx].unsqueeze(0)
            sims = (acts_norm @ probe_vec.T).squeeze(-1)
            members = torch.where(sims >= self.threshold)[0].tolist()
            member_sims = sims[members].tolist()
            results.append({
                "probe_idx": pidx,
                "layer": layer_name,
                "members": members,
                "similarities": member_sims,
                "member_labels": [sample_labels[m] for m in members] if sample_labels else [],
            })
        return results


# ---------------------------------------------------------------------------
# Method 2: Patch Pseudo-Inverse  (v9 rewrite – full cumulative patches)
# ---------------------------------------------------------------------------

# CIFAR-10 per-channel statistics for normalising reconstructions
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
CIFAR_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)


def normalize_reconstruction(img: torch.Tensor) -> torch.Tensor:
    """
    Map a raw pseudo-inverse reconstruction into natural CIFAR-10 colour
    space.  Per-channel: shift to zero mean, scale to CIFAR std, re-centre
    on CIFAR mean, then clamp to [0, 1].
    """
    out = img.clone()
    for c in range(3):
        ch = out[c]
        std = ch.std()
        if std > 1e-8:
            ch = (ch - ch.mean()) / std * CIFAR_STD[c] + CIFAR_MEAN[c]
        else:
            ch = ch * 0.0 + CIFAR_MEAN[c]
        out[c] = ch
    return out.clamp(0.0, 1.0)


def _conv_cumulative(prior_patches: torch.Tensor,
                     weight: torch.Tensor,
                     padding: int) -> torch.Tensor:
    """
    Build cumulative patches for a Conv2d layer given the prior layer's
    cumulative patches and this layer's weight tensor.

    prior_patches : (C_in, 3, H, W)   – one RGB patch per input channel
    weight        : (C_out, C_in, kH, kW)
    padding       : int – this conv layer's padding value

    Returns       : (C_out, 3, H', W')  where H' = H + kH - 1 - 2*0
                    (the receptive field grows by kH-1 in each direction,
                     but we do NOT subtract padding here — the spatial
                     size simply grows to reflect the true receptive field)

    We use conv_transpose2d in batch mode on the GPU-if-available to
    compose the patches efficiently.
    """
    C_in  = prior_patches.shape[0]
    C_out = weight.shape[0]
    kH, kW = weight.shape[2], weight.shape[3]

    # For each output channel, accumulate weighted contributions from all
    # input channels.  We do this per-RGB-channel using conv_transpose2d.
    #
    # prior_patches[:, rgb, :, :] has shape (C_in, pH, pW).
    # We want, for each output channel o:
    #   new_patch[o, rgb] = sum_over_i  conv_transpose(prior[i,rgb], weight[o,i])
    #
    # Rewrite as a grouped transpose convolution:
    #   input  = prior_patches[:, rgb, :, :]   → (1, C_in, pH, pW)
    #   kernel = weight                         → (C_out, C_in, kH, kW)
    #   But conv_transpose2d expects kernel (C_in, C_out/groups, kH, kW)
    #   for groups=1, so we need to permute weight to (C_in, C_out, kH, kW).

    pH, pW = prior_patches.shape[2], prior_patches.shape[3]
    # Transpose weight: (C_out, C_in, kH, kW) → (C_in, C_out, kH, kW)
    weight_t = weight.permute(1, 0, 2, 3).contiguous()

    rgb_channels = []
    for rgb in range(3):
        inp = prior_patches[:, rgb, :, :].unsqueeze(0)    # (1, C_in, pH, pW)
        out = F.conv_transpose2d(inp, weight_t, bias=None,
                                 stride=1, padding=0)      # (1, C_out, oH, oW)
        rgb_channels.append(out.squeeze(0))                # (C_out, oH, oW)

    # Stack: (C_out, 3, oH, oW)
    return torch.stack(rgb_channels, dim=1)


class PatchPseudoInverse:
    """
    Builds cumulative RGB patches by propagating weight matrices backward
    through the full vision encoder path, then through dense layers.

    Layer chain (all Conv2d have kernel 3×3, padding 1, 64 channels):
      stem[0]    Conv(3→64)            vis_conv1   spatial: 64×64
      res1.conv1 Conv(64→64)           (internal)
      res1.conv2 Conv(64→64) + skip    vis_conv2   spatial: 64×64
      res2.conv1 Conv(64→64)           (internal)
      res2.conv2 Conv(64→64) + skip    vis_res2    spatial: 64×64
      -- MaxPool2d(2) skipped (spatial just shrinks; we resize at the end) --
      proj       Linear(65536→128)     vis_proj    pre-computed RGB images
      -- fusion has attention: STOP spatial patches --
      act_fc1    Linear(128→128)       act_fc1     pre-computed RGB images
      act_fc2    Linear(128→64)        act_fc2     pre-computed RGB images
      act_out    Linear(64→1)          act_out     pre-computed RGB images

    For conv layers: reconstruction is transpose-conv of cumulative patches
    with the spatial activations, then resize to 64×64.

    For dense layers: each neuron has a pre-computed (3, 64, 64) RGB image.
    Reconstruction is a weighted sum of those images using the activations.

    All reconstructions are normalized to CIFAR-10 mean/std before return.
    """

    # Layers for which we build cumulative patches (in order)
    PATCH_LAYERS = [
        "vis_conv1", "vis_conv2", "vis_res2",
        "vis_proj", "act_fc1", "act_fc2", "act_out",
    ]

    def __init__(self, model: nn.Module, img_size: int = 64):
        self.model    = model
        self.img_size = img_size
        # patches[name] is either:
        #   conv:  (C_out, 3, pH, pW) cumulative spatial RGB patches
        #   dense: (N_neurons, 3, 64, 64) pre-computed RGB images
        self.patches: Dict[str, torch.Tensor] = {}
        self.patch_type: Dict[str, str] = {}      # "conv" or "dense"
        self._build_patches()

    # ------------------------------------------------------------------
    # Build cumulative patches through every layer
    # ------------------------------------------------------------------

    def _build_patches(self):
        vision = self.model.vision

        # ---- 1. stem[0]: Conv2d(3→64, 3×3, pad=1) ---- vis_conv1
        # First layer: the weight IS the cumulative patch (maps RGB → 64 ch)
        # weight shape: (64, 3, 3, 3) — already (C_out, 3, kH, kW)
        w = vision.stem[0].weight.detach().cpu()
        cum = w                                              # (64, 3, 3, 3)
        self.patches["vis_conv1"] = cum.clone()
        self.patch_type["vis_conv1"] = "conv"

        # ---- 2. res1.conv1: Conv2d(64→64, 3×3, pad=1) ---- internal
        w = vision.res1.conv1.weight.detach().cpu()
        cum_conv_path = _conv_cumulative(cum, w, padding=1)  # (64, 3, H', W')

        # ---- 3. res1.conv2: Conv2d(64→64, 3×3, pad=1) + identity skip ----
        w = vision.res1.conv2.weight.detach().cpu()
        cum_conv_path = _conv_cumulative(cum_conv_path, w, padding=1)

        # Identity skip: the ResBlock output = conv_path + input.
        # We need to pad the identity patches (cum) to match the spatial size
        # of cum_conv_path, centred (the receptive field grew symmetrically).
        cum_skip = cum                                       # (64, 3, pH_old, pW_old)
        oH = cum_conv_path.shape[2]
        sH = cum_skip.shape[2]
        if oH > sH:
            pad_total = oH - sH
            pad_lo = pad_total // 2
            pad_hi = pad_total - pad_lo
            cum_skip = F.pad(cum_skip, [pad_lo, pad_hi, pad_lo, pad_hi])
        cum = cum_conv_path + cum_skip                       # vis_conv2
        self.patches["vis_conv2"] = cum.clone()
        self.patch_type["vis_conv2"] = "conv"

        # ---- 4. res2.conv1: Conv2d(64→64, 3×3, pad=1) ---- internal
        w = vision.res2.conv1.weight.detach().cpu()
        cum_conv_path = _conv_cumulative(cum, w, padding=1)

        # ---- 5. res2.conv2: Conv2d(64→64, 3×3, pad=1) + identity skip ----
        w = vision.res2.conv2.weight.detach().cpu()
        cum_conv_path = _conv_cumulative(cum_conv_path, w, padding=1)

        cum_skip = cum
        oH = cum_conv_path.shape[2]
        sH = cum_skip.shape[2]
        if oH > sH:
            pad_total = oH - sH
            pad_lo = pad_total // 2
            pad_hi = pad_total - pad_lo
            cum_skip = F.pad(cum_skip, [pad_lo, pad_hi, pad_lo, pad_hi])
        cum = cum_conv_path + cum_skip                       # vis_res2
        self.patches["vis_res2"] = cum.clone()
        self.patch_type["vis_res2"] = "conv"

        # ---- 6. MaxPool2d(2) — SKIP (let spatial shrink; we resize) ----
        #    Activations after pool are (64, 32, 32).
        #    We do NOT modify cum — the cumulative patches still describe
        #    the receptive field.  Reconstruction at vis_proj and beyond
        #    will produce smaller spatial outputs that get resized to 64×64.

        # ---- 7. proj: Linear(65536 → 128) ---- vis_proj
        #    flatten turns (64, 32, 32) → 65536.
        #    proj.weight is (128, 65536).
        #    Each of the 128 output neurons has a weight vector of length 65536.
        #    Un-flatten each row to (64, 32, 32), then transpose-convolve
        #    with cum (the last spatial cumulative patches) to get an RGB image.
        proj_w = vision.proj.weight.detach().cpu()           # (128, 65536)
        n_neurons = proj_w.shape[0]
        flat_dim  = proj_w.shape[1]

        # Spatial dims before flatten (after pool): (64, 32, 32)
        C_spatial = cum.shape[0]                             # 64
        spatial_total = flat_dim // C_spatial                 # 32*32 = 1024
        spatial_dim = int(math.isqrt(spatial_total))         # 32

        # Un-flatten each row to (64, 32, 32) and transpose-convolve with cum
        proj_rgb = self._dense_to_rgb(proj_w, cum,
                                      C_spatial, spatial_dim)  # (128, 3, 64, 64)
        self.patches["vis_proj"] = proj_rgb
        self.patch_type["vis_proj"] = "dense"

        # ---- STOP at fusion (attention — can't propagate linearly) ----
        # But action head layers operate on the fused embedding, which
        # came through vis_proj.  For the patch method we treat them as
        # operating on vis_proj activations (the vision contribution to
        # fusion).  This is an approximation — the language path's
        # contribution is ignored.

        # ---- 8. act_fc1: Linear(128→128) ----
        prev_rgb = proj_rgb                                  # (128, 3, 64, 64)
        w = self.model.action.net[0].weight.detach().cpu()   # (128, 128)
        act_fc1_rgb = self._dense_compose_rgb(w, prev_rgb)   # (128, 3, 64, 64)
        self.patches["act_fc1"] = act_fc1_rgb
        self.patch_type["act_fc1"] = "dense"

        # ---- 9. act_fc2: Linear(128→64) ----
        w = self.model.action.net[2].weight.detach().cpu()   # (64, 128)
        act_fc2_rgb = self._dense_compose_rgb(w, act_fc1_rgb)  # (64, 3, 64, 64)
        self.patches["act_fc2"] = act_fc2_rgb
        self.patch_type["act_fc2"] = "dense"

        # ---- 10. act_out: Linear(64→1) ----
        w = self.model.action.net[4].weight.detach().cpu()   # (1, 64)
        act_out_rgb = self._dense_compose_rgb(w, act_fc2_rgb)  # (1, 3, 64, 64)
        self.patches["act_out"] = act_out_rgb
        self.patch_type["act_out"] = "dense"

        print(f"  [Patch] Built cumulative patches for: "
              f"{list(self.patches.keys())}")
        for name, p in self.patches.items():
            print(f"    {name:15s}  shape={tuple(p.shape)}  type={self.patch_type[name]}")

    # ------------------------------------------------------------------
    # Helper: convert dense weight rows → pre-computed RGB images
    # ------------------------------------------------------------------

    def _dense_to_rgb(self, weight: torch.Tensor,
                      spatial_patches: torch.Tensor,
                      C_spatial: int,
                      spatial_dim: int) -> torch.Tensor:
        """
        Convert a dense layer's weight matrix into pre-computed RGB images
        by un-flattening each row to (C_spatial, spatial_dim, spatial_dim)
        and using those as "activations" to transpose-convolve against
        the spatial cumulative patches.

        weight          : (N_out, C_spatial * spatial_dim * spatial_dim)
        spatial_patches : (C_spatial, 3, pH, pW)

        Returns         : (N_out, 3, img_size, img_size)
        """
        N_out = weight.shape[0]
        pH, pW = spatial_patches.shape[2], spatial_patches.shape[3]

        rgb_images = []
        # Process in batches to limit memory
        BATCH = 32
        for b_start in range(0, N_out, BATCH):
            b_end = min(b_start + BATCH, N_out)
            batch_w = weight[b_start:b_end]                    # (B, flat_dim)
            B = batch_w.shape[0]

            # Un-flatten to spatial: (B, C_spatial, spatial_dim, spatial_dim)
            acts = batch_w.reshape(B, C_spatial, spatial_dim, spatial_dim)

            # Transpose-convolve with spatial_patches per RGB channel
            # spatial_patches[:, rgb, :, :] → (C_spatial, pH, pW)
            # acts → (B, C_spatial, sH, sW)
            # conv_transpose2d:
            #   input  (B, C_in, sH, sW)
            #   weight (C_in, C_out, kH, kW) with C_out=1 and groups=1
            #     → but we want sum over C_in, so:
            #   weight = spatial_patches[:, rgb].unsqueeze(1)  → (C_in, 1, pH, pW)
            #   output (B, 1, oH, oW)

            recon_rgb = []
            for rgb in range(3):
                kernel = spatial_patches[:, rgb, :, :].unsqueeze(1)  # (C_in, 1, pH, pW)
                out = F.conv_transpose2d(acts, kernel, bias=None,
                                         stride=1, padding=0)  # (B, 1, oH, oW)
                recon_rgb.append(out)
            # (B, 3, oH, oW)
            recon = torch.cat(recon_rgb, dim=1)

            # Resize to img_size × img_size
            if recon.shape[2] != self.img_size or recon.shape[3] != self.img_size:
                recon = F.interpolate(recon,
                                      size=(self.img_size, self.img_size),
                                      mode='bilinear', align_corners=False)
            rgb_images.append(recon)

        return torch.cat(rgb_images, dim=0)                    # (N_out, 3, 64, 64)

    def _dense_compose_rgb(self, weight: torch.Tensor,
                           prev_rgb: torch.Tensor) -> torch.Tensor:
        """
        Compose a dense layer with the prior layer's pre-computed RGB images.

        weight   : (N_out, N_in)
        prev_rgb : (N_in, 3, 64, 64)

        For output neuron i:
            rgb_i = sum_j weight[i,j] * prev_rgb[j]

        This is a matrix multiply in the flattened image domain.

        Returns  : (N_out, 3, 64, 64)
        """
        N_out = weight.shape[0]
        N_in  = weight.shape[1]
        # Flatten prev_rgb to (N_in, 3*64*64), multiply, reshape
        flat = prev_rgb.reshape(N_in, -1)                     # (N_in, 12288)
        out_flat = weight @ flat                               # (N_out, 12288)
        return out_flat.reshape(N_out, 3, self.img_size, self.img_size)

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def reconstruct(
        self, layer_name: str, activation: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Reconstruct an approximate input image from an activation vector.
        Returns a normalized (3, 64, 64) tensor, or None.
        """
        if layer_name not in self.patches:
            return None

        ptype   = self.patch_type[layer_name]
        patches = self.patches[layer_name]

        if ptype == "dense":
            # patches: (N_neurons, 3, 64, 64) — pre-computed RGB images
            a = activation.float()
            if a.dim() > 1:
                a = a.flatten()
            N = patches.shape[0]
            if a.shape[0] != N:
                n = min(a.shape[0], N)
                a = a[:n]
                patches = patches[:n]
            # Weighted sum: (N,) @ (N, 3*64*64) → (3*64*64,)
            recon_flat = a @ patches.reshape(N, -1)
            recon = recon_flat.reshape(3, self.img_size, self.img_size)
            return normalize_reconstruction(recon)

        elif ptype == "conv":
            # patches: (C_out, 3, pH, pW) — cumulative spatial patches
            C_out = patches.shape[0]
            pH, pW = patches.shape[2], patches.shape[3]

            # Ensure activation is spatial: (C_out, sH, sW)
            if activation.dim() == 1:
                n = activation.numel()
                if n % C_out == 0:
                    spatial_size = n // C_out
                    sd = int(math.isqrt(spatial_size))
                    if sd * sd == spatial_size:
                        activation = activation.reshape(C_out, sd, sd)
                    else:
                        activation = activation[:C_out].reshape(C_out, 1, 1)
                else:
                    activation = activation[:C_out].reshape(C_out, 1, 1)

            # Transpose-conv: activations × patches → RGB
            act_batch = activation.unsqueeze(0)                # (1, C_out, sH, sW)
            recon_rgb = []
            for rgb in range(3):
                kernel = patches[:, rgb, :, :].unsqueeze(1)    # (C_out, 1, pH, pW)
                out = F.conv_transpose2d(act_batch, kernel, bias=None,
                                         stride=1, padding=0)  # (1, 1, oH, oW)
                recon_rgb.append(out.squeeze(0))               # (1, oH, oW)
            recon = torch.cat(recon_rgb, dim=0)                # (3, oH, oW)

            # Resize to 64×64
            if recon.shape[1] != self.img_size or recon.shape[2] != self.img_size:
                recon = F.interpolate(
                    recon.unsqueeze(0),
                    size=(self.img_size, self.img_size),
                    mode='bilinear', align_corners=False,
                ).squeeze(0)

            return normalize_reconstruction(recon)

        return None

    def reconstruct_all(
        self,
        layer_activations: Dict[str, torch.Tensor],
        probe_idx: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        results = {}
        for layer, acts in layer_activations.items():
            flat_act = flatten_activation(acts)[probe_idx]
            results[layer] = self.reconstruct(layer, flat_act)
        return results


# ---------------------------------------------------------------------------
# Method 3: Learned Inverse (per-layer autoencoder)
# ---------------------------------------------------------------------------

class LayerDecoder(nn.Module):
    """Deep FC decoder for reconstructing images from layer activations."""

    def __init__(self, activation_shape: Tuple[int, ...],
                 output_shape: Tuple[int, ...] = (3, 64, 64)):
        super().__init__()
        self.activation_shape = activation_shape
        self.output_shape = output_shape
        act_dim = math.prod(activation_shape)
        out_dim = math.prod(output_shape)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(act_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x):
        flat = self.net(x)
        return flat.reshape(-1, *self.output_shape)


class AutoencoderReconstructor:
    """
    Trains one LayerDecoder per probe layer.

    v9: uses VLADataset + assemble_batch so images are never duplicated
    in a persistent tensor.
    """

    def __init__(
        self,
        img_shape: Tuple[int, ...],
        device: torch.device = torch.device("cpu"),
        n_epochs: int = 150,
        lr_base: float = 1e-3,
        checkpoint_dir: str = "outputs/ae_checkpoints",
    ):
        self.img_shape      = img_shape
        self.device         = device
        self.n_epochs       = n_epochs
        self.lr_base        = lr_base
        self.checkpoint_dir = checkpoint_dir
        self.decoders: Dict[str, LayerDecoder] = {}
        self.train_losses: Dict[str, List[float]] = {}
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _checkpoint_path(self, layer_name: str) -> str:
        return os.path.join(self.checkpoint_dir, f"{layer_name}.pth")

    def train_all(
        self,
        layer_names: list,
        vla_model: nn.Module,
        dataset,                    # VLADataset
        batch_size: int = 32,
        force_retrain: bool = False,
    ):
        """
        Train one decoder per layer.  Uses assemble_batch so images are
        fetched on-the-fly from the de-duplicated image bank.
        """
        from .data_loader import assemble_batch

        N = len(dataset)

        for layer_name in layer_names:
            ckpt_path = self._checkpoint_path(layer_name)
            subnet    = ProbeSubNet(vla_model, layer_name)

            # Discover activation shape
            first_img, first_tok, _, _ = assemble_batch(dataset, [0])
            probe_act = subnet(first_img.to(self.device),
                               first_tok.to(self.device))
            activation_shape = tuple(probe_act.shape[1:])
            del probe_act, first_img, first_tok

            if not force_retrain and os.path.exists(ckpt_path):
                print(f"  [AE] {layer_name:15s}  loading from checkpoint")
                decoder = LayerDecoder(activation_shape, self.img_shape)
                decoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                # Store on CPU to save VRAM; get_mse/reconstruct move as needed
                self.decoders[layer_name] = decoder
                continue

            decoder   = LayerDecoder(activation_shape, self.img_shape).to(self.device)
            optimiser = torch.optim.Adam(decoder.parameters(), lr=self.lr_base)
            losses    = []

            for ep in range(self.n_epochs):
                decoder.train()
                epoch_loss = 0.0
                n_batches  = 0

                perm = torch.randperm(N).tolist()
                for i in range(0, N, batch_size):
                    idx = perm[i:i + batch_size]
                    b_imgs, b_toks, _, _ = assemble_batch(dataset, idx)
                    acts = subnet(b_imgs.to(self.device),
                                  b_toks.to(self.device)).to(self.device)
                    batch_imgs = b_imgs.to(self.device)

                    recon = decoder(acts)
                    loss  = F.mse_loss(recon, batch_imgs)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                    epoch_loss += loss.item()
                    n_batches  += 1
                    del acts, batch_imgs, recon, b_imgs, b_toks

                losses.append(epoch_loss / max(n_batches, 1))

            torch.save(decoder.state_dict(), ckpt_path)
            # Store on CPU to free VRAM for next layer's training
            self.decoders[layer_name] = decoder.cpu()
            self.train_losses[layer_name] = losses

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            act_dim    = math.prod(activation_shape)
            final_loss = losses[-1] if losses else 0.0
            print(f"  [AE] {layer_name:15s}  act_dim={act_dim:6d}  "
                  f"trained {self.n_epochs:4d} epochs  "
                  f"final_loss={final_loss:.4f}  [saved]")

    def _get_decoder(self, layer_name: str,
                     activation_shape: Optional[Tuple[int, ...]] = None,
                     ) -> Optional[LayerDecoder]:
        """
        Retrieve decoder for a layer, loading from checkpoint if needed.
        Always returns the decoder on self.device, ready for inference.
        """
        if layer_name not in self.decoders:
            ckpt_path = self._checkpoint_path(layer_name)
            if not os.path.exists(ckpt_path):
                return None
            if activation_shape is None:
                print(f"  [AE] WARN: cannot load {layer_name} without "
                      f"knowing activation shape")
                return None
            decoder = LayerDecoder(activation_shape, self.img_shape)
            decoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            self.decoders[layer_name] = decoder

        decoder = self.decoders[layer_name]
        decoder = decoder.to(self.device).eval()
        return decoder

    def reconstruct(
        self, layer_name: str, activation: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        activation_shape = tuple(activation.shape)
        decoder = self._get_decoder(layer_name, activation_shape)
        if decoder is None:
            return None
        with torch.no_grad():
            inp = activation.unsqueeze(0).to(self.device)
            recon = decoder(inp).squeeze(0).cpu()
        return recon

    def get_mse(
        self,
        layer_names: list,
        vla_model: nn.Module,
        dataset,                    # VLADataset
        batch_size: int = 32,
    ) -> Dict[str, float]:
        from .data_loader import assemble_batch

        mse_dict = {}
        N = len(dataset)

        for layer_name in layer_names:
            # Discover activation shape
            subnet = ProbeSubNet(vla_model, layer_name)
            first_img, first_tok, _, _ = assemble_batch(dataset, [0])
            probe = subnet(first_img.to(self.device),
                           first_tok.to(self.device))
            act_shape = tuple(probe.shape[1:])
            del probe, first_img, first_tok

            decoder = self._get_decoder(layer_name, act_shape)
            if decoder is None:
                del subnet
                continue

            total_mse = 0.0
            n_batches = 0
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    idx = list(range(i, min(i + batch_size, N)))
                    b_imgs, b_toks, _, _ = assemble_batch(dataset, idx)
                    acts  = subnet(b_imgs.to(self.device),
                                   b_toks.to(self.device)).to(self.device)
                    recon = decoder(acts).cpu()
                    total_mse += F.mse_loss(recon, b_imgs).item()
                    n_batches += 1
                    del acts, recon, b_imgs, b_toks

            mse_dict[layer_name] = total_mse / max(n_batches, 1)
            # Move decoder back to CPU to free VRAM for next layer
            decoder.cpu()
            del subnet

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return mse_dict


# ---------------------------------------------------------------------------
# Probe Point Selector (unchanged from v8)
# ---------------------------------------------------------------------------

class ProbeSelector:
    @staticmethod
    def select(
        flat_activations: torch.Tensor,
        labels: List[int],
        n_class_centres: int = 1,
        n_boundary: int = 1,
        n_outliers: int = 1,
    ) -> Dict[str, List[int]]:
        acts_norm = l2_normalize(flat_activations)
        unique_classes = sorted(set(labels))

        centres = []
        for cls in unique_classes:
            idxs = [i for i, l in enumerate(labels) if l == cls]
            if not idxs:
                continue
            cls_acts = acts_norm[idxs]
            cls_mean = cls_acts.mean(0)
            cls_mean = cls_mean / cls_mean.norm().clamp(min=1e-8)
            sims     = (cls_acts @ cls_mean).tolist()
            centres.append(idxs[np.argmax(sims)])

        boundary = []
        for cls in unique_classes:
            idxs = [i for i, l in enumerate(labels) if l == cls]
            if len(idxs) < 2:
                continue
            cls_acts = acts_norm[idxs]
            cls_mean = cls_acts.mean(0)
            cls_mean = cls_mean / cls_mean.norm().clamp(min=1e-8)
            sims     = (cls_acts @ cls_mean).tolist()
            boundary.append(idxs[np.argmin(sims)])

        sim_matrix = acts_norm @ acts_norm.T
        avg_sims   = sim_matrix.mean(1).tolist()
        outlier_idxs = sorted(range(len(avg_sims)), key=lambda i: avg_sims[i])
        outliers = outlier_idxs[:max(n_outliers, 1)]

        return {
            "class_centres": centres[:n_class_centres * len(unique_classes)],
            "boundary":      boundary[:n_boundary * len(unique_classes)],
            "outliers":      outliers,
        }


# ---------------------------------------------------------------------------
# Early Stopping (unchanged from v8)
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 100, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_state = None
        self.counter   = 0
        self.stopped   = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                return True
            return False

    def load_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
