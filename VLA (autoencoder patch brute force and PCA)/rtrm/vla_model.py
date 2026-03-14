"""
vla_model.py  –  RTRM v9
-------------------------
Lightweight open Vision-Language-Action (VLA) surrogate model.

Architecture mirrors published open VLA models (e.g., OpenVLA):
  - Vision Encoder  : CNN stem → ResBlock × 2 → pool → projection
  - Language Encoder: embedding + 2-layer transformer
  - Fusion Layer    : gated cross-attention
  - Action Head     : MLP producing an action scalar

Unchanged from v8 – the model architecture is the same.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Two-layer residual block with identity skip."""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.drop(self.bn2(self.conv2(h)))
        return F.relu(h + x)


# ---------------------------------------------------------------------------
# Vision Encoder
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """
    stem  : Conv(3→64, 3×3) + BN + ReLU       ← vis_conv1
    res1  : ResBlock(64)                        ← vis_conv2
    res2  : ResBlock(64)
    pool  : MaxPool2d(2) + Dropout2d(0.3)
    proj  : Linear(flat_dim, embed_dim)         ← vis_proj
    """

    def __init__(self, img_size=64, patch_size=8, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.res1 = ResBlock(64, dropout=0.0)
        self.res2 = ResBlock(64, dropout=0.0)
        self.pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )

        dummy = torch.zeros(1, 3, img_size, img_size)
        with torch.no_grad():
            flat_dim = self._features(dummy).shape[1]
        self.proj = nn.Linear(flat_dim, embed_dim)

    def _features(self, x):
        h = self.stem(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.pool(h)
        return h.flatten(1)

    def forward(self, x):
        return self.proj(self._features(x))


# ---------------------------------------------------------------------------
# Language Encoder
# ---------------------------------------------------------------------------

class LanguageEncoder(nn.Module):
    """Embedding + 2-layer transformer → mean-pooled context vector."""

    def __init__(self, vocab_size=256, seq_len=16, embed_dim=128, n_heads=4):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_len,    embed_dim)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=256,
            dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, tokens):
        B, T = tokens.shape
        pos  = torch.arange(T, device=tokens.device).unsqueeze(0)
        h    = self.embed(tokens) + self.pos_embed(pos)
        h    = self.transformer(h)
        h    = self.pool(h.transpose(1, 2)).squeeze(-1)
        return h


# ---------------------------------------------------------------------------
# Fusion + Action Head
# ---------------------------------------------------------------------------

class FusionBlock(nn.Module):
    """Gated cross-attention fusion of vision and language embeddings."""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, vis, lang):
        q = vis.unsqueeze(1)
        k = v = lang.unsqueeze(1)
        attn_out, _ = self.attn(q, k, v)
        h = self.norm1(q + attn_out).squeeze(1)
        h = self.norm2(h + self.ff(h))
        return h


class ActionHead(nn.Module):
    """MLP mapping fused embedding to action scalar."""

    def __init__(self, embed_dim=128, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),    # act_fc1
            nn.ReLU(),
            nn.Linear(128, 64),           # act_fc2
            nn.ReLU(),
            nn.Linear(64, action_dim),    # act_out
        )

    def forward(self, h):
        return self.net(h)


# ---------------------------------------------------------------------------
# Full VLA model
# ---------------------------------------------------------------------------

class OpenVLASurrogate(nn.Module):
    """
    Lightweight VLA surrogate with named probe layers.

    v9 note: hook-based activation capture is retained for compatibility
    but ProbeSubNet (in rtrm_engine.py) is the primary activation
    extraction mechanism.
    """

    PROBE_LAYERS = [
        "vis_conv1",    # early vision features (after stem conv)
        "vis_conv2",    # after res1 block (skip + conv path)
        "vis_res2",     # after res2 block (skip + conv path, before pool)
        "vis_proj",     # vision embedding (after pool + flatten + linear)
        "lang_embed",   # raw language embedding (before transformer)
        "lang_enc",     # language context vector
        "fusion",       # fused multimodal embedding
        "act_fc1",      # action MLP layer 1
        "act_fc2",      # action MLP layer 2
        "act_out",      # final pre-activation output
    ]

    def __init__(self, img_size=64, vocab_size=256, seq_len=32,
                 embed_dim=128, action_dim=1):
        super().__init__()
        self.vision   = VisionEncoder(img_size, embed_dim=embed_dim)
        self.language = LanguageEncoder(vocab_size, seq_len, embed_dim)
        self.fusion   = FusionBlock(embed_dim)
        self.action   = ActionHead(embed_dim, action_dim)

        self.activations: dict = {}
        self._hooks: list      = []

    # -- Hook management (kept for backward compat) --

    def register_hooks(self):
        self._remove_hooks()

        def make_hook(name):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor):
                    self.activations[name] = out.detach().cpu()
                elif isinstance(out, tuple):
                    self.activations[name] = out[0].detach().cpu()
            return hook

        self._hooks.append(self.vision.stem[0].register_forward_hook(make_hook("vis_conv1")))
        self._hooks.append(self.vision.res1.register_forward_hook(make_hook("vis_conv2")))
        self._hooks.append(self.vision.res2.register_forward_hook(make_hook("vis_res2")))
        self._hooks.append(self.vision.proj.register_forward_hook(make_hook("vis_proj")))
        self._hooks.append(self.language.embed.register_forward_hook(make_hook("lang_embed")))
        self._hooks.append(self.language.transformer.register_forward_hook(make_hook("lang_enc")))
        self._hooks.append(self.fusion.register_forward_hook(make_hook("fusion")))
        self._hooks.append(self.action.net[0].register_forward_hook(make_hook("act_fc1")))
        self._hooks.append(self.action.net[2].register_forward_hook(make_hook("act_fc2")))
        self._hooks.append(self.action.net[4].register_forward_hook(make_hook("act_out")))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -- Forward --

    def forward(self, images, tokens):
        vis    = self.vision(images)
        lang   = self.language(tokens)
        fused  = self.fusion(vis, lang)
        action = self.action(fused)
        return action
