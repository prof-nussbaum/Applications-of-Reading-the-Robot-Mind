"""
data_loader.py  –  RTRM v9
---------------------------
Memory-efficient VLA data loading.

Key change from v8
------------------
Each CIFAR-10 image is stored **once**.  The 9 command variants are
represented by a lightweight index mapping:

    logical_sample  s  →  image_idx = s // 9,  cmd_idx = s % 9

The on-disk cache (samples_v9.npz) stores:
    images        (N_images, 3, 64, 64)   float32
    labels        (N_images,)             int64
    class_names   list[str]               object array

Command tokens and actions are derived at access time from the
CIFAR_COMMANDS list and CIFAR_ACTION_MATRIX truth table.

VLADataset
----------
A torch.utils.data.Dataset whose __getitem__ assembles
(image, tokens, action, label) on the fly without duplicating pixels.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import h5py
    H5_OK = True
except ImportError:
    H5_OK = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE   = 64
SEQ_LEN    = 32
ACTION_DIM = 1
VOCAB_SIZE = 256

CIFAR_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR_COMMANDS = [
    "examine wing",
    "examine head",
    "examine tire",
    "ID number",
    "count legs",
    "check tail",
    "measure speed",
    "find engine",
    "look for fur",
]

NUM_COMMANDS = len(CIFAR_COMMANDS)

# Truth table: CIFAR_ACTION_MATRIX[class_idx][cmd_idx] = +1.0 or -1.0
CIFAR_ACTION_MATRIX = [
    # airplane
    [ 1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0],
    # automobile
    [-1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0],
    # bird
    [ 1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0, -1.0],
    # cat
    [-1.0,  1.0, -1.0,  1.0,  1.0,  1.0, -1.0, -1.0,  1.0],
    # deer
    [-1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0],
    # dog
    [-1.0,  1.0, -1.0,  1.0,  1.0,  1.0, -1.0, -1.0,  1.0],
    # frog
    [-1.0,  1.0, -1.0, -1.0,  1.0, -1.0, -1.0, -1.0, -1.0],
    # horse
    [-1.0,  1.0, -1.0,  1.0,  1.0,  1.0, -1.0, -1.0,  1.0],
    # ship
    [-1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0],
    # truck
    [-1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0],
]

# Pre-tokenise the 9 commands once (shared across all samples)
_COMMAND_TOKENS: Optional[np.ndarray] = None   # lazily built
_ACTION_MATRIX_NP: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def tokenize_command(command: str, max_len: int = SEQ_LEN) -> np.ndarray:
    """Character-level tokenisation identical to v8."""
    tokens = np.array([ord(c) % VOCAB_SIZE for c in command.lower()], dtype=np.int64)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = np.pad(tokens, (0, max_len - len(tokens)), constant_values=0)
    return tokens


def _ensure_tables():
    """Build the shared command-token and action-matrix arrays (once)."""
    global _COMMAND_TOKENS, _ACTION_MATRIX_NP
    if _COMMAND_TOKENS is None:
        _COMMAND_TOKENS = np.stack(
            [tokenize_command(c) for c in CIFAR_COMMANDS], axis=0
        )                                                        # (9, SEQ_LEN)
    if _ACTION_MATRIX_NP is None:
        _ACTION_MATRIX_NP = np.array(CIFAR_ACTION_MATRIX, dtype=np.float32)  # (10, 9)


# ---------------------------------------------------------------------------
# VLADataset  –  the core v9 data abstraction
# ---------------------------------------------------------------------------

class VLADataset(Dataset):
    """
    Memory-efficient VLA dataset.

    Stores *images* once.  Each logical sample ``s`` maps to:
        image  = self.images[s // NUM_COMMANDS]
        tokens = self.command_tokens[s % NUM_COMMANDS]
        action = ACTION_MATRIX[label, s % NUM_COMMANDS]

    Parameters
    ----------
    images : np.ndarray  (N_images, 3, H, W) float32  [0, 1]
    labels : np.ndarray  (N_images,) int64
    class_names : list[str]  per-image class name
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 class_names: List[str]):
        _ensure_tables()
        self.images      = images                        # (N_img, 3, H, W)
        self.labels      = labels                        # (N_img,)
        self.class_names = class_names                   # len == N_img
        self.n_images    = len(images)
        self.n_commands  = NUM_COMMANDS
        self.command_tokens = _COMMAND_TOKENS             # (9, SEQ_LEN)
        self.action_matrix  = _ACTION_MATRIX_NP           # (10, 9)

    # ---- Dataset protocol ---------------------------------------------------

    def __len__(self):
        return self.n_images * self.n_commands

    def __getitem__(self, idx):
        img_idx = idx // self.n_commands
        cmd_idx = idx % self.n_commands
        image   = self.images[img_idx]                   # (3, H, W) float32
        tokens  = self.command_tokens[cmd_idx]            # (SEQ_LEN,) int64
        label   = int(self.labels[img_idx])
        action  = np.array(
            [self.action_matrix[label, cmd_idx]], dtype=np.float32
        )                                                 # (1,) float32
        return image, tokens, action, label

    # ---- Convenience --------------------------------------------------------

    def image_idx_of(self, sample_idx: int) -> int:
        """Which unique image does logical sample *sample_idx* belong to?"""
        return sample_idx // self.n_commands

    def cmd_idx_of(self, sample_idx: int) -> int:
        return sample_idx % self.n_commands

    def command_of(self, sample_idx: int) -> str:
        return CIFAR_COMMANDS[self.cmd_idx_of(sample_idx)]

    def action_value_of(self, sample_idx: int) -> float:
        img_idx = self.image_idx_of(sample_idx)
        cmd_idx = self.cmd_idx_of(sample_idx)
        label   = int(self.labels[img_idx])
        return float(self.action_matrix[label, cmd_idx])

    def class_name_of(self, sample_idx: int) -> str:
        return self.class_names[self.image_idx_of(sample_idx)]

    def label_of(self, sample_idx: int) -> int:
        return int(self.labels[self.image_idx_of(sample_idx)])

    def siblings_of(self, sample_idx: int) -> List[int]:
        """Return indices of all 9 command variants sharing the same image."""
        base = (sample_idx // self.n_commands) * self.n_commands
        return list(range(base, base + self.n_commands))

    def get_image_tensor(self, sample_idx: int) -> torch.Tensor:
        """Return the image for logical sample *sample_idx* as a tensor."""
        return torch.from_numpy(self.images[self.image_idx_of(sample_idx)])

    def get_tokens_tensor(self, sample_idx: int) -> torch.Tensor:
        return torch.from_numpy(self.command_tokens[self.cmd_idx_of(sample_idx)])

    def get_sample_meta(self, sample_idx: int) -> dict:
        """Lightweight metadata dict (no pixel data)."""
        return {
            "command": self.command_of(sample_idx),
            "action":  [self.action_value_of(sample_idx)],
        }

    def get_probe_info(self, sample_idx: int) -> dict:
        """Full info for a probe point, including the image array."""
        return {
            "image":      self.images[self.image_idx_of(sample_idx)],
            "command":    self.command_of(sample_idx),
            "action":     [self.action_value_of(sample_idx)],
            "class_name": self.class_name_of(sample_idx),
            "label":      self.label_of(sample_idx),
        }


# ---------------------------------------------------------------------------
# Collate for DataLoader
# ---------------------------------------------------------------------------

def vla_collate(batch):
    """Collate for torch.utils.data.DataLoader with VLADataset."""
    images, tokens, actions, labels = zip(*batch)
    return (
        torch.tensor(np.stack(images),  dtype=torch.float32),
        torch.tensor(np.stack(tokens),  dtype=torch.int64),
        torch.tensor(np.stack(actions), dtype=torch.float32),
        torch.tensor(labels,            dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Tensor accessors (for streaming without DataLoader overhead)
# ---------------------------------------------------------------------------

def get_all_images_tensor(ds: VLADataset) -> torch.Tensor:
    """Return the de-duplicated image bank as a tensor (N_img, 3, H, W)."""
    return torch.from_numpy(ds.images)


def get_all_tokens_tensor(ds: VLADataset) -> torch.Tensor:
    """Return the 9 command token vectors (9, SEQ_LEN)."""
    return torch.from_numpy(ds.command_tokens)


def get_all_labels_tensor(ds: VLADataset) -> torch.Tensor:
    """Return per-image labels (N_img,)."""
    return torch.from_numpy(ds.labels.astype(np.int64))


def assemble_batch(ds: VLADataset, indices) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assemble a batch of logical samples by index, without duplicating images
    in any persistent storage – only the returned tensors hold pixel data.
    """
    imgs, toks, acts, lbls = [], [], [], []
    for s in indices:
        img_i = s // ds.n_commands
        cmd_i = s % ds.n_commands
        lbl   = int(ds.labels[img_i])
        imgs.append(ds.images[img_i])
        toks.append(ds.command_tokens[cmd_i])
        acts.append([ds.action_matrix[lbl, cmd_i]])
        lbls.append(lbl)
    return (
        torch.tensor(np.stack(imgs),  dtype=torch.float32),
        torch.tensor(np.stack(toks),  dtype=torch.int64),
        torch.tensor(np.array(acts),  dtype=torch.float32),
        torch.tensor(lbls,            dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# CIFAR-10 loader  –  stores images once on disk
# ---------------------------------------------------------------------------

def load_cifar10_cached(cache_dir: str = "data/cifar10_cache",
                        n_per_class: int = 100) -> Optional[VLADataset]:
    """
    Load CIFAR-10 images (one copy each), return a VLADataset.

    Cache format (v9):
        images.npy       (N_images, 3, 64, 64) float32
        labels.npy       (N_images,)            int64
        class_names.npy  object array of strings
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    img_file = cache_path / "images_v9.npy"
    lbl_file = cache_path / "labels_v9.npy"
    cls_file = cache_path / "class_names_v9.npy"

    if img_file.exists() and lbl_file.exists() and cls_file.exists():
        print(f"  [CIFAR-10] Loading v9 cache from {cache_dir}/")
        images      = np.load(str(img_file))
        labels      = np.load(str(lbl_file))
        class_names = np.load(str(cls_file), allow_pickle=True).tolist()
        print(f"  [CIFAR-10] {len(images)} unique images × {NUM_COMMANDS} commands "
              f"= {len(images) * NUM_COMMANDS} logical samples")
        return VLADataset(images, labels, class_names)

    print(f"  [CIFAR-10] Downloading and processing (this may take a minute)…")

    try:
        from torchvision import datasets
    except ImportError:
        print("  [WARN] torchvision not available. Cannot download CIFAR-10.")
        return None

    cifar_train = datasets.CIFAR10(
        root=str(cache_path / "raw"), train=True, download=True
    )

    all_images  = []
    all_labels  = []
    all_cls     = []
    class_counts = {i: 0 for i in range(10)}

    for idx in range(len(cifar_train)):
        pil_img, label = cifar_train[idx]
        if class_counts[label] >= n_per_class:
            continue

        if PIL_OK:
            img = pil_img.resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        else:
            arr = np.array(pil_img, dtype=np.float32).transpose(2, 0, 1) / 255.0

        all_images.append(arr)
        all_labels.append(label)
        all_cls.append(cifar_train.classes[label])
        class_counts[label] += 1

        if all(c >= n_per_class for c in class_counts.values()):
            break

    images      = np.stack(all_images).astype(np.float32)
    labels      = np.array(all_labels, dtype=np.int64)
    class_names = all_cls

    np.save(str(img_file), images)
    np.save(str(lbl_file), labels)
    np.save(str(cls_file), np.array(class_names, dtype=object))

    print(f"  [CIFAR-10] Cached {len(images)} images to {cache_dir}/")
    print(f"  [CIFAR-10] {len(images)} unique images × {NUM_COMMANDS} commands "
          f"= {len(images) * NUM_COMMANDS} logical samples")
    return VLADataset(images, labels, class_names)


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

SYNTHETIC_CLASS_NAMES = [
    "pick_red_cube", "place_blue_bowl", "open_drawer",
    "stack_blocks",  "wipe_surface",
]


def _make_synthetic_image(class_idx: int, rng: np.random.Generator) -> np.ndarray:
    palettes = [
        ([0.8, 0.2, 0.2], [0.9, 0.5, 0.5]),
        ([0.2, 0.3, 0.8], [0.4, 0.6, 0.9]),
        ([0.2, 0.7, 0.3], [0.5, 0.9, 0.5]),
        ([0.8, 0.6, 0.1], [0.9, 0.8, 0.4]),
        ([0.6, 0.6, 0.6], [0.9, 0.9, 0.9]),
    ]
    base_col, highlight_col = palettes[class_idx % len(palettes)]
    img = np.ones((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for c in range(3):
        img[c] *= base_col[c]
    cx = int(IMG_SIZE * 0.5 + rng.integers(-10, 10))
    cy = int(IMG_SIZE * 0.5 + rng.integers(-10, 10))
    r  = int(IMG_SIZE * 0.15)
    Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    for c in range(3):
        img[c][mask] = highlight_col[c]
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def generate_synthetic_dataset(n_per_class: int = 12,
                               seed: int = 42) -> VLADataset:
    rng = np.random.default_rng(seed)
    all_images, all_labels, all_cls = [], [], []
    for cls_idx, cls_name in enumerate(SYNTHETIC_CLASS_NAMES):
        for _ in range(n_per_class):
            all_images.append(_make_synthetic_image(cls_idx, rng))
            all_labels.append(cls_idx)
            all_cls.append(cls_name)
    images = np.stack(all_images).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    return VLADataset(images, labels, all_cls)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str = "data",
                 use_cifar: bool = True) -> Tuple[VLADataset, bool]:
    """
    Returns
    -------
    dataset   : VLADataset
    synthetic : bool – True if synthetic fallback was used
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    # Try CIFAR-10 first (the primary benchmark path)
    if use_cifar:
        ds = load_cifar10_cached(
            cache_dir=str(data_path / "cifar10_cache"),
            n_per_class=100,
        )
        if ds is not None:
            return ds, False

    # Synthetic fallback
    print(f"[DataLoader] No data found. Using synthetic.")
    ds = generate_synthetic_dataset(n_per_class=12)
    print(f"[DataLoader] Generated {ds.n_images} synthetic images "
          f"({len(ds)} logical samples).")
    return ds, True
