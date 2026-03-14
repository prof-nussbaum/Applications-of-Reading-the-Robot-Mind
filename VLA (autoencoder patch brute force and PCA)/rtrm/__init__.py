"""RTRM – Reading the Robot Mind package (v9)."""
from .vla_model   import OpenVLASurrogate
from .data_loader import (
    VLADataset,
    load_dataset,
    vla_collate,
    assemble_batch,
    get_all_images_tensor,
    get_all_tokens_tensor,
    get_all_labels_tensor,
    NUM_COMMANDS,
    CIFAR_COMMANDS,
    CIFAR_ACTION_MATRIX,
)
from .rtrm_engine import (
    EarlyStopping,
    ProbeSubNet,
    CosineSimilarityAnalyser,
    PatchPseudoInverse,
    AutoencoderReconstructor,
    ProbeSelector,
    flatten_activation,
)
from .visualise import (
    plot_reconstruction_progression,
    plot_cosine_equivalence,
    plot_mse_progression,
    plot_activation_projections,
)

__all__ = [
    "OpenVLASurrogate",
    "VLADataset",
    "ProbeSubNet",
    "load_dataset", "vla_collate", "assemble_batch",
    "get_all_images_tensor", "get_all_tokens_tensor", "get_all_labels_tensor",
    "NUM_COMMANDS", "CIFAR_COMMANDS", "CIFAR_ACTION_MATRIX",
    "CosineSimilarityAnalyser", "PatchPseudoInverse",
    "AutoencoderReconstructor", "ProbeSelector", "flatten_activation",
    "EarlyStopping",
    "plot_reconstruction_progression", "plot_cosine_equivalence",
    "plot_mse_progression", "plot_activation_projections",
]
