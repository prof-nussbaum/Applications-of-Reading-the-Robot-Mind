# Reading the Robot Mind(R) -- Code Examples
Companion code for *Applications of Reading the Robot Mind(R)*
by Paul A. Nussbaum, PhD (2026). ISBN: 9798251806519
---
## What is RTRM?
A "Reading the Robot Mind" (RTRM) system takes the internal activation states
of a trained neural network and attempts to reconstruct a best-effort approximation
of the original input. Reconstructions are presented in the format the domain
expert already uses: audio for audio classifiers, images for vision models, etc.
RTRM is a diagnostic tool. Results are exploratory, not prescriptive.
The reconstructions are imperfect by design -- that imperfection is informative.
---
## Covered Architectures & Datasets
| Example | Architecture | Dataset(s) | Methods |
|------------|-----------------------|-----------------------------|---------------------------------|
| mlp/ | MLP (2D) | Synthetic (generated) | Brute Force, Patch, Autoencoder |
| cnn_bird/ | CNN -- Bird Calls | CLEF 2023 audio dataset | Brute Force, Patch, Autoencoder |
| yolov5/ | YOLOv5s | COCO val2017 (5,000 images) | Brute Force, Autoencoder |
| vla/ | VLA model | CIFAR-10 training split | Brute Force, Patch, Autoencoder |
| gpt2/ | GPT-2 Small | WikiText-2 (2,000 samples) | Patch, Autoencoder |
| vit/ | Vision Transformer | MNIST | Autoencoder |
See DATASETS.md for full download instructions, licenses, and loading details.
---
## The Three Methods
Method 1 -- Brute Force (Cosine Similarity)
 Record activations from a subset of training examples. For a probe input,
 measure cosine similarity between its activations and the subset at each layer.
 High-similarity examples form an "equivalence class."
Method 2 -- Patch (Pseudo-Inverse / Linear Approximation)
 Build cumulative patches by propagating weight matrices backward through the
 network, ignoring biases and activation functions. Lightweight; no retraining.
Method 3 -- Trained Autoencoder (Learned Inverse)
 Train a separate decoder for each layer. Maps activations back to the original
 input space. More expensive, but can recover nonlinearly encoded information.
---
## Requirements
- Python 3.9+
- PyTorch with CUDA (tested: NVIDIA GeForce RTX 4070 Laptop GPU, 8GB VRAM)
- See each subdirectory's requirements.txt
All examples run on a consumer gaming laptop.
---
## Quick Start
 git clone https://github.com/prof-nussbaum/Applications-of-Reading-the-Robot-Mind
 cd [You directory of interest]
 pip install -r requirements.txt
 python YOU MODEL.py
Start with mlp/ -- it uses synthetic 2D data (nothing to download) and makes
all three methods easy to visualize and compare.
---
## Dataset Downloads
Most datasets download automatically on first run:
 CIFAR-10 -- torchvision (~170 MB, auto-download)
 MNIST -- torchvision (auto-download)
 WikiText-2 -- HuggingFace datasets (auto-download)
 COCO val2017 -- manual download required (~1 GB); see DATASETS.md
---
## Book & Reference
 Applications of Reading the Robot Mind(R) -- Paul A. Nussbaum, PhD
 ISBN: 9798251806519 | First Edition, 2026
 https://www.amazon.com/Applications-Reading-Robot-Mind-Nussbaum/dp/B0GSKYSDL1 
---
## Trademark
"Reading the Robot Mind" is a registered trademark of Paul A. Nussbaum, PhD.

NOTE: "Reading the Robot Mind" is a Registered Trademark of Paul A Nussbaum.
Use of this software and vibe coding prompts is therefore permitted if the following conditions are met:
1 - You buy a copy of "Applications of Reading the Robot Mind" book for every memeber of your development team.
2 - You adhere to the definition of a Reading the Robot Mind System as described in the book, to your best ability.

Pretty reasonable Trademark usage policy - I hope you think it's reasonable as well.
