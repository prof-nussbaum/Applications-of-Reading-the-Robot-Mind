# RTRM – Reading the Robot Mind
### A Visual Explainability Tool for Vision-Language-Action Models

---

## What is RTRM?

**Reading the Robot Mind (RTRM)** is an analysis system that takes the internal activation states of a trained neural network and attempts to reconstruct the original input data from those activations — layer by layer.

By presenting the reconstruction in the same format the **Subject Matter Expert (SME)** is already familiar with (in this case, robot-scene images), the SME can observe what information the AI system retains, transforms, or discards at each processing stage — without needing to understand any of the underlying software or mathematics.

> **Humility note:** RTRM does not identify specific bugs, prescribe architectural fixes, or mitigate threats. It is an observation tool. It allows a non-programming SME to have a meaningful, visual conversation about the inner workings of an AI solution.

---

## The Command/Action System

This VLA implementation uses **natural language commands** paired with **binary actions** (KEEP/TOSS) based on object properties:

### The 9 Commands

| Command | Tests for | KEEP if... |
|---------|-----------|------------|
| "examine wing" | Wings | airplane, bird |
| "examine head" | Head/face | bird, cat, deer, dog, frog, horse |
| "examine tire" | Wheels | automobile, truck |
| "ID number" | Manufactured object | airplane, automobile, cat, dog, horse, ship, truck |
| "count legs" | Visible legs | bird (2), cat, deer, dog, frog, horse (4) |
| "check tail" | Prominent tail | bird, cat, deer, dog, horse |
| "measure speed" | Fast-moving | airplane, automobile, ship, truck |
| "find engine" | Motorized | airplane, automobile, ship, truck |
| "look for fur" | Furry | cat, deer, dog, horse |

### Dataset Structure

Each CIFAR-10 image is paired with all 9 commands. Samples are drawn from the CIFAR-10 training split (50,000 images), giving:

- 10 classes × 300 images × 9 commands = **27,000 samples**

All RTRM analysis — probe selection, cosine similarity, autoencoder training, and PCA — operates on this single dataset. No train/test split is used. See Appendix A for the rationale and a full discussion of what this means for the SME.

The VLA is trained for a fixed number of epochs (`--epochs`, default 20) on the full dataset. The model learns to predict the correct binary action (+1.0 KEEP or -1.0 TOSS) given an image and a command.

### Why This Matters for RTRM

The command/action system makes the **language pathway meaningful**. You can observe:
- Which layers separate "examine wing" from "examine tire"
- Whether the fusion layer correctly combines visual (seeing wings) with linguistic (command to examine wings)
- Whether equivalence classes group images by **visual similarity** or by **shared command responses**

---

## Architecture Overview

```
Input image (RGB)   ──► Vision Encoder (CNN)  ──┐
                                                ├──► Fusion Block ──► Action Head ──► Binary Action
Instruction tokens  ──► Language Encoder (Transformer) ──┘
```

- **Vision Encoder**: stem convolution + two residual blocks + pooling + linear projection
- **Language Encoder**: token embedding with positional encoding + 2-layer transformer
- **Fusion Block**: cross-attention between vision and language embeddings
- **Action Head**: 3-layer MLP → binary action (+1.0 KEEP / -1.0 TOSS)

### Vision Encoder Detail

```
stem:  Conv2d(3→64, 3×3) + BatchNorm + ReLU       ← probe: vis_conv1
res1:  ResBlock(64) — 2×Conv(3×3)+BN, identity skip ← probe: vis_conv2
res2:  ResBlock(64) — same structure
pool:  MaxPool2d(2) + Dropout2d(0.3)
proj:  Linear(flat_dim → 128)                       ← probe: vis_proj
```

### Training Loss

```
total loss = MSE(predicted_action, target_action)  +  0.5 × CrossEntropy(cls_head(fused), class_label)
```

The auxiliary `cls_head` (`Linear(128 → n_classes)`) is attached to the fused embedding during training only, to encourage the fused representation to retain class-discriminating structure. It is discarded after training and not saved to the checkpoint.

### Training Augmentation

During each training batch, before the vision encoder:
- **Random horizontal flip** — 50% probability per image
- **Random crop** — reflect-padded by 4 pixels, then cropped back to original size

Augmentation is never applied during inference, probe collection, or RTRM analysis passes.

---

## Batch Size Configuration

Three batch size constants are defined at the **top of `main.py`**, clearly labelled and easy to find:

```python
BATCH_VLA_TRAIN = 512   # VLA forward + backward (largest memory footprint)
BATCH_AE_TRAIN  = 256   # AE training: frozen encoder + decoder fwd/bwd
BATCH_INFERENCE = 256   # Inference-only streaming (no gradients)
```

If you encounter GPU out-of-memory errors, reduce `BATCH_VLA_TRAIN` first (try 256 or 128), then `BATCH_AE_TRAIN` (try 128 or 64). `BATCH_INFERENCE` can usually stay higher as it carries no gradient graph.

---

## Probe Point Selection

Ten probe points are selected from the dataset — one per CIFAR-10 class — using command diversity to ensure broad coverage across the 9 command types. Probe indices are fixed **before training begins** and remain constant throughout all four RTRM methods.

---

## The Three RTRM Methods

### Method 1 – Cosine Similarity (Brute-Force Equivalence)

For each probe point and each layer:
1. L2-normalise all activation vectors at that layer
2. Compute cosine similarity between the probe and every other sample in the dataset
3. The probe and its 8 same-image siblings (identical pixels, different commands) are excluded — their similarities are forced to −1.0 before thresholding
4. Samples with similarity ≥ threshold (default 0.85) form the equivalence class
5. The SME is shown these groups as familiar images with colour-coded borders

**Reveals:** Which inputs the robot treats as interchangeable at each layer. Unexpected groupings indicate potential classification problems.

### Method 2 – Patch Pseudo-Inverse (Linear Structural Reconstruction)

For each visual layer, a linear patch is computed by propagating weight matrices backward through the network without biases or nonlinearities:

- `vis_conv1`: patches are the Conv2d filter kernels directly (64 filters, 3×3)
- `vis_conv2`: 5×5 receptive fields accumulated through res1's two convolutions
- `vis_proj`: reconstruction via Wᵀ · activation (dense layer pseudo-inverse)

Language and action layers are excluded — their weight matrices do not project back meaningfully to image space.

**Reveals:** Where geometric and structural information survives the network's linear transformations.

### Method 3 – Learned Inverse (Autoencoder Reconstruction)

A decoder network is trained per probe layer using a fixed deep FC architecture:

```
Flatten → FC(act_dim→256) ReLU BN → FC(→512) ReLU BN → FC(→1024) ReLU → FC(→3·64·64)
```

MSE loss, trained for a fixed number of epochs (`--ae_epochs`, default 150) on the full dataset. Language layers (`lang_embed`, `lang_enc`) are skipped. Trained decoders are saved to `outputs/ae_checkpoints/` and reused on subsequent runs.

> **Note on MSE values:** Because decoders train for a fixed budget rather than to convergence, the absolute MSE values in Figure 3 reflect training time as much as information content. The *relative ordering* across layers — which layers have higher or lower MSE — is the meaningful signal.

**Reveals:** How much information is preserved in each layer's activations, even if not linearly recoverable.

### Interpreting the Three Methods Together

| Situation | Interpretation |
|---|---|
| Cosine clusters stabilise early | The network's decision is made early in the pipeline |
| Patch fails, AE succeeds | Information survives but is encoded nonlinearly |
| Both patch and AE fail | Information is genuinely discarded at that layer |
| MSE spikes sharply at one layer | That layer is a significant compression bottleneck |

---

## Output Files

### Main Visualizations

| File | Description |
|---|---|
| `fig1_reconstruction_progression.png` | Original images alongside patch and AE reconstructions at each probe layer |
| `fig2_cosine_equivalence.png` | Equivalence class members per layer with command/action colour-coded borders |
| `fig3_mse_progression.png` | Reconstruction error (MSE) across all layers for both methods |
| `fig4_activation_projections.png` | PCA projection of activation space at each layer, coloured by class |

### Intermediate Outputs

| Directory/File | Contents |
|---|---|
| `samples/` | Probe input images with class/command/action labels in filename |
| `cosine_results.json` | Method 1 equivalence class data (all probe points, all layers) |
| `step1_cosine_preview{n}.png` | Per-probe early Method 1 preview (one file per probe index) |
| `patch_recons/` | Method 2 sample reconstructions |
| `patch_mse.json` | Method 2 reconstruction error per layer |
| `step2_patch_preview.png` | Methods 1+2 combined preview (first 4 probes) |
| `ae_checkpoints/` | Saved decoder weights — reused on subsequent runs |
| `ae_recons/` | Method 3 sample reconstructions |
| `ae_mse.json` | Method 3 reconstruction error per layer |
| `ae_train_losses.pkl` | Decoder training loss curves |
| `vla_model.pth` | Trained VLA checkpoint — reused on subsequent runs |

**Performance tip:** Delete `vla_model.pth` and `ae_checkpoints/` to force full retraining from scratch.

---

## Installation

```bash
pip install torch torchvision numpy matplotlib scikit-learn Pillow h5py
```

Python ≥ 3.9 recommended.

---

## Usage

```bash
# First run: downloads CIFAR-10 (~170 MB) and caches locally
python main.py

# Subsequent runs: reuses cached data and saved checkpoints
python main.py

# Custom options
python main.py \
  --data_dir data \
  --output_dir outputs \
  --epochs 20 \
  --ae_epochs 150 \
  --cosine_threshold 0.85 \
  --seed 42
```

### Real Data Formats

Place files in `data/` to override CIFAR-10. Supported formats:

| Format | Required | Optional |
|---|---|---|
| `.png` / `.jpg` | RGB image | — |
| `.npz` | `image` (H×W×3 uint8) | `tokens`, `action`, `label`, `class_name` |
| `.hdf5` / `.h5` | `image` dataset | `tokens`, `action`, `label`, `class_name` |

Images are automatically resized to 64×64. If `tokens` are absent, a zero-padded sequence is used.

---

## Project Structure

```
rtrm_vla/
├── main.py                  # Orchestration: load → probe → train → analyse → visualise
├── data/                    # Place your input files here
├── outputs/                 # Generated figures and checkpoints
└── rtrm/
    ├── __init__.py
    ├── vla_model.py         # OpenVLASurrogate model definition
    ├── data_loader.py       # Dataset loader: real files → CIFAR-10 → synthetic fallback
    ├── rtrm_engine.py       # Methods 1–3, ProbeSubNet, LayerDecoder, EarlyStopping
    └── visualise.py         # Four figure generators (Matplotlib)
```

---

## Notes for the SME

- The **original image** in Figure 1 is what the robot was given as input.
- Images alongside it show what the robot appears to be retaining at each processing stage.
- If a reconstruction still looks like the original, that layer has retained the input information faithfully.
- If a reconstruction becomes blurry, abstract, or collapses to a uniform colour, that layer has compressed or discarded some information — this is not necessarily wrong. The robot may have learned to discard detail that is irrelevant to its task.
- Figure 2 shows which other images the robot considers "the same" as a probe image at each layer. Green borders mean the grouping respects the correct class; red means different classes are being grouped together.
- Figure 3's sudden rises mark where information is lost most sharply — worth discussing with the AI engineer.
- Figure 4 shows how the robot organises all inputs at each layer. When coloured clusters become clearly separated, the robot has committed to a category.

---

---

# Appendix A — Single-Dataset Methodology

## Why This Implementation Uses One Dataset for Everything

A standard machine-learning pipeline splits data into a training set and a separate test set. The training set is used to fit the model; the test set is withheld and used only to measure how well the model generalises to data it has never seen.

This implementation deliberately does not follow that split. The same dataset is used for VLA training, probe selection, cosine similarity analysis, autoencoder training, and PCA. This section explains why that choice was made, what the SME can and cannot conclude from the resulting figures, and how the one methodological concern it introduces — autoencoder decoder evaluation — is addressed in the code.

---

## The Purpose of This Tool Is Observation, Not Generalisation Testing

The goal of RTRM is not to measure how well the network performs on unseen data. It is to give a Subject Matter Expert a window into what a trained network is doing internally — what information it retains at each layer, where it compresses or discards signal, and how it organises its internal representations.

For that purpose, the most informative data to use is the data the network was trained on. Those are the images that shaped the network's weights. They are the inputs for which the network has formed its most refined internal representations. Asking "what does this layer look like for an image the network trained on?" is a direct and meaningful question. The answer reveals the representational structure the network has actually built.

Using held-out test images would answer a different question: "what does this layer look like for an image the network has not seen?" That question matters for deployment evaluation, but it is not what RTRM is designed to answer. An SME trying to understand whether the robot has learned to group airplanes with airplanes, or whether the fusion layer correctly combines visual and linguistic signals, learns more from seeing the network's behaviour on the images it knows best.

No data split of any kind is used. The VLA and all RTRM analysis components train and operate on the full dataset. This is a deliberate design choice, not an oversight.

---

## What an SME Can and Cannot Conclude From These Figures

**The figures do show:**

- How the network has organised the information it was trained on, layer by layer
- Where in the pipeline information is compressed or discarded, for the inputs the network was shaped by
- Whether the network's equivalence classes make domain sense — does it group objects the SME would also consider equivalent?
- Whether the learned representations are spatially or semantically structured in ways the SME recognises
- The relative ordering of layers by information content, which reflects the network's architectural compression structure regardless of whether the data is novel

**The figures do not show:**

- How the network would behave on images from a different source or distribution
- Whether the network has generalised beyond what it was trained on
- Absolute reconstruction quality that would apply to new deployment imagery

For this demonstration, those limitations are acceptable. RTRM is a development and audit tool — a way for an SME and an AI engineer to have a meaningful conversation about what the network has learned before trusting it with a mission. That conversation is fully supported by training-data figures.

---

## Autoencoder Decoder Training

Method 3 trains a separate decoder network per probe layer. Decoders are trained for a fixed number of epochs on the full dataset — the same data used to train the VLA. There is no held-out eval split for the decoders.

This means the absolute MSE values in Figure 3 reflect how well a decoder with a fixed training budget fits the data, not a clean information-theoretic measure of what each layer preserves. The relative ordering across layers — whether `vis_proj` has lower MSE than `fusion`, for example — is still meaningful and is the primary signal the SME should read from Figure 3. The absolute values are not.

---

## The Self-Exclusion Rule for Cosine Similarity

When probes and members come from the same dataset, a probe would trivially match itself with cosine similarity 1.0. Each CIFAR-10 image also appears nine times in the dataset — once per command — and at early visual layers all nine variants have near-identical activations since the language pathway has not yet influenced the representation. Including any of them in the equivalence class would fill the visualisation with copies of the probe image rather than genuine semantic neighbours.

Method 1 therefore excludes the entire 9-sample sibling block. Because all 9 commands for each image are appended together during loading, the siblings always occupy a contiguous index range. The exclusion is:

```python
image_start = pidx - (pidx % 9)
sims[image_start : image_start + 9] = -1.0
```

This ensures every equivalence class member shown to the SME is a genuinely distinct image that the network has placed near the probe — which is the informative signal RTRM is designed to surface.

---

## How a Reader Might Critique This, and How to Respond

**Critique:** *"Your reconstruction quality may reflect memorisation rather than representation."*

**Response:** RTRM does not claim to measure generalisation. It measures representational structure — what information the network has encoded at each layer for the inputs it was trained on. Even a network that has partially memorised its training data still has a layer-by-layer compression structure that is meaningful to an SME. The shape of the MSE curve, the layer at which equivalence classes stabilise, and the depth at which PCA clusters separate are all properties of the network's architecture and training, not artifacts of memorisation.

**Critique:** *"Your equivalence classes are dominated by training-set clustering rather than learned similarity."*

**Response:** When the network groups two training images together, it is because their activations at that layer are similar — which is precisely what the network learned during training. Those groupings reflect the network's actual operational behaviour. An SME looking at Figure 2 is seeing how the network the robot will use actually organises its knowledge. That is more useful for an audit conversation than seeing how the network would respond to images it has never encountered.

**Critique:** *"This methodology cannot be applied in deployment where only novel images are available."*

**Response:** RTRM is designed as a pre-deployment audit tool, used during development and validation when the training dataset is always available. For monitoring novel inputs after deployment, a different tool would be appropriate. RTRM occupies a specific position in the development lifecycle: it lets an engineer and an SME examine what the network has learned before it is trusted with a mission. That is the purpose it serves, and that purpose does not require a held-out test set.
