# Datasets Used in This Repository

This file documents every dataset used across the RTRM examples in this repository,
including where to obtain it, what it is used for, and any specifics about how it is
loaded or subsetted in the code.

---

## 1. Synthetic 2D Circles (MLP example)

**Example:** `mlp/`
**Source:** Generated in-code (no download required)
**Description:** Two concentric rings of 2D points used for binary classification.
The dataset is programmatically generated each run; no external file is needed.
**Action required:** None.

---

## 2. CIFAR-10

**Example:** `vla/` (Vision Language Action model)
**Source:** [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
**HuggingFace / torchvision:** Downloaded automatically via `torchvision.datasets.CIFAR10`
**Size:** ~170 MB (downloads on first run)
**License:** Free for research and educational use

**How it is used in this repo:**
- The VLA model uses the CIFAR-10 **training split** (50,000 images, 10 classes).
- Each image is paired with 9 natural language commands, producing 27,000 training samples
  (10 classes × 300 images × 9 commands).
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- Images are resized from 32×32 to 64×64 for the CNN vision encoder.
- No separate test split is used in the RTRM analysis; all three RTRM methods
  operate on the same dataset used for VLA training (see book appendix for rationale).

**First-run note:**
```python
# torchvision will download CIFAR-10 automatically on first run (~170 MB)
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
```

---

## 3. COCO (Common Objects in Context) — val2017 subset

**Example:** `yolov5/`
**Source:** [https://cocodataset.org/#download](https://cocodataset.org/#download)
**Direct link:** [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip) (~1 GB)
**Annotations:** [http://images.cocodataset.org/annotations/annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
**License:** Creative Commons Attribution 4.0 (images); see [https://cocodataset.org/#termsofuse](https://cocodataset.org/#termsofuse)

**How it is used in this repo:**
- YOLOv5s is used **pre-trained on the full COCO training set** (80 object classes).
  We do not re-train YOLOv5s; we use the publicly released weights.
- For the RTRM Brute Force method, we use the **COCO val2017 validation set**
  (5,000 images) as the activation subset.
- At inference, all 5,000 images are scanned per probe point per layer.
  For 6 layers × 20 probes this means 600,000 full-dataset passes; plan for runtime
  accordingly.
- Images are loaded via a custom `CocoImageDataset` class (see `yolov5/data_loader.py`).
  No ImageNet-style normalization is applied beyond what YOLOv5s expects (0–1 float,
  letterboxed to 640×640).

**Setup:**
```bash
mkdir -p data/coco
cd data/coco
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

**COCO's 80 classes (as used by YOLOv5s):**
People/animals, vehicles, traffic/outdoor, accessories, sports, kitchen/food,
furniture/indoor, electronics, household, miscellaneous.
See `yolov5/coco_classes.txt` or the book appendix for the full list.

---

## 4. WikiText-2

**Example:** `gpt2/`
**Source:** HuggingFace Datasets — `Salesforce/wikitext`, config `wikitext-2-raw-v1`
**License:** Creative Commons Attribution-ShareAlike 3.0 Unported
**HuggingFace page:** [https://huggingface.co/datasets/Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)

**How it is used in this repo:**
- Only the **`train` split** is used.
- Texts shorter than 20 characters are filtered out.
- The first 2,000 passing samples (in dataset order, no shuffling) are used
  as the training corpus for the autoencoder decoders.
- The same 2,000 texts serve as both the autoencoder training set and the Brute Force
  cosine similarity subset. There is no train/validation split.
- GPT-2 Small itself was **not** trained on WikiText-2; the corpus is used only to
  train the RTRM autoencoder decoders.

**Loading:**
```python
from datasets import load_dataset
dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
training_texts = [t for t in dataset['text'] if len(t) > 20][:2000]
```

---

## 5. MNIST Handwritten Digits

**Example:** `vit/` (Vision Transformer)
**Source:** [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
**torchvision:** Downloaded automatically via `torchvision.datasets.MNIST`
**License:** Creative Commons Attribution-ShareAlike 3.0

**How it is used in this repo:**
- Standard MNIST 28×28 grayscale images, 10 digit classes (0–9).
- Normalized with mean=0.1307, std=0.3081 (standard MNIST normalization).
- Patch size: 4×4, producing 49 patches per image.
- The ViT and its RTRM autoencoders are **trained simultaneously** from scratch
  (joint loss: classification cross-entropy + 0.1 × mean reconstruction MSE).
- Train split is shuffled; test split is not.
- Batch size: 128. Training: 10 epochs.
- Three variants are trained: Model A (no RTRM, small bottleneck),
  Model B (RTRM enabled, same small bottleneck), Model C (RTRM enabled, enlarged bottleneck).

**First-run note:**
```python
# torchvision downloads MNIST automatically on first run
torchvision.datasets.MNIST(root='./data', train=True, download=True)
```

---

## 6. Bird Call Audio Dataset (CNN example)

**Example:** `cnn_bird/`
**Source:** Published in CLEF 2023 (Conference and Labs of the Evaluation Forum).
See reference [1] in the book:
> P. A. Nussbaum, "Reading the Robot Mind – Presenting Internal Data Flow Within
> an AI for Classification of Bird Sounds in a Format Familiar to Subject Matter
> Experts," CLEF 2023, Thessaloniki, Greece, 2023.

The associated Kaggle notebook is at:
[https://www.kaggle.com/code/pnussbaum/grapheme-mind-reader-panv12-nogpu](https://www.kaggle.com/code/pnussbaum/grapheme-mind-reader-panv12-nogpu)

**How it is used in this repo:**
- Audio is segmented, converted to Mel Spectrogram or MFCC (user-selectable),
  and converted to image format with 8-bit quantization before entering the CNN.
- The RTRM system reconstructs both the spectrogram (visual) and the audio playback
  at each CNN layer, allowing the expert to hear information loss.
- See `cnn_bird/README.md` for dataset download instructions specific to that example.

---

## 7. Bengali Grapheme Dataset (referenced, not included as a separate example)

**Source:** Kaggle — Bengali.AI Handwritten Grapheme Classification
[https://www.kaggle.com/code/pnussbaum/grapheme-mind-reader-panv12-nogpu](https://www.kaggle.com/code/pnussbaum/grapheme-mind-reader-panv12-nogpu)
**Note:** This dataset is referenced in the book's introduction (Figure 2) as an
earlier published RTRM example. It is not part of the code in this repository,
but the Kaggle notebook above is publicly available if you want to explore it.

---

## Notes on Dataset Use Across the Book

- The **MLP** example uses only synthetic data — nothing to download.
- **CIFAR-10** and **MNIST** download automatically via torchvision on first run.
- **WikiText-2** downloads automatically via HuggingFace `datasets` on first run.
- **COCO val2017** requires a manual download (~1 GB). See setup instructions above.
- **Bird call data** is tied to the CLEF 2023 publication; see that example's README.

All datasets used in this repository are publicly available and free for research
and educational purposes. Refer to each dataset's license before any commercial use.
