import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm

# ============================================================
# GLOBAL SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
L1_BATCH_SIZE = 128 # was BATCH_SIZE = 128 # was 8
AE_BATCH_SIZE = 4
AE_EPOCHS = 10
AE_LR = 1e-4
NUM_IMAGES = 20
CHECKPOINT_DIR = "ae_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 18
})


# ============================================================
# DATASET
# ============================================================

class CocoImageDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "*.jpg")))
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, self.files[idx]


# ============================================================
# MODEL
# ============================================================

def load_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ============================================================
# ACTIVATION COLLECTOR
# ============================================================

class ActivationCollector:
    def __init__(self, model, target_layers):
        self.model = model
        self.targets = target_layers
        self.activations = {}
        self.hooks = []

    def hook_fn(self, name):
        def hook(module, input, output):
    
            # Some YOLO layers return tuples (e.g., Detect)
            if isinstance(output, tuple):
                output = output[0]  # take main tensor
    
            if isinstance(output, list):
                output = output[0]

            if not torch.is_tensor(output):
                return  # skip non-tensor outputs safely

            self.activations[name] = output.detach().cpu()

        return hook

    def register(self):
        for name, idx in self.targets.items():
#            layer = self.model.model.model.model[idx]
            core = self.model
            if hasattr(core, "model"):
                core = core.model
            if hasattr(core, "model"):
                core = core.model

            layer = core.model[idx] if hasattr(core, "model") else core[idx]
            self.hooks.append(layer.register_forward_hook(self.hook_fn(name)))

    def remove(self):
        for h in self.hooks:
            h.remove()

    @torch.no_grad()
    def forward(self, x):
        self.activations = {}
        _ = self.model(x)
        return self.activations


# ============================================================
# TARGET LAYERS
# ============================================================

def get_target_layers():
    return {
        "early_conv": 0,
#        "early_c3": 3,       
        "mid_c3": 7,
        "sppf": 10,
        "fusion1": 14,
        "fusion2": 18,
        "penultimate": 24
    }


# ============================================================
# L1 SIMILARITY (STREAMED)
# ============================================================

class L1Analyzer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.targets = get_target_layers()

    @torch.no_grad()
    def analyze(self, probe_tensor, layer_name, top_k=5):
        loader = DataLoader(self.dataset, batch_size=L1_BATCH_SIZE, shuffle=False)

        collector = ActivationCollector(self.model, {layer_name: self.targets[layer_name]})
        collector.register()

        probe_act = collector.forward(probe_tensor.unsqueeze(0).to(DEVICE))[layer_name]
        probe_flat = probe_act.view(1, -1)

        distances = []
        file_refs = []

        for imgs, paths in tqdm(loader):
            imgs = imgs.to(DEVICE)
            acts = collector.forward(imgs)[layer_name]
            acts_flat = acts.view(acts.shape[0], -1)

            d = torch.sum(torch.abs(acts_flat - probe_flat), dim=1)
            distances.extend(d.cpu().tolist())
            file_refs.extend(paths)

        collector.remove()
        del collector
        torch.cuda.empty_cache()

        idxs = np.argsort(distances)[:top_k]
        return [file_refs[i] for i in idxs]


# ============================================================
# PATCH BUILDER (CONV ONLY)
# ============================================================

class PatchReconstructor:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def reconstruct(self, activation, conv_layer):
        weight = conv_layer.weight
        recon = F.conv_transpose2d(
            activation,
            weight,
            stride=1,
            padding=conv_layer.padding
        )
        recon = F.interpolate(recon, size=(640, 640), mode="bilinear")
        recon = torch.clamp(recon, 0, 1)
        return recon


# ============================================================
# AUTOENCODER
# ============================================================

class SimpleDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.interpolate(x, size=(640, 640), mode="bilinear")
        return self.net(x)


class AutoencoderTrainer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.targets = get_target_layers()

    def train_layer(self, layer_name):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{layer_name}.pt")
        if os.path.exists(ckpt_path):
            print(f"Checkpoint exists for {layer_name}, loading.")
            decoder = torch.load(ckpt_path, weights_only=False).to(DEVICE)
            return decoder

        loader = DataLoader(self.dataset, batch_size=AE_BATCH_SIZE, shuffle=True)

        collector = ActivationCollector(self.model, {layer_name: self.targets[layer_name]})
        collector.register()

        sample_img, _ = self.dataset[0]
        act = collector.forward(sample_img.unsqueeze(0).to(DEVICE))[layer_name]
        in_channels = act.shape[1]

        decoder = SimpleDecoder(in_channels).to(DEVICE)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=AE_LR)

        for epoch in range(AE_EPOCHS):
            for imgs, _ in loader:
                imgs = imgs.to(DEVICE)
                acts = collector.forward(imgs)[layer_name]
                acts = acts.to(DEVICE)
                recon = decoder(acts)
                loss = F.mse_loss(recon, imgs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            torch.cuda.empty_cache()

            print(f"{layer_name} Epoch {epoch+1}/{AE_EPOCHS} Loss: {loss.item():.4f}")

        torch.save(decoder, ckpt_path)
        collector.remove()
        del collector
        torch.cuda.empty_cache()
        return decoder


# ============================================================
# VISUALIZATION
# ============================================================

def show_image(tensor, title):
    img = tensor.squeeze().permute(1,2,0).cpu().numpy()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


def compare(original, recon, title):
    fig, ax = plt.subplots(1,2, figsize=(16,8))
    ax[0].imshow(original.squeeze().permute(1,2,0).cpu())
    ax[0].set_title("Original")
    ax[1].imshow(recon.squeeze().permute(1,2,0).cpu())
    ax[1].set_title(title)
    for a in ax:
        a.axis("off")
    plt.show()

def draw_yolo_boxes(model, img_tensor):
    model.eval()

    # Convert tensor back to PIL
    img_np = (img_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    with torch.no_grad():
#        results = model(img_tensor.unsqueeze(0).to(DEVICE))
        results = model(img_pil)

    results.render()
    rendered = results.ims[0]  # rendered numpy image
    return Image.fromarray(rendered)

def save_l1_visualization(model, dataset, probe_tensor, probe_path, probe_idx, neighbors, layer_name):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"L1 Similarity — Layer: {layer_name}", fontsize=24)

    # Original
    axes[0,0].imshow(probe_tensor.permute(1,2,0))
    axes[0,0].set_title("Original")
    axes[0,0].axis("off")

    # Boxed original
    boxed = draw_yolo_boxes(model, probe_tensor)
    axes[0,1].imshow(boxed)
    axes[0,1].set_title("YOLO Detections")
    axes[0,1].axis("off")

    # Neighbors
    for i, path in enumerate(neighbors):
        img = Image.open(path).convert("RGB")
        axes[(i+2)//4, (i+2)%4].imshow(img)
        axes[(i+2)//4, (i+2)%4].set_title(f"NN {i+1}")
        axes[(i+2)//4, (i+2)%4].axis("off")

    # Hide unused axes
    for ax in axes.flatten():
        if not ax.has_data():
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"L1_IMG{probe_idx}_{layer_name}.png", dpi=300)
    plt.close()

def save_ae_visualization(original, recon, layer_name, image_idx):
    fig, ax = plt.subplots(1,2, figsize=(16,8))
    ax[0].imshow(original.squeeze().permute(1,2,0).cpu())
    ax[0].set_title("Original")

    ax[1].imshow(recon.squeeze().permute(1,2,0).detach().cpu())
    ax[1].set_title("Autoencoder Reconstruction")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.savefig(f"AE_IMG{image_idx}_{layer_name}.png", dpi=300)
    plt.close()

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    model = load_model()

    # ===== L1 ANALYSIS =====
    dataset = CocoImageDataset("data/coco/images/val2017")
    analyzer = L1Analyzer(model, dataset)
    print ("L1 Analysis Beginning")
    for layer_name in get_target_layers().keys():
        for image_idx in range(NUM_IMAGES) : 
            sample_img, path = dataset[image_idx]
            print("Using image:", path)
            print(f"Starting top 5 nearest L1 distance search in {layer_name} ")
            similar = analyzer.analyze(sample_img, layer_name, top_k=5)
            save_l1_visualization(model, dataset, sample_img, path, image_idx, similar, layer_name)
            print(f"Completed {layer_name} ---Layer")
            print("Most similar images:", similar)
    print ("L1 Analysis Ended")

    # ===== AUTOENCODER TRAIN =====
    ae_trainer = AutoencoderTrainer(model, dataset)

    print ("AE Analysis Beginning")

    for layer_name in get_target_layers().keys():
        print (f"Begin Training for Layer {layer_name}")
        decoder = ae_trainer.train_layer(layer_name)
        print (f"Ended Training for Layer {layer_name}")

        for image_idx in range(NUM_IMAGES) : 
            sample_img, path = dataset[image_idx]
            collector = ActivationCollector(model, {layer_name: get_target_layers()[layer_name]})
            collector.register()

            act = collector.forward(sample_img.unsqueeze(0).to(DEVICE))[layer_name]
            act = act.to(DEVICE)

            recon = decoder(act)

            save_ae_visualization(sample_img.unsqueeze(0), recon, layer_name, image_idx)

            collector.remove()
            del collector
            torch.cuda.empty_cache()


    # ===== RECONSTRUCTION =====
    collector = ActivationCollector(model, get_target_layers())
    collector.register()

    act = collector.forward(sample_img.unsqueeze(0).to(DEVICE))["mid_c3"]
    recon = decoder(act)

    compare(sample_img.unsqueeze(0), recon, "Autoencoder Reconstruction")


if __name__ == "__main__":
    main()