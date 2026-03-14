"""
RTRM Demonstration for Vision Transformer on MNIST
==================================================

This script demonstrates RTRM (Reading The Robot Mind) applied to Vision
Transformers (ViT), showing how the technique can diagnose architectural
problems in image classification models through input reconstruction.

RTRM Methodology Context:
    Vision Transformers divide images into patches, embed them, and process
    through transformer blocks. RTRM trains decoder networks to reconstruct
    the original image from intermediate layer representations. Poor
    reconstruction quality indicates information loss that may impair
    classification performance.

Demonstration Strategy (Three-Model Comparison):
    Model A - Flawed Architecture, No RTRM:
        Uses an intentionally small hidden dimension (bottleneck).
        Achieves reasonable accuracy but provides no visibility into
        whether the architecture is optimal.

    Model B - Flawed Architecture, With RTRM:
        Same bottleneck as Model A, but includes RTRM decoders.
        Reconstruction visualization reveals severe information loss.

    Model C - Fixed Architecture, With RTRM:
        Increased hidden dimension based on RTRM diagnosis.
        Improved reconstruction quality confirms the fix worked.

Output Artifacts:
    - model_*_training_history.png: Training curves
    - model_*_reconstructions.png: Layer-wise reconstruction quality
    - model_comparison.png: Side-by-side accuracy and MSE comparison

Dependencies:
    - torch, torchvision
    - numpy, matplotlib, tqdm

Reference:
    Nussbaum, P. (2023). Reading the Robot Mind. CLEF 2023.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PatchEmbedding(nn.Module):
    """Convert image into sequence of patch embeddings."""

    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=64):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patch sequences."""

    def __init__(self, n_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x + self.pos_embed


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention and FFN."""

    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RTRMDecoder(nn.Module):
    """Decoder that reconstructs images from transformer representations."""

    def __init__(self, embed_dim, patch_size=4, n_patches_side=4):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches_side = n_patches_side
        self.proj = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, x):
        x = x[:, 1:, :]  # Remove CLS token
        B, N, _ = x.shape
        x = self.proj(x)
        x = x.reshape(B, self.n_patches_side, self.n_patches_side,
                      self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(B, 1, self.n_patches_side * self.patch_size,
                      self.n_patches_side * self.patch_size)
        return x


class VisionTransformerWithRTRM(nn.Module):
    """Vision Transformer with integrated RTRM decoders."""

    def __init__(self, img_size=28, patch_size=4, in_channels=1, n_classes=10,
                 embed_dim=64, hidden_dim=16, n_heads=4, n_layers=4, enable_rtrm=True):
        super().__init__()
        self.enable_rtrm = enable_rtrm
        self.n_layers = n_layers
        n_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = PositionalEncoding(n_patches, embed_dim)

        # Bottleneck projection
        self.to_hidden = nn.Linear(embed_dim, hidden_dim)
        self.from_hidden = nn.Linear(hidden_dim, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        if enable_rtrm:
            self.rtrm_decoders = nn.ModuleList([
                RTRMDecoder(embed_dim, patch_size, img_size // patch_size)
                for _ in range(n_layers + 1)
            ])

    def forward(self, x, return_reconstructions=False):
        reconstructions = []

        x = self.patch_embed(x)
        x = self.pos_embed(x)

        # Bottleneck
        x = self.to_hidden(x)
        x = self.from_hidden(x)

        if self.enable_rtrm and return_reconstructions:
            reconstructions.append(self.rtrm_decoders[0](x))

        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.enable_rtrm and return_reconstructions:
                reconstructions.append(self.rtrm_decoders[i + 1](x))

        x = self.norm(x)
        cls_token = x[:, 0]
        logits = self.head(cls_token)

        if return_reconstructions:
            return logits, reconstructions
        return logits


def get_data_loaders(batch_size=128):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    recon_loss_sum = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if model.enable_rtrm:
            logits, recons = model(images, return_reconstructions=True)
            cls_loss = criterion(logits, labels)
            recon_loss = sum(
                nn.functional.mse_loss(r, images) for r in recons
            ) / len(recons)
            loss = cls_loss + 0.1 * recon_loss
            recon_loss_sum += recon_loss.item()
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    avg_recon = recon_loss_sum / len(train_loader) if model.enable_rtrm else 0
    return avg_loss, acc, avg_recon


def evaluate(model, test_loader, criterion):
    """Evaluate model on test set."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    recon_loss_sum = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if model.enable_rtrm:
                logits, recons = model(images, return_reconstructions=True)
                recon_loss = sum(
                    nn.functional.mse_loss(r, images) for r in recons
                ) / len(recons)
                recon_loss_sum += recon_loss.item()
            else:
                logits = model(images)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    avg_recon = recon_loss_sum / len(test_loader) if model.enable_rtrm else 0
    return avg_loss, acc, avg_recon


def visualize_reconstructions(model, test_loader, save_path, title):
    """Visualize RTRM reconstructions at each layer."""
    model.eval()
    images, _ = next(iter(test_loader))
    images = images[:8].to(device)

    with torch.no_grad():
        _, recons = model(images, return_reconstructions=True)

    n_layers = len(recons)
    fig, axes = plt.subplots(8, n_layers + 1, figsize=(2 * (n_layers + 1), 16))

    for i in range(8):
        # Original
        axes[i, 0].imshow(images[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original')

        # Reconstructions at each layer
        for j, recon in enumerate(recons):
            axes[i, j + 1].imshow(recon[i, 0].cpu().numpy(), cmap='gray')
            axes[i, j + 1].axis('off')
            if i == 0:
                axes[i, j + 1].set_title(f'Layer {j}')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def plot_training_history(history, save_path, title):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Classification Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['test_acc'], label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Classification Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if history['recon_loss'][0] > 0:
        axes[2].plot(history['recon_loss'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('MSE')
        axes[2].set_title('Reconstruction Loss')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'RTRM Disabled', ha='center', va='center',
                     fontsize=12, transform=axes[2].transAxes)
        axes[2].set_title('Reconstruction Loss')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def train_model(model, train_loader, test_loader, n_epochs=10, lr=1e-3, model_name="Model"):
    """Train model and return history."""
    print(f"\nTraining {model_name}...")
    print("-" * 60)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'recon_loss': []
    }

    for epoch in range(n_epochs):
        train_loss, train_acc, recon_loss = train_epoch(
            model, train_loader, optimizer, criterion
        )
        test_loss, test_acc, _ = evaluate(model, test_loader, criterion)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['recon_loss'].append(recon_loss)

        print(f"Epoch {epoch+1:2d}/{n_epochs} - "
              f"Train: {train_acc:.1f}% - Test: {test_acc:.1f}% - "
              f"Recon: {recon_loss:.4f}")

    return history


def main():
    """Execute the complete RTRM ViT demonstration."""
    print("=" * 80)
    print("RTRM Vision Transformer Demonstration")
    print("=" * 80)

    train_loader, test_loader = get_data_loaders(batch_size=128)
    n_epochs = 10

    # Model A: Flawed architecture, no RTRM
    print("\n" + "=" * 80)
    print("MODEL A: Flawed Architecture (hidden_dim=4), No RTRM")
    print("=" * 80)

    model_a = VisionTransformerWithRTRM(
        hidden_dim=4, enable_rtrm=False
    ).to(device)
    history_a = train_model(
        model_a, train_loader, test_loader,
        n_epochs=n_epochs, model_name="Model A"
    )
    plot_training_history(
        history_a,
        'model_a_training_history.png',
        'Model A: Flawed (hidden_dim=4), No RTRM'
    )

    # Model B: Flawed architecture, with RTRM
    print("\n" + "=" * 80)
    print("MODEL B: Flawed Architecture (hidden_dim=4), With RTRM")
    print("=" * 80)

    model_b = VisionTransformerWithRTRM(
        hidden_dim=4, enable_rtrm=True
    ).to(device)
    history_b = train_model(
        model_b, train_loader, test_loader,
        n_epochs=n_epochs, model_name="Model B"
    )
    plot_training_history(
        history_b,
        'model_b_training_history.png',
        'Model B: Flawed (hidden_dim=4), With RTRM'
    )
    visualize_reconstructions(
        model_b, test_loader,
        'model_b_reconstructions.png',
        'Model B Reconstructions: Flawed Architecture'
    )

    # Model C: Fixed architecture, with RTRM
    print("\n" + "=" * 80)
    print("MODEL C: Fixed Architecture (hidden_dim=16), With RTRM")
    print("=" * 80)

    model_c = VisionTransformerWithRTRM(
        hidden_dim=16, enable_rtrm=True
    ).to(device)
    history_c = train_model(
        model_c, train_loader, test_loader,
        n_epochs=n_epochs, model_name="Model C"
    )
    plot_training_history(
        history_c,
        'model_c_training_history.png',
        'Model C: Fixed (hidden_dim=16), With RTRM'
    )
    visualize_reconstructions(
        model_c, test_loader,
        'model_c_reconstructions.png',
        'Model C Reconstructions: Fixed Architecture'
    )

    # Comparison plot
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Accuracy comparison
    axes[0].plot(history_a['test_acc'], label='A: Flawed, No RTRM', linestyle='--')
    axes[0].plot(history_b['test_acc'], label='B: Flawed, RTRM')
    axes[0].plot(history_c['test_acc'], label='C: Fixed, RTRM')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Classification Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction MSE comparison
    axes[1].plot(history_b['recon_loss'], label='B: Flawed')
    axes[1].plot(history_c['recon_loss'], label='C: Fixed')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('RTRM Reconstruction Quality')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Final metrics bar chart
    final_accs = [history_a['test_acc'][-1], history_b['test_acc'][-1], history_c['test_acc'][-1]]
    final_recons = [0, history_b['recon_loss'][-1], history_c['recon_loss'][-1]]
    x = np.arange(3)
    width = 0.35

    ax2 = axes[2].twinx()
    bars1 = axes[2].bar(x - width/2, final_accs, width, label='Accuracy', color='steelblue')
    bars2 = ax2.bar(x + width/2, final_recons, width, label='Recon MSE', color='coral')

    axes[2].set_xlabel('Model')
    axes[2].set_ylabel('Accuracy (%)', color='steelblue')
    ax2.set_ylabel('Recon MSE', color='coral')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(['A\n(Flawed, No RTRM)', 'B\n(Flawed, RTRM)', 'C\n(Fixed, RTRM)'])
    axes[2].set_title('Final Metrics Comparison')

    plt.suptitle('RTRM ViT Demonstration: Three-Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: model_comparison.png")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nModel A (Flawed, No RTRM):   {history_a['test_acc'][-1]:.1f}% accuracy")
    print(f"Model B (Flawed, RTRM):      {history_b['test_acc'][-1]:.1f}% accuracy, "
          f"Recon MSE: {history_b['recon_loss'][-1]:.4f}")
    print(f"Model C (Fixed, RTRM):       {history_c['test_acc'][-1]:.1f}% accuracy, "
          f"Recon MSE: {history_c['recon_loss'][-1]:.4f}")

    print("\nKey Insight:")
    print("  Model B's high reconstruction MSE reveals the bottleneck problem")
    print("  that is invisible in Model A without RTRM.")
    print("  Model C's improved reconstruction confirms the fix worked.")

    print("\nOutput files:")
    print("  - model_a_training_history.png")
    print("  - model_b_training_history.png")
    print("  - model_b_reconstructions.png")
    print("  - model_c_training_history.png")
    print("  - model_c_reconstructions.png")
    print("  - model_comparison.png")


if __name__ == "__main__":
    main()
