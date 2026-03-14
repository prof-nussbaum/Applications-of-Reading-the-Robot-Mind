"""
RTRM Demonstration on Simple MLP Binary Classifier
==================================================

This script provides a pedagogical introduction to RTRM (Reading The Robot
Mind) using a simple multi-layer perceptron trained on 2D binary classification.
The low-dimensional setting allows direct visualization of how information
transforms and degrades through network layers.

RTRM Methodology Context:
    Subject Matter Experts (SMEs) need to understand how neural networks
    process their domain data. This script demonstrates three complementary
    methods for reconstructing inputs from hidden layer states, each revealing
    different aspects of information flow.

    The 2D input space allows us to directly visualize reconstructions,
    making this an ideal starting point before scaling to complex architectures
    like transformers (see rtrm_autoencoder.py) or vision models (see
    rtrm_vit.py and rtrm_yolo.py).

Three RTRM Methods Demonstrated:
    Method 1 - Cosine Equivalence Classes:
        Identifies which training samples have similar representations at
        each layer. As depth increases, more samples collapse into equivalence
        classes, revealing information loss through dimensionality reduction.

    Method 2 - Patch-Based Structural Reconstruction:
        Analytically propagates linear filter contributions back to input
        space without any training. Shows the geometric transformation each
        layer applies to the input manifold.

    Method 3 - Learned Inverse (Autoencoder):
        Trains small networks to invert each layer's transformation.
        Provides the most flexible reconstruction but requires training data.

Pedagogical Visualization Sequence:
    1. Original data with strategic probe points (establish SME baseline)
    2. Patch method progression (watch familiar patterns degrade)
    3. Autoencoder method progression (compare reconstruction approaches)
    4. Cosine equivalence clusters (which samples group together)
    5. Quantitative error metrics (numerical validation)
    6. Side-by-side method comparison at each layer

Output Artifacts:
    - 01_original_data_baseline.png
    - 02_patch_progression_all_layers.png
    - 03_autoencoder_progression_all_layers.png
    - 04_cosine_equivalence_visualization.png
    - 05_error_quantification.png
    - 06_method_comparison_layer_*.png

Dependencies:
    - torch
    - numpy
    - matplotlib

Reference:
    Nussbaum, P. (2023). Reading the Robot Mind. CLEF 2023.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def make_two_circles(n=2000, noise=0.1):
    """
    Generate two concentric circles dataset for binary classification.

    Args:
        n: Total number of samples.
        noise: Standard deviation of Gaussian noise added to radii.

    Returns:
        Tuple of (X, y) where X is [n, 2] and y is [n] with values 0 or 1.
    """
    n2 = n // 2

    theta0 = np.random.rand(n2) * 2 * np.pi
    r0 = 0.5 + noise * np.random.randn(n2)
    x0 = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)

    theta1 = np.random.rand(n2) * 2 * np.pi
    r1 = 1.5 + noise * np.random.randn(n2)
    x1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)

    X = np.vstack([x0, x1])
    y = np.array([0] * n2 + [1] * n2)
    return X, y


class MLP(nn.Module):
    """Simple MLP: 2 -> 64 -> 32 -> 16 -> 8 -> 1 with tanh activations."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 1)
        ])
        self.acts = [torch.tanh] * 5

    def forward(self, x, capture=False):
        activations = []
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
            activations.append(x)
        if capture:
            return x, activations
        return x


class InverseNet(nn.Module):
    """Network that learns to invert layer activations back to input space."""

    def __init__(self, dim):
        super().__init__()
        if dim <= 2:
            self.net = nn.Sequential(
                nn.Linear(dim, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 2)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 2)
            )

    def forward(self, x):
        return self.net(x)


def normalize_vectors(v):
    """Normalize vectors to unit length."""
    return v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)


def build_patches_no_bias(model):
    """Build cumulative filter patches without bias terms."""
    patches = []
    W = model.layers[0].weight.data
    patches.append([torch.tensor([W[j, 0], W[j, 1]]) for j in range(W.shape[0])])

    for i in range(1, len(model.layers)):
        Wi = model.layers[i].weight.data
        prev = patches[i - 1]
        layer_patches = []
        for k in range(Wi.shape[0]):
            p = torch.zeros(2)
            for j in range(Wi.shape[1]):
                p += Wi[k, j] * prev[j]
            layer_patches.append(p)
        patches.append(layer_patches)
    return patches


def reconstruct_with_patches(layer_idx, sample_idx, activations, patches):
    """Reconstruct input using patch method."""
    recon = torch.zeros(2)
    for n, patch in enumerate(patches[layer_idx]):
        recon += activations[layer_idx][sample_idx, n] * patch
    return recon[0].item(), recon[1].item()


def cosine_equivalence_counts(layer_idx, probe_idx, thresholds, activations):
    """Count samples with cosine similarity above each threshold."""
    layer_acts = activations[layer_idx]
    probe = layer_acts[probe_idx:probe_idx + 1]
    sims = layer_acts @ probe.T
    return [int((sims[:, 0] >= t).sum()) for t in thresholds]


def cosine_equivalence_membership(layer_idx, probe_idx, threshold, activations):
    """Get boolean mask of samples similar to probe."""
    layer_acts = activations[layer_idx]
    probe = layer_acts[probe_idx:probe_idx + 1]
    sims = layer_acts @ probe.T
    return (sims[:, 0] >= threshold).cpu().numpy()


def main():
    """Execute the complete RTRM MLP demonstration."""
    X, y = make_two_circles()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(2 * y - 1, dtype=torch.float32).unsqueeze(1)

    # Train classifier
    print("=" * 80)
    print("TRAINING CLASSIFIER")
    print("=" * 80)

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(5000):
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        if epoch % 1000 == 0:
            with torch.no_grad():
                acc = ((model(X_t) > 0) == (y_t > 0)).float().mean()
            print(f"Epoch {epoch:4d} - Loss {loss.item():.6f} - Acc {acc.item()*100:.2f}%")
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, activations = model(X_t, capture=True)
    activations = [normalize_vectors(a) for a in activations]

    # Strategic probe selection
    print("\n" + "=" * 80)
    print("STRATEGIC PROBE SELECTION")
    print("=" * 80)

    with torch.no_grad():
        pred_vals = model(X_t).numpy().flatten()

    boundary_idx = int(np.argmin(np.abs(pred_vals)))
    class0_indices = np.where(y == 0)[0]
    class1_indices = np.where(y == 1)[0]

    probe_indices = {
        "Boundary": boundary_idx,
        "Class0_Center": class0_indices[np.argmin(np.linalg.norm(X[class0_indices], axis=1))],
        "Class1_Center": class1_indices[np.argmin(np.linalg.norm(X[class1_indices], axis=1))],
        "Class0_Edge": class0_indices[np.argmax(np.linalg.norm(X[class0_indices], axis=1))],
        "Class1_Edge": class1_indices[np.argmax(np.linalg.norm(X[class1_indices], axis=1))],
    }

    colors = {
        "Boundary": "purple", "Class0_Center": "blue", "Class1_Center": "red",
        "Class0_Edge": "cyan", "Class1_Edge": "orange"
    }

    for name, idx in probe_indices.items():
        print(f"  {name:20s}: idx={idx:4d}, pos=({X[idx,0]:6.3f}, {X[idx,1]:6.3f}), class={y[idx]}")

    # Method 1: Cosine equivalence
    print("\n" + "=" * 80)
    print("METHOD 1: COSINE EQUIVALENCE CLASSES")
    print("=" * 80)

    thresholds = [0.99, 0.95, 0.90, 0.80]
    cosine_results = {}
    for name, idx in probe_indices.items():
        cosine_results[name] = []
        for layer_idx in range(len(activations)):
            counts = cosine_equivalence_counts(layer_idx, idx, thresholds, activations)
            cosine_results[name].append(counts)

    # Method 2: Patch-based reconstruction
    print("\n" + "=" * 80)
    print("METHOD 2: PATCH-BASED RECONSTRUCTION")
    print("=" * 80)

    patches = build_patches_no_bias(model)
    patch_reconstructions = []
    for layer_idx in range(len(activations)):
        layer_recons = []
        for i in range(len(X)):
            rec_x, rec_y = reconstruct_with_patches(layer_idx, i, activations, patches)
            layer_recons.append([rec_x, rec_y])
        patch_reconstructions.append(np.array(layer_recons))
        print(f"  Layer {layer_idx}: Reconstructed {len(X)} points")

    # Method 3: Learned inverse
    print("\n" + "=" * 80)
    print("METHOD 3: LEARNED INVERSE (AUTOENCODER)")
    print("=" * 80)

    inverse_nets = []
    for layer_idx, A in enumerate(activations):
        inv = InverseNet(A.shape[1])
        lr = 5e-4 if A.shape[1] <= 8 else 1e-3
        epochs = 5000 if A.shape[1] <= 8 else 2000
        opt = optim.Adam(inv.parameters(), lr=lr)

        for epoch in range(epochs):
            opt.zero_grad()
            loss = ((inv(A) - X_t) ** 2).mean()
            loss.backward()
            opt.step()
        print(f"  Layer {layer_idx} (dim={A.shape[1]}): Final loss = {loss.item():.6f}")
        inverse_nets.append(inv)

    autoencoder_reconstructions = []
    with torch.no_grad():
        for layer_idx in range(len(activations)):
            ae_recons = inverse_nets[layer_idx](activations[layer_idx]).numpy()
            autoencoder_reconstructions.append(ae_recons)

    # Compute errors
    patch_errors = {name: [] for name in probe_indices}
    ae_errors = {name: [] for name in probe_indices}
    for name, idx in probe_indices.items():
        for layer_idx in range(len(patch_reconstructions)):
            orig = X[idx]
            patch_rec = patch_reconstructions[layer_idx][idx]
            ae_rec = autoencoder_reconstructions[layer_idx][idx]
            patch_errors[name].append(np.linalg.norm(patch_rec - orig))
            ae_errors[name].append(np.linalg.norm(ae_rec - orig))

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Visualization 1: Original data baseline
    print("\n1. Creating baseline visualization...")
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10, alpha=0.4)
    for name, idx in probe_indices.items():
        plt.scatter(X[idx, 0], X[idx, 1], c=colors[name], s=200, marker='*',
                    edgecolors='black', linewidths=2, label=name, zorder=5)
    plt.title("Step 1: Original Data (SME Baseline)", fontsize=14, fontweight='bold')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('01_original_data_baseline.png', dpi=150)
    print("   Saved: 01_original_data_baseline.png")
    plt.close()

    # Visualization 2: Patch method progression
    print("\n2. Creating patch method progression...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for layer_idx in range(min(5, len(patch_reconstructions))):
        ax = axes[layer_idx]
        recons = patch_reconstructions[layer_idx]
        ax.scatter(recons[:, 0], recons[:, 1], c=y, cmap="coolwarm", s=10, alpha=0.6)
        ax.scatter(X[:, 0], X[:, 1], c='gray', s=2, alpha=0.1)
        for name, idx in probe_indices.items():
            ax.scatter(recons[idx, 0], recons[idx, 1], c=colors[name], s=150,
                       marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        ax.set_title(f"Layer {layer_idx} (dim={activations[layer_idx].shape[1]})")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    axes[5].axis('off')
    plt.suptitle("Patch-Based Reconstruction Progression", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('02_patch_progression_all_layers.png', dpi=150)
    print("   Saved: 02_patch_progression_all_layers.png")
    plt.close()

    # Visualization 3: Autoencoder method progression
    print("\n3. Creating autoencoder method progression...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for layer_idx in range(min(5, len(autoencoder_reconstructions))):
        ax = axes[layer_idx]
        recons = autoencoder_reconstructions[layer_idx]
        ax.scatter(recons[:, 0], recons[:, 1], c=y, cmap="coolwarm", s=10, alpha=0.6)
        ax.scatter(X[:, 0], X[:, 1], c='gray', s=2, alpha=0.1)
        for name, idx in probe_indices.items():
            ax.scatter(recons[idx, 0], recons[idx, 1], c=colors[name], s=150,
                       marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        ax.set_title(f"Layer {layer_idx} (dim={activations[layer_idx].shape[1]})")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    axes[5].axis('off')
    plt.suptitle("Autoencoder Reconstruction Progression", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('03_autoencoder_progression_all_layers.png', dpi=150)
    print("   Saved: 03_autoencoder_progression_all_layers.png")
    plt.close()

    # Visualization 4: Cosine equivalence
    print("\n4. Creating cosine equivalence visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    threshold = 0.90
    probe_name = "Boundary"
    probe_idx = probe_indices[probe_name]
    for layer_idx in range(min(5, len(activations))):
        ax = axes[layer_idx]
        membership = cosine_equivalence_membership(layer_idx, probe_idx, threshold, activations)
        ax.scatter(X[~membership, 0], X[~membership, 1], c='lightgray', s=10, alpha=0.3)
        ax.scatter(X[membership, 0], X[membership, 1], c='blue', s=20, alpha=0.6)
        ax.scatter(X[probe_idx, 0], X[probe_idx, 1], c='red', s=200, marker='*',
                   edgecolors='black', linewidths=2, zorder=5)
        count = membership.sum()
        ax.set_title(f"Layer {layer_idx}: {count} similar (threshold={threshold})")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    axes[5].axis('off')
    plt.suptitle(f"Cosine Equivalence Classes for {probe_name} Probe", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('04_cosine_equivalence_visualization.png', dpi=150)
    print("   Saved: 04_cosine_equivalence_visualization.png")
    plt.close()

    # Visualization 5: Error quantification
    print("\n5. Creating error quantification plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    layers = list(range(len(activations)))
    for name in probe_indices:
        ax1.plot(layers, patch_errors[name], marker='o', label=name, linewidth=2)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Reconstruction Error")
    ax1.set_title("Patch Method Errors", fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for name in probe_indices:
        ax2.plot(layers, ae_errors[name], marker='s', label=name, linewidth=2)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Reconstruction Error")
    ax2.set_title("Autoencoder Method Errors", fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Reconstruction Error by Layer", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('05_error_quantification.png', dpi=150)
    print("   Saved: 05_error_quantification.png")
    plt.close()

    # Visualization 6: Method comparison per layer
    print("\n6. Creating method comparison plots...")
    for layer_idx in range(len(activations)):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original
        ax = axes[0]
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10, alpha=0.4)
        for name, idx in probe_indices.items():
            ax.scatter(X[idx, 0], X[idx, 1], c=colors[name], s=150, marker='*',
                       edgecolors='black', linewidths=1.5, zorder=5)
        ax.set_title("Original Data")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Patch method
        ax = axes[1]
        recons = patch_reconstructions[layer_idx]
        ax.scatter(recons[:, 0], recons[:, 1], c=y, cmap="coolwarm", s=10, alpha=0.6)
        for name, idx in probe_indices.items():
            ax.scatter(recons[idx, 0], recons[idx, 1], c=colors[name], s=150, marker='*',
                       edgecolors='black', linewidths=1.5, zorder=5)
        ax.set_title("Patch Reconstruction")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Autoencoder method
        ax = axes[2]
        recons = autoencoder_reconstructions[layer_idx]
        ax.scatter(recons[:, 0], recons[:, 1], c=y, cmap="coolwarm", s=10, alpha=0.6)
        for name, idx in probe_indices.items():
            ax.scatter(recons[idx, 0], recons[idx, 1], c=colors[name], s=150, marker='*',
                       edgecolors='black', linewidths=1.5, zorder=5)
        ax.set_title("Autoencoder Reconstruction")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.suptitle(f"Layer {layer_idx} Comparison (dim={activations[layer_idx].shape[1]})",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'06_method_comparison_layer_{layer_idx}.png', dpi=150)
        plt.close()
    print("   Saved: 06_method_comparison_layer_*.png")

    print("\n" + "=" * 80)
    print("RTRM MLP DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print("  - 01_original_data_baseline.png")
    print("  - 02_patch_progression_all_layers.png")
    print("  - 03_autoencoder_progression_all_layers.png")
    print("  - 04_cosine_equivalence_visualization.png")
    print("  - 05_error_quantification.png")
    print("  - 06_method_comparison_layer_*.png")


if __name__ == "__main__":
    main()
