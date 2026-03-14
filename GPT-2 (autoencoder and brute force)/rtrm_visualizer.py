"""
RTRM Visualization and Analysis Utilities
=========================================

This module provides visualization tools for the RTRM (Reading The Robot
Mind) autoencoder system. It transforms quantitative reconstruction metrics
into visual formats that Subject Matter Experts (SMEs) can interpret without
requiring deep technical knowledge of transformer architectures.

RTRM Methodology Context:
    RTRM reconstructs inputs from intermediate network states to reveal
    information flow. This module visualizes those reconstructions to help
    SMEs identify:
    - Which layers preserve input information faithfully
    - Where information bottlenecks cause reconstruction degradation
    - How different probe inputs degrade at different rates
    - Token-level reconstruction accuracy patterns

    By presenting results visually, SMEs can assess model behavior using
    their domain expertise rather than requiring ML knowledge.

Visualization Components:
    RTRMVisualizer: Main visualization class providing:
        - plot_reconstruction_progression: Layer-by-layer text reconstruction
        - plot_token_level_accuracy: Per-token match visualization
        - compare_multiple_probes: Multi-probe MSE curves and heatmaps
        - plot_training_history: Decoder training loss curves
        - plot_cosine_equivalence: Token similarity tables per layer
        - generate_comprehensive_report: Full analysis report generation

    analyze_layer_information_bottlenecks: Standalone function that detects
        sharp MSE increases between adjacent layers (>2x jump) indicating
        significant information discard points.

Visual Design Philosophy:
    Reconstructions are color-coded by quality:
        - Green: Excellent reconstruction (MSE < 0.01)
        - Yellow: Good reconstruction (MSE < 0.05)
        - Orange: Degraded reconstruction (MSE < 0.15)
        - Red: Poor reconstruction (MSE >= 0.15)

    This allows SMEs to quickly scan visualizations and identify
    problematic layers without interpreting raw metrics.

Output Artifacts:
    - reconstruction_progression.png: Layer-by-layer reconstruction display
    - token_accuracy.png: Per-token match visualization
    - probe_comparison.png: Multi-probe MSE curves and retention heatmap
    - training_history.png: Decoder training loss curves
    - cosine_equivalence.png: Token similarity tables per layer

Dependencies:
    - rtrm_autoencoder.py (RTRMAutoencoder class)
    - torch
    - numpy, matplotlib, seaborn

Reference:
    Nussbaum, P. (2023). Reading the Robot Mind. CLEF 2023.
"""

import os
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


class RTRMVisualizer:
    """
    Advanced visualization tools for RTRM analysis.

    Provides comprehensive visual analysis of layer-wise reconstruction
    quality, information retention, token-level accuracy, and comparative
    analysis between probe points.

    Attributes:
        rtrm: RTRMAutoencoder instance with trained decoders.
        extractor: Reference to the GPT2ActivationExtractor.
    """

    def __init__(self, rtrm_system):
        """
        Initialize the visualizer with an RTRM system.

        Args:
            rtrm_system: RTRMAutoencoder instance with trained decoders.
        """
        self.rtrm = rtrm_system
        self.extractor = rtrm_system.extractor

    def plot_reconstruction_progression(
        self,
        text: str,
        save_path: str = 'reconstruction_progression.png',
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Create visualization showing reconstruction quality at each layer.

        Displays the original text and its reconstruction from each analyzed
        layer, with color-coding to indicate quality. Also includes an MSE
        plot showing information loss across layers.

        Args:
            text: Probe point text to analyze.
            save_path: Path to save the visualization.
            figsize: Figure dimensions (width, height) in inches.
        """
        layers = sorted(self.rtrm.layers_to_analyze)
        n_layers = len(layers)

        reconstructions = []
        mses = []

        for layer_idx in layers:
            recon_text, mse = self.rtrm.reconstruct_text(text, layer_idx)
            reconstructions.append(recon_text)
            mses.append(mse)

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_layers + 2, 2, figure=fig, hspace=0.4, wspace=0.3)

        # Original text at top
        ax_original = fig.add_subplot(gs[0, :])
        ax_original.text(
            0.5, 0.5,
            f"Original: {text}",
            ha='center',
            va='center',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
        )
        ax_original.set_xlim(0, 1)
        ax_original.set_ylim(0, 1)
        ax_original.axis('off')
        ax_original.set_title('Input Text', fontsize=12, fontweight='bold')

        # Reconstructions for each layer
        for i, (layer_idx, recon, mse) in enumerate(
            zip(layers, reconstructions, mses)
        ):
            ax = fig.add_subplot(gs[i + 1, 0])

            # Color code by reconstruction quality
            if mse < 0.01:
                color = 'lightgreen'
            elif mse < 0.05:
                color = 'lightyellow'
            elif mse < 0.15:
                color = 'orange'
            else:
                color = 'lightcoral'

            ax.text(
                0.5, 0.5,
                f"Layer {layer_idx}: {recon}",
                ha='center',
                va='center',
                fontsize=9,
                wrap=True,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.5)
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_ylabel(
                f"MSE: {mse:.4f}",
                fontsize=9,
                rotation=0,
                labelpad=40,
                va='center'
            )

        # MSE plot on right side
        ax_mse = fig.add_subplot(gs[1:, 1])
        ax_mse.plot(
            layers, mses,
            marker='o',
            linewidth=2,
            markersize=8,
            color='darkblue',
            label='Reconstruction Error'
        )
        ax_mse.fill_between(layers, mses, alpha=0.3, color='skyblue')
        ax_mse.set_xlabel('Layer Index', fontsize=11)
        ax_mse.set_ylabel('MSE (log scale)', fontsize=11)
        ax_mse.set_yscale('log')
        ax_mse.set_title(
            'Information Loss Across Layers',
            fontsize=12,
            fontweight='bold'
        )
        ax_mse.grid(True, alpha=0.3)
        ax_mse.set_xticks(layers)

        # Quality threshold reference lines
        ax_mse.axhline(
            y=0.01, color='g', linestyle='--', alpha=0.5, label='Excellent'
        )
        ax_mse.axhline(
            y=0.05, color='y', linestyle='--', alpha=0.5, label='Good'
        )
        ax_mse.axhline(
            y=0.15, color='r', linestyle='--', alpha=0.5, label='Poor'
        )
        ax_mse.legend(loc='upper left', fontsize=9)

        plt.suptitle(
            'RTRM Layer-wise Reconstruction Analysis',
            fontsize=14,
            fontweight='bold'
        )

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.close()

    def plot_token_level_accuracy(
        self,
        text: str,
        layer_idx: int,
        save_path: str = 'token_accuracy.png',
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Visualize which tokens are reconstructed correctly at a specific layer.

        Creates a bar chart showing match/mismatch for each token position,
        plus a text comparison of original vs reconstructed.

        Args:
            text: Input text to analyze.
            layer_idx: Which layer to examine.
            save_path: Path to save the plot.
            figsize: Figure dimensions (width, height) in inches.
        """
        # Get original tokens
        inputs = self.extractor.tokenizer(text, return_tensors='pt')
        original_tokens = inputs['input_ids'][0].tolist()
        original_words = [
            self.extractor.tokenizer.decode([t])
            for t in original_tokens
        ]

        # Get reconstruction
        reconstructed_text, mse = self.rtrm.reconstruct_text(text, layer_idx)
        recon_inputs = self.extractor.tokenizer(
            reconstructed_text,
            return_tensors='pt'
        )
        recon_tokens = recon_inputs['input_ids'][0].tolist()

        # Pad to same length for comparison
        max_len = max(len(original_tokens), len(recon_tokens))
        original_tokens += [0] * (max_len - len(original_tokens))
        recon_tokens += [0] * (max_len - len(recon_tokens))
        original_words += ['[PAD]'] * (max_len - len(original_words))

        # Compare token by token
        matches = [
            1 if o == r else 0
            for o, r in zip(original_tokens, recon_tokens)
        ]
        accuracy = sum(matches) / len(matches) * 100

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Token comparison bar chart
        x = np.arange(len(matches))
        colors = ['green' if m else 'red' for m in matches]
        ax1.bar(x, matches, color=colors, alpha=0.6)
        ax1.set_ylabel('Match', fontsize=11)
        ax1.set_title(
            f'Layer {layer_idx} Token-Level Reconstruction '
            f'(Accuracy: {accuracy:.1f}%)',
            fontsize=12,
            fontweight='bold'
        )
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(original_words, rotation=45, ha='right', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)

        # Legend
        green_patch = mpatches.Patch(color='green', alpha=0.6, label='Correct')
        red_patch = mpatches.Patch(color='red', alpha=0.6, label='Incorrect')
        ax1.legend(handles=[green_patch, red_patch], loc='upper right')

        # Text comparison panel
        ax2.text(
            0.5, 0.7,
            "Original Text:",
            ha='center',
            fontsize=11,
            fontweight='bold',
            transform=ax2.transAxes
        )
        ax2.text(
            0.5, 0.5,
            text,
            ha='center',
            fontsize=10,
            wrap=True,
            transform=ax2.transAxes
        )
        ax2.text(
            0.5, 0.3,
            "Reconstructed Text:",
            ha='center',
            fontsize=11,
            fontweight='bold',
            transform=ax2.transAxes
        )
        ax2.text(
            0.5, 0.1,
            reconstructed_text,
            ha='center',
            fontsize=10,
            wrap=True,
            transform=ax2.transAxes
        )
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Token accuracy visualization saved to {save_path}")
        plt.close()

    def compare_multiple_probes(
        self,
        probe_texts: List[str],
        save_path: str = 'probe_comparison.png',
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Compare reconstruction quality across multiple probe points.

        Creates two visualizations:
        1. Line plot of MSE vs layer for each probe
        2. Heatmap of information retention scores

        Args:
            probe_texts: List of texts to compare.
            save_path: Path to save the comparison plot.
            figsize: Figure dimensions (width, height) in inches.
        """
        layers = sorted(self.rtrm.layers_to_analyze)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # MSE comparison line plot
        for i, text in enumerate(probe_texts):
            mses = []
            for layer_idx in layers:
                _, mse = self.rtrm.reconstruct_text(text, layer_idx)
                mses.append(mse)

            # Truncate long labels
            if len(text) > 30:
                label = f"Probe {i + 1}: {text[:30]}..."
            else:
                label = f"Probe {i + 1}: {text}"
            ax1.plot(layers, mses, marker='o', linewidth=2, label=label, alpha=0.7)

        ax1.set_xlabel('Layer Index', fontsize=11)
        ax1.set_ylabel('Reconstruction MSE (log scale)', fontsize=11)
        ax1.set_yscale('log')
        ax1.set_title(
            'Reconstruction Quality Comparison',
            fontsize=12,
            fontweight='bold'
        )
        ax1.legend(fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(layers)

        # Information retention heatmap
        heatmap_data = []
        for text in probe_texts:
            retention_scores = []
            for layer_idx in layers:
                _, mse = self.rtrm.reconstruct_text(text, layer_idx)
                # Convert MSE to retention score (inverse relationship)
                retention = 1 / (1 + mse)
                retention_scores.append(retention)
            heatmap_data.append(retention_scores)

        heatmap_data = np.array(heatmap_data)

        im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        ax2.set_xlabel('Layer Index', fontsize=11)
        ax2.set_ylabel('Probe Point', fontsize=11)
        ax2.set_title(
            'Information Retention Heatmap',
            fontsize=12,
            fontweight='bold'
        )
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels(layers)
        ax2.set_yticks(range(len(probe_texts)))
        ax2.set_yticklabels([f"Probe {i + 1}" for i in range(len(probe_texts))])

        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Retention Score', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probe comparison saved to {save_path}")
        plt.close()

    def plot_training_history(
        self,
        save_path: str = 'training_history.png',
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Visualize the training history of all decoders.

        Creates a grid of subplots showing loss curves for each layer's
        decoder training.

        Args:
            save_path: Path to save the plot.
            figsize: Figure dimensions (width, height) in inches.
        """
        if not self.rtrm.training_history:
            print("No training history available.")
            return

        layers = sorted(self.rtrm.training_history.keys())
        n_layers = len(layers)

        # Create subplot grid
        n_cols = (n_layers + 1) // 2
        fig, axes = plt.subplots(2, n_cols, figsize=figsize)
        axes = axes.flatten()

        for idx, layer_idx in enumerate(layers):
            history = self.rtrm.training_history[layer_idx]
            losses = history['train_loss']
            epochs = range(1, len(losses) + 1)

            ax = axes[idx]
            ax.plot(epochs, losses, linewidth=1.5, color='darkblue', alpha=0.7)
            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel('Loss', fontsize=9)
            ax.set_title(f'Layer {layer_idx}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            # Annotate final loss
            final_loss = losses[-1]
            ax.annotate(
                f'Final: {final_loss:.6f}',
                xy=(len(losses), final_loss),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3)
            )

        # Hide unused subplots
        for idx in range(n_layers, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(
            'RTRM Decoder Training History',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()

    def plot_cosine_equivalence(
        self,
        probe_text: str,
        cosine_results: Dict[int, List[dict]],
        save_path: str = 'cosine_equivalence.png'
    ):
        """
        Visualize nearest training tokens for each probe token position.

        Creates a table for each layer showing the probe token and its
        top matching tokens from the training corpus.

        Args:
            probe_text: Original probe text.
            cosine_results: Output from RTRMAutoencoder.analyze_cosine_equivalence().
            save_path: Path to save the plot.
        """
        layers = sorted(cosine_results.keys())
        n_layers = len(layers)

        fig, axes = plt.subplots(n_layers, 1, figsize=(18, 4 * n_layers))

        # Handle single layer case
        if n_layers == 1:
            axes = [axes]

        for idx, layer_idx in enumerate(layers):
            ax = axes[idx]

            position_matches = cosine_results[layer_idx]

            # Build table data
            table_data = [[
                'Pos', 'Probe Token',
                'Top Match 1', 'Sim',
                'Top Match 2', 'Sim',
                'Top Match 3', 'Sim'
            ]]

            for pos, pos_data in enumerate(position_matches):
                probe_token = pos_data['probe_token']
                matches = pos_data['matches']

                row = [f"{pos}", f"[{probe_token}]"]
                for match in matches[:3]:
                    row.append(f"[{match['token']}]")
                    row.append(f"{match['similarity']:.2f}")

                # Pad if fewer than 3 matches
                while len(row) < 8:
                    row.extend(['', ''])

                table_data.append(row)

            # Create table
            table = ax.table(
                cellText=table_data,
                cellLoc='center',
                loc='center',
                colWidths=[0.05, 0.15, 0.15, 0.08, 0.15, 0.08, 0.15, 0.08]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.8)

            # Style header row
            for col in range(8):
                table[(0, col)].set_facecolor('#CCCCCC')
                table[(0, col)].set_text_props(weight='bold')

            # Style data rows
            for row in range(1, len(table_data)):
                # Highlight probe token column
                table[(row, 1)].set_facecolor('#90EE90')

                # Color high-similarity matches
                if len(table_data[row]) > 3 and table_data[row][3]:
                    try:
                        sim1 = float(table_data[row][3])
                        if sim1 >= 0.95:
                            table[(row, 2)].set_facecolor('#FFFF99')
                        elif sim1 >= 0.85:
                            table[(row, 2)].set_facecolor('#FFD699')
                    except ValueError:
                        pass

            ax.set_title(
                f'Layer {layer_idx} - Bag-of-Words Cosine Similarity (Per Token)',
                fontweight='bold',
                fontsize=12
            )
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cosine equivalence plot saved to {save_path}")
        plt.close()

    def generate_comprehensive_report(
        self,
        probe_texts: List[str],
        output_dir: str = './rtrm_analysis'
    ):
        """
        Generate a comprehensive analysis report with all visualizations.

        Creates a complete set of visualizations for all probe points:
        - Training history
        - Per-probe reconstruction progression
        - Per-probe token accuracy
        - Multi-probe comparison
        - Reconstruction quality summary

        Args:
            probe_texts: List of probe points to analyze.
            output_dir: Directory to save all output files.
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 80}")
        print("Generating Comprehensive RTRM Report")
        print(f"{'=' * 80}\n")

        # 1. Training history
        print("1. Plotting training history...")
        self.plot_training_history(
            save_path=os.path.join(output_dir, 'training_history.png')
        )

        # 2. Individual probe analysis
        for i, text in enumerate(probe_texts):
            print(f"\n2.{i + 1}. Analyzing probe point {i + 1}...")

            # Reconstruction progression
            self.plot_reconstruction_progression(
                text,
                save_path=os.path.join(
                    output_dir,
                    f'probe_{i + 1}_progression.png'
                )
            )

            # Token-level analysis for middle layer
            mid_layer_idx = len(self.rtrm.layers_to_analyze) // 2
            mid_layer = sorted(self.rtrm.layers_to_analyze)[mid_layer_idx]
            self.plot_token_level_accuracy(
                text,
                layer_idx=mid_layer,
                save_path=os.path.join(
                    output_dir,
                    f'probe_{i + 1}_token_accuracy.png'
                )
            )

            # Text analysis file
            self.rtrm.analyze_probe_point(
                text,
                output_file=os.path.join(
                    output_dir,
                    f'probe_{i + 1}_analysis.txt'
                )
            )

        # 3. Comparative analysis (if multiple probes)
        if len(probe_texts) > 1:
            print("\n3. Creating comparative analysis...")
            self.compare_multiple_probes(
                probe_texts,
                save_path=os.path.join(output_dir, 'probe_comparison.png')
            )

        # 4. Summary reconstruction quality
        print("\n4. Generating summary plot...")
        self.rtrm.plot_reconstruction_quality(
            save_path=os.path.join(
                output_dir,
                'reconstruction_quality_summary.png'
            )
        )

        print(f"\n{'=' * 80}")
        print("Report generated successfully!")
        print(f"All outputs saved to: {output_dir}")
        print(f"{'=' * 80}\n")


def analyze_layer_information_bottlenecks(
    rtrm_system,
    probe_texts: List[str]
) -> Dict[int, List[dict]]:
    """
    Identify which layers act as information bottlenecks.

    A bottleneck is detected when reconstruction MSE increases sharply
    (more than 2x) between adjacent layers. This indicates significant
    information discard that may cause downstream errors.

    Args:
        rtrm_system: RTRMAutoencoder instance with trained decoders.
        probe_texts: List of probe points to analyze.

    Returns:
        Dictionary mapping layer indices to lists of bottleneck events.
        Each event contains:
            - text: The probe text (truncated if long)
            - mse_before: MSE at previous layer
            - mse_after: MSE at this layer
            - increase_factor: Ratio of mse_after to mse_before
    """
    layers = sorted(rtrm_system.layers_to_analyze)
    bottlenecks = {layer: [] for layer in layers}

    print("\n" + "=" * 80)
    print("Information Bottleneck Analysis")
    print("=" * 80 + "\n")

    for text in probe_texts:
        mses = []
        for layer_idx in layers:
            _, mse = rtrm_system.reconstruct_text(text, layer_idx)
            mses.append(mse)

        # Detect sharp increases (> 2x jump)
        for i in range(1, len(mses)):
            if mses[i] > 2 * mses[i - 1]:
                # Truncate long texts for display
                display_text = text[:50] + '...' if len(text) > 50 else text
                bottlenecks[layers[i]].append({
                    'text': display_text,
                    'mse_before': mses[i - 1],
                    'mse_after': mses[i],
                    'increase_factor': mses[i] / mses[i - 1]
                })

    # Report findings
    for layer_idx in layers:
        if bottlenecks[layer_idx]:
            print(
                f"\nLayer {layer_idx} shows bottleneck behavior "
                f"for {len(bottlenecks[layer_idx])} probe(s):"
            )
            for item in bottlenecks[layer_idx]:
                print(f"  - {item['text']}")
                print(
                    f"    MSE jump: {item['mse_before']:.6f} -> "
                    f"{item['mse_after']:.6f} "
                    f"({item['increase_factor']:.1f}x increase)"
                )

    if not any(bottlenecks.values()):
        print(
            "No significant bottlenecks detected "
            "(no >2x MSE increases between adjacent layers)"
        )

    print("\n" + "=" * 80 + "\n")

    return bottlenecks


if __name__ == "__main__":
    print("RTRM Visualization Utilities")
    print("=" * 40)
    print()
    print("This is a utility module. Import it in your main script:")
    print()
    print("    from rtrm_visualizer import (")
    print("        RTRMVisualizer,")
    print("        analyze_layer_information_bottlenecks")
    print("    )")
    print()
    print("    visualizer = RTRMVisualizer(rtrm_system)")
    print("    visualizer.generate_comprehensive_report(probe_texts)")
