"""
RTRM Complete Workflow Example
==============================

This script provides an end-to-end demonstration of the Reading The Robot Mind
(RTRM) methodology applied to GPT-2. It orchestrates the complete analysis
pipeline from initialization through visualization.

RTRM Methodology Context:
    RTRM enables Subject Matter Experts (SMEs) to perform quality assurance
    on neural network pipelines by reconstructing familiar input representations
    from intermediate layer states. This script serves as the main entry point
    for practitioners who want to understand how GPT-2 transforms text through
    its layers.

    By training decoders to invert each layer's representations back to token
    space, SMEs can observe where semantic information is preserved or discarded
    without requiring deep knowledge of transformer internals.

Workflow Steps:
    1. System initialization with configurable layer selection
    2. Domain-specific training data preparation from WikiText-2
    3. Autoencoder decoder training for selected transformer layers
    4. Probe point analysis showing layer-wise text reconstruction
    5. Cosine equivalence analysis (bag-of-words token similarity)
    6. Comprehensive visualization generation
    7. Information bottleneck detection

Output Artifacts:
    - Trained decoder weights: ./rtrm_models/
    - Text analysis reports: ./rtrm_results/probe_*_analysis.txt
    - Cosine similarity plots: ./rtrm_results/probe_*_cosine.png
    - Layer progression visualizations: ./rtrm_results/*.png

Dependencies:
    - rtrm_autoencoder.py (core RTRM implementation)
    - rtrm_visualizer.py (visualization utilities)
    - datasets (Hugging Face)
    - torch

Reference:
    Nussbaum, P. (2023). Reading the Robot Mind. CLEF 2023.
"""

import torch
from datasets import load_dataset

from rtrm_autoencoder import RTRMAutoencoder
from rtrm_visualizer import RTRMVisualizer, analyze_layer_information_bottlenecks


def prepare_training_data():
    """
    Prepare domain-specific training data from WikiText-2.

    For best results, use 50-200+ diverse, representative samples from
    a dataset matching your target domain. WikiText-2 aligns with GPT-2's
    evaluation domain.

    Returns:
        List of training text samples filtered for minimum length.
    """
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train"
    )
    training_texts = [
        text for text in dataset['text']
        if len(text) > 20
    ][:2000]

    return training_texts


def select_probe_points():
    """
    Select strategic probe points for RTRM analysis.

    Probe points should include:
        - Representative examples from your domain
        - Edge cases you want to verify
        - Examples that have failed in the past
        - Inputs near decision boundaries

    Returns:
        List of probe point text strings.
    """
    probe_points = [
        # Growing text sequences (test context accumulation)
        "Mary had a little lamb.",
        "Mary had a little lamb. It's fleece was white as snow.",
        "Mary had a little lamb. It's fleece was white as snow. "
        "Everywhere that Mary went,",
        "Mary had a little lamb. It's fleece was white as snow. "
        "Everywhere that Mary went, the lamb was sure to go.",

        # Nonsensical text (stress test for robustness)
        "Lamb fleece Mary.",
        "Lamb fleece Mary. Snow had went.",
        "Lamb fleece Mary. Snow had went. Go sure white.",
        "Lamb fleece Mary. Snow had went. Go sure white. "
        "Little it's everywhere.",

        # Representative example
        "Natural language processing systems analyze text to extract meaning.",

        # Edge case: very simple
        "The dog barked.",

        # Edge case: complex structure
        "Although machine learning has made remarkable progress in recent "
        "years, understanding the internal mechanisms remains challenging.",

        # Domain-specific example (customize for your use case)
        "The quarterly earnings report exceeded analyst expectations by "
        "significant margin.",

        # Ambiguous/challenging (multiple meanings of "bank")
        "The bank stands by the river.",
    ]

    return probe_points


def main():
    """Execute the complete RTRM analysis workflow."""
    print("\n" + "=" * 80)
    print("RTRM Analysis Workflow - GPT-2 Small")
    print("Reading The Robot Mind: Autoencoder Method")
    print("=" * 80 + "\n")

    # Check CUDA availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA available: {device_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB\n")
    else:
        print("CUDA not available - training will be slower")
        print("Consider using a GPU for better performance\n")

    # =========================================================================
    # Step 1: Initialize RTRM System
    # =========================================================================
    print("STEP 1: Initializing RTRM System...")
    print("-" * 80)

    # Layer selection options:
    #   - [0, 4, 8, 12] for quick overview
    #   - [0, 2, 4, 6, 8, 10, 12] for balanced analysis (default)
    #   - list(range(13)) for all layers
    rtrm = RTRMAutoencoder(
        model_name='gpt2',
        layers_to_analyze=[0, 2, 4, 6, 8, 10, 12]
    )
    print("RTRM system initialized\n")

    # =========================================================================
    # Step 2: Prepare Training Data
    # =========================================================================
    print("STEP 2: Preparing Training Data...")
    print("-" * 80)

    training_texts = prepare_training_data()
    print(f"Prepared {len(training_texts)} training samples")
    print(f"Sample: '{training_texts[0][:60]}...'\n")

    # =========================================================================
    # Step 3: Train Decoders
    # =========================================================================
    print("STEP 3: Training Decoders...")
    print("-" * 80)
    print("Training progress will be shown below.\n")

    config = {
        'epochs': 25,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'save_dir': './rtrm_models'
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    rtrm.train_decoders(texts=training_texts, **config)
    print("\nDecoder training complete\n")

    # =========================================================================
    # Step 4: Analyze Probe Points
    # =========================================================================
    print("STEP 4: Analyzing Probe Points...")
    print("-" * 80)

    probe_points = select_probe_points()
    print(f"Selected {len(probe_points)} probe points for analysis\n")

    for i, probe in enumerate(probe_points):
        print(f"\nProbe {i + 1}/{len(probe_points)}:")
        rtrm.analyze_probe_point(
            probe,
            output_file=f'./rtrm_results/probe_{i + 1}_analysis.txt'
        )

    print("\nProbe point analysis complete\n")

    # =========================================================================
    # Step 5: Cosine Equivalence Analysis
    # =========================================================================
    print("STEP 5: Cosine Equivalence Analysis...")
    print("-" * 80)

    visualizer = RTRMVisualizer(rtrm)

    for i, probe in enumerate(probe_points):
        print(f"\nCosine analysis for probe {i + 1}/{len(probe_points)}...")
        cosine_results = rtrm.analyze_cosine_equivalence(
            probe,
            training_texts,
            threshold=0.85,
            top_k=3
        )

        visualizer.plot_cosine_equivalence(
            probe,
            cosine_results,
            save_path=f'./rtrm_results/probe_{i + 1}_cosine.png'
        )

    print("\nCosine equivalence analysis complete\n")

    # =========================================================================
    # Step 6: Generate Visualizations
    # =========================================================================
    print("STEP 6: Generating Visualizations...")
    print("-" * 80)

    visualizer.generate_comprehensive_report(
        probe_texts=probe_points,
        output_dir='./rtrm_results'
    )

    print("Visualizations complete\n")

    # =========================================================================
    # Step 7: Bottleneck Analysis
    # =========================================================================
    print("STEP 7: Analyzing Information Bottlenecks...")
    print("-" * 80)

    analyze_layer_information_bottlenecks(rtrm, probe_points)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("RTRM Analysis Complete")
    print("=" * 80 + "\n")

    print("Results saved to:")
    print("  - Model weights: ./rtrm_models/")
    print("  - Analysis results: ./rtrm_results/")
    print("  - Text analyses: ./rtrm_results/probe_*_analysis.txt")
    print("  - Cosine equivalence: ./rtrm_results/probe_*_cosine.png")
    print("  - Visualizations: ./rtrm_results/*.png")

    print("\n" + "=" * 80)
    print("Key Insights to Look For")
    print("=" * 80)
    print("""
1. RECONSTRUCTION QUALITY (Autoencoder Method):
   - Early layers (0-4) should reconstruct almost perfectly
   - Middle layers (5-8) show where semantic compression occurs
   - Late layers (9-12) focus on task objectives, not reconstruction

2. COSINE EQUIVALENCE (Bag-of-Words Method):
   - Which training tokens are most similar to each probe token?
   - When do tokens start mapping to semantically similar alternatives?
   - At which layer does "bank" (financial) vs "bank" (river) diverge?

3. INFORMATION BOTTLENECKS:
   - Sharp accuracy drops indicate information discard
   - Gradual degradation suggests smooth transformation
   - Check if bottlenecks align with architectural choices

4. LAYER-WISE PATTERNS:
   - When does task-specific processing begin?
   - Which layers preserve input details vs. abstract features?
   - Are there unexpected information losses?

5. COMPARATIVE INSIGHTS:
   - How do simple vs. complex inputs degrade differently?
   - Are edge cases handled appropriately?
   - Do different probe points show consistent patterns?
    """)

    print("=" * 80)
    print("Next Steps")
    print("=" * 80)
    print("""
1. Review the generated visualizations in ./rtrm_results/
2. Examine text analyses for semantic preservation
3. Compare reconstruction patterns across probe points
4. Check cosine equivalence for token clustering behavior
5. Identify layers where critical information is lost
6. Consider architectural adjustments if bottlenecks are problematic
7. Use insights for model debugging and validation
    """)

    print("=" * 80)
    print("Important Note")
    print("=" * 80)
    print("""
RTRM provides visibility into model internals but does not:
- Guarantee optimal performance
- Identify all failure modes
- Replace comprehensive testing
- Prove model safety or reliability

Use RTRM as one tool among many for understanding and improving
your AI system. Always validate findings with domain expertise
and rigorous testing.
    """)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        print("Check the error message and adjust parameters as needed.")
        raise
