"""
RTRM Autoencoder Implementation for GPT-2
=========================================

This module implements the core RTRM (Reading The Robot Mind) autoencoder
method for transformer language models. It provides the pseudo-inverse
functions that reconstruct input tokens from intermediate layer activations,
enabling Subject Matter Experts (SMEs) to visualize information retention
across network depth.

RTRM Methodology Context:
    Neural networks process inputs through sequential transformations that
    progressively discard information irrelevant to the task. RTRM trains
    lightweight decoder networks to invert these transformations, producing
    reconstructions in the original input space (tokens) that SMEs can
    directly interpret without requiring deep technical knowledge.

    For a layer with hidden state h, the decoder D approximates:
        D(h) -> original_tokens

    Reconstruction quality degradation across layers reveals where the
    network discards information, highlighting potential sources of
    downstream errors that SMEs can investigate.

Architecture Components:
    - GPT2ActivationExtractor: Hooks into transformer blocks to capture
      intermediate hidden states after each layer.
    - LayerDecoder: Lightweight MLP with weight-tied LM head that maps
      hidden states back to vocabulary logits.
    - TextReconstructionDataset: Caches extracted activations per-layer
      for efficient decoder training.
    - RTRMAutoencoder: Orchestrates training and analysis across all
      selected layers.

Key Features:
    - Weight tying: Reuses GPT-2's embedding matrix as decoder output,
      ensuring reconstructions use the same token space.
    - Per-layer caching: Avoids redundant forward passes during training.
    - Hinge loss training: Encourages correct token logits to exceed
      incorrect alternatives by a margin.
    - Cosine equivalence analysis: Identifies semantically similar tokens
      across training corpus via embedding similarity.

Output Artifacts:
    - Trained decoder weights: ./rtrm_models/decoder_layer_*.pt
    - Cached activations: ./rtrm_activations/
    - Analysis text files and plots (via analyze_probe_point)

Dependencies:
    - transformers (Hugging Face GPT-2)
    - torch
    - numpy, matplotlib, tqdm

Reference:
    Nussbaum, P. (2023). Reading the Robot Mind. CLEF 2023.
"""

import hashlib
import os
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2ActivationExtractor:
    """
    Extract intermediate activations from GPT-2 at each transformer layer.

    GPT-2 Small has 12 transformer blocks. This class captures activations
    after each block's output (post layer norm, pre next block).

    Attributes:
        device: PyTorch device for computation.
        model: Loaded GPT-2 model in evaluation mode.
        tokenizer: GPT-2 tokenizer with pad token configured.
        num_layers: Number of transformer layers (12 for GPT-2 Small).
        hidden_size: Hidden dimension (768 for GPT-2 Small).
    """

    def __init__(self, model_name: str = 'gpt2'):
        """
        Initialize the GPT-2 model and prepare for activation extraction.

        Args:
            model_name: HuggingFace model identifier. Default 'gpt2' loads
                GPT-2 Small with 124M parameters.
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self.num_layers = len(self.model.transformer.h)
        self.hidden_size = self.model.config.n_embd

        print(
            f"Loaded {model_name}: {self.num_layers} layers, "
            f"{self.hidden_size}D hidden size"
        )

    def extract_activations(
        self,
        text: str,
        max_length: int = 128
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Extract activations from all layers for a given text input.

        Args:
            text: Input text string to process.
            max_length: Maximum sequence length for tokenization.

        Returns:
            Tuple containing:
                - activations: Dict mapping layer indices (0-12) to tensors
                  of shape [seq_len, hidden_size]. Layer 0 is post-embedding,
                  layers 1-12 are post each transformer block.
                - input_ids: Token ID tensor of shape [seq_len].
                - attention_mask: Mask tensor of shape [seq_len].
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        attention_mask = attention_mask.to(torch.float32)

        activations = {}

        with torch.no_grad():
            # Layer 0: Post-embedding (before any transformer blocks)
            hidden_states = self.model.transformer.wte(input_ids)
            position_ids = torch.arange(
                0, input_ids.size(1),
                dtype=torch.long,
                device=self.device
            )
            position_embeds = self.model.transformer.wpe(position_ids)
            hidden_states = hidden_states + position_embeds
            hidden_states = self.model.transformer.drop(hidden_states)

            activations[0] = hidden_states.squeeze(0)

            # Layers 1-12: Post each transformer block
            for i, block in enumerate(self.model.transformer.h):
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask
                )[0]
                activations[i + 1] = hidden_states.squeeze(0)

        return activations, input_ids.squeeze(0), attention_mask.squeeze(0)


class LayerDecoder(nn.Module):
    """
    Decoder network that reconstructs token predictions from layer activations.

    Uses weight tying: reuses GPT-2's embedding matrix as the output projection
    (frozen). This matches how GPT-2 generates tokens and ensures reconstructions
    use the same vocabulary space without adding trainable parameters for the
    output layer.

    Attributes:
        layer_idx: Which transformer layer this decoder inverts.
        input_dim: Dimension of input activations.
        decoder: MLP that projects activations toward embedding space.
        lm_head: Frozen embedding matrix for vocabulary projection.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_matrix: torch.Tensor,
        layer_idx: int
    ):
        """
        Initialize the layer decoder.

        Args:
            input_dim: Dimension of the layer's activations (768 for GPT-2).
            embedding_matrix: GPT-2's token embedding weights [vocab_size, 768].
                This is registered as a buffer (frozen, not trained).
            layer_idx: Which layer this decoder is for (informational only).
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.input_dim = input_dim

        # Adaptive architecture based on input dimensionality
        if input_dim <= 512:
            # Deeper network for low-dimensional bottleneck layers
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 1536),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1536, 1024),
                nn.ReLU(),
                nn.Linear(1024, 768)
            )
        else:
            # Standard architecture for higher dimensions
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 768)
            )

        # Weight-tied output: reuse GPT-2's embedding matrix (frozen)
        self.register_buffer('lm_head', embedding_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute vocabulary logits from layer activations.

        Args:
            x: Activations of shape [batch_size, seq_len, input_dim].

        Returns:
            Logits of shape [batch_size, seq_len, vocab_size].
        """
        hidden = self.decoder(x)
        logits = torch.matmul(hidden, self.lm_head.T)
        return logits


class TextReconstructionDataset(Dataset):
    """
    Dataset for training layer decoders.

    Extracts and caches activations for a specific layer to avoid redundant
    forward passes during training. Uses content-based hashing to detect
    when cached activations can be reused.

    Attributes:
        texts: Original text samples.
        max_length: Maximum sequence length.
        extractor: GPT2ActivationExtractor instance.
        layer_idx: Which layer's activations to extract.
        data: List of dicts containing cached activations and targets.
    """

    def __init__(
        self,
        texts: List[str],
        extractor: GPT2ActivationExtractor,
        layer_idx: int,
        max_length: int = 128,
        activations_dir: str = './rtrm_activations'
    ):
        """
        Initialize the dataset, loading from cache if available.

        Args:
            texts: List of text samples for training.
            extractor: GPT2ActivationExtractor instance.
            layer_idx: Which layer to extract activations for.
            max_length: Maximum sequence length for tokenization.
            activations_dir: Directory for caching extracted activations.
        """
        self.texts = texts
        self.max_length = max_length
        self.extractor = extractor
        self.layer_idx = layer_idx
        os.makedirs(activations_dir, exist_ok=True)

        # Content-based cache key
        text_hash = hashlib.md5(''.join(texts).encode()).hexdigest()
        cache_file = os.path.join(
            activations_dir,
            f"activations_{text_hash}_layer_{layer_idx}.pkl"
        )

        if os.path.exists(cache_file):
            print(
                f"Loading cached activations for layer {layer_idx} "
                f"from {cache_file}..."
            )
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded {len(self.data)} cached samples for layer {layer_idx}")
        else:
            print(
                f"Extracting activations for layer {layer_idx} "
                f"from {len(texts)} samples..."
            )
            self.data = []
            for text in tqdm(texts, desc=f"Layer {layer_idx}"):
                activations, input_ids, attention_mask = \
                    extractor.extract_activations(text, max_length)

                # Keep only target layer, free memory from others
                layer_activation = activations[layer_idx].clone()
                del activations

                self.data.append({
                    'text': text,
                    'activation': layer_activation,
                    'target_tokens': input_ids,
                    'attention_mask': attention_mask
                })

            print(f"Saving layer {layer_idx} activations to {cache_file}...")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"Cached {len(self.data)} samples for layer {layer_idx}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return sample at index."""
        return self.data[idx]


class RTRMAutoencoder:
    """
    Main RTRM system using the Autoencoder (Learned Inverse) method.

    Trains separate decoder networks for each selected layer to reconstruct
    original input tokens from that layer's activations. This reveals what
    information is preserved vs. discarded at each processing stage.

    Attributes:
        extractor: GPT2ActivationExtractor for forward passes.
        model_dir: Directory for saving/loading decoder weights.
        layers_to_analyze: List of layer indices with decoders.
        decoders: Dict mapping layer index to LayerDecoder.
        training_history: Dict mapping layer index to training metrics.
    """

    def __init__(
        self,
        model_name: str = 'gpt2',
        layers_to_analyze: Optional[List[int]] = None,
        model_dir: str = './rtrm_models'
    ):
        """
        Initialize the RTRM autoencoder system.

        Args:
            model_name: GPT-2 model variant to analyze.
            layers_to_analyze: Which layers to train decoders for.
                Default [0, 2, 4, 6, 8, 10, 12] covers key architectural points.
            model_dir: Directory for saving/loading decoder weights.
        """
        self.extractor = GPT2ActivationExtractor(model_name)
        self.model_dir = model_dir

        if layers_to_analyze is None:
            # Default: embedding + every other layer + final
            self.layers_to_analyze = [0, 2, 4, 6, 8, 10, 12]
        else:
            self.layers_to_analyze = layers_to_analyze

        self.decoders = {}
        self.training_history = {}

        # Get embedding matrix for weight tying
        embedding_matrix = self.extractor.model.transformer.wte.weight.data

        for layer_idx in self.layers_to_analyze:
            decoder = LayerDecoder(
                input_dim=self.extractor.hidden_size,
                embedding_matrix=embedding_matrix,
                layer_idx=layer_idx
            )
            self.decoders[layer_idx] = decoder.to(self.extractor.device)
            print(f"Initialized decoder for layer {layer_idx}")

        self._try_load_decoders()

    def _try_load_decoders(self):
        """Load pre-trained decoders if they exist."""
        if not os.path.exists(self.model_dir):
            return

        loaded = []
        for layer_idx in self.layers_to_analyze:
            decoder_path = os.path.join(
                self.model_dir,
                f"decoder_layer_{layer_idx}.pt"
            )
            if os.path.exists(decoder_path):
                self.decoders[layer_idx].load_state_dict(
                    torch.load(decoder_path, map_location=self.extractor.device)
                )
                self.decoders[layer_idx].eval()
                loaded.append(layer_idx)

        if loaded:
            print(f"\nLoaded pre-trained decoders for layers: {loaded}")
            print(f"(Delete files in {self.model_dir}/ to retrain)\n")

    def train_decoders(
        self,
        texts: List[str],
        epochs: int = 700,
        batch_size: int = 16,
        learning_rate: float = 5e-4,
        save_dir: str = './rtrm_models',
        activations_dir: str = './rtrm_activations'
    ):
        """
        Train decoder networks for all layers that don't have saved weights.

        Args:
            texts: Training text samples. Use 50-200+ diverse samples from
                the target domain for best results.
            epochs: Number of training epochs per decoder.
            batch_size: Batch size for training.
            learning_rate: Initial learning rate (reduced on plateau).
            save_dir: Directory to save trained decoder weights.
            activations_dir: Directory to cache extracted activations.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Check which decoders need training
        existing_decoders = []
        missing_decoders = []

        for layer_idx in self.layers_to_analyze:
            decoder_path = os.path.join(
                save_dir,
                f"decoder_layer_{layer_idx}.pt"
            )
            if os.path.exists(decoder_path):
                existing_decoders.append(layer_idx)
            else:
                missing_decoders.append(layer_idx)

        if existing_decoders:
            print(f"\nFound existing decoders for layers: {existing_decoders}")
            print(f"(Delete files in {save_dir}/ to retrain)\n")

        if not missing_decoders:
            print("All decoders already trained. Skipping training.\n")
            return

        print(f"Training decoders for layers: {missing_decoders}\n")

        for layer_idx in missing_decoders:
            print(f"\n{'=' * 60}")
            print(f"Training decoder for Layer {layer_idx}")
            print(f"{'=' * 60}")

            # Create dataset for this layer (uses cache if available)
            dataset = TextReconstructionDataset(
                texts,
                self.extractor,
                layer_idx,
                activations_dir=activations_dir
            )

            def collate_fn(batch):
                """Move batch tensors to device."""
                for item in batch:
                    item['activation'] = item['activation'].to(
                        self.extractor.device
                    )
                    item['target_tokens'] = item['target_tokens'].to(
                        self.extractor.device
                    )
                    item['attention_mask'] = item['attention_mask'].to(
                        self.extractor.device
                    )
                return batch

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )

            decoder = self.decoders[layer_idx]
            optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=200
            )

            history = {'train_loss': []}
            best_loss = float('inf')

            for epoch in range(epochs):
                decoder.train()
                epoch_loss = 0.0
                num_batches = 0

                for batch in dataloader:
                    activations_batch = [item['activation'] for item in batch]
                    tokens_batch = [item['target_tokens'] for item in batch]
                    masks_batch = [item['attention_mask'] for item in batch]

                    activations = torch.stack(activations_batch)
                    target_tokens = torch.stack(tokens_batch)
                    masks = torch.stack(masks_batch).to(torch.float32)

                    logits = decoder(activations)

                    # Hinge loss: encourage correct logit > max wrong logit
                    batch_size_cur, seq_len, vocab_size = logits.shape
                    logits_flat = logits.view(-1, vocab_size)
                    tokens_flat = target_tokens.view(-1)
                    masks_flat = masks.view(-1)

                    correct_logits = logits_flat.gather(
                        1, tokens_flat.unsqueeze(1)
                    ).squeeze(1)

                    logits_masked = logits_flat.clone()
                    logits_masked.scatter_(
                        1, tokens_flat.unsqueeze(1), float('-inf')
                    )
                    wrong_max_logits = logits_masked.max(dim=1).values

                    margin = 1.0
                    loss = F.relu(margin + wrong_max_logits - correct_logits)
                    loss = (loss * masks_flat).sum() / masks_flat.sum()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / num_batches
                history['train_loss'].append(avg_loss)
                scheduler.step(avg_loss)

                if (epoch + 1) % 25 == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(
                        decoder.state_dict(),
                        os.path.join(save_dir, f"decoder_layer_{layer_idx}.pt")
                    )

            self.training_history[layer_idx] = history
            print(f"Layer {layer_idx} - Best Loss: {best_loss:.6f}")

    def reconstruct_text(
        self,
        text: str,
        layer_idx: int
    ) -> Tuple[str, float]:
        """
        Reconstruct text from a specific layer's activations.

        This is the core RTRM operation: given an input text, extract the
        activations at the specified layer, run them through the trained
        decoder, and produce a reconstruction.

        Args:
            text: Original input text.
            layer_idx: Which layer to reconstruct from.

        Returns:
            Tuple of (reconstructed_text, token_accuracy).
            - reconstructed_text: Decoded text from predicted tokens.
            - token_accuracy: Fraction of tokens correctly reconstructed.

        Raises:
            ValueError: If no decoder is trained for the specified layer.
        """
        if layer_idx not in self.decoders:
            raise ValueError(f"No decoder trained for layer {layer_idx}")

        activations, input_ids, attention_mask = \
            self.extractor.extract_activations(text)
        layer_activation = activations[layer_idx].unsqueeze(0)
        target_tokens = input_ids.to(self.extractor.device)

        decoder = self.decoders[layer_idx]
        decoder.eval()

        with torch.no_grad():
            logits = decoder(layer_activation)
            predicted_tokens = torch.argmax(logits[0], dim=-1)
            mask = attention_mask.to(self.extractor.device)

            # Debug output
            print(f"\n=== Debug Layer {layer_idx} ===")
            print(f"Target tokens:    {target_tokens[:10].tolist()}")
            print(f"Predicted tokens: {predicted_tokens[:10].tolist()}")
            print(f"Attention mask:   {attention_mask[:10].tolist()}")

            for i in range(min(5, int(mask.sum().item()))):
                target_word = self.extractor.tokenizer.decode(
                    [target_tokens[i].item()]
                )
                pred_word = self.extractor.tokenizer.decode(
                    [predicted_tokens[i].item()]
                )
                match = "MATCH" if target_tokens[i] == predicted_tokens[i] else "DIFF"
                print(f"Pos {i}: '{target_word}' vs '{pred_word}' [{match}]")

            # Calculate accuracy over non-padding tokens
            correct = (predicted_tokens == target_tokens) & (mask == 1)
            accuracy = correct.sum().item() / mask.sum().item()

            # Decode non-padding tokens to text
            valid_length = int(mask.sum().item())
            predicted_token_list = predicted_tokens[:valid_length].tolist()

            reconstructed_text = self.extractor.tokenizer.decode(
                predicted_token_list,
                skip_special_tokens=True
            )

        return reconstructed_text, accuracy

    def analyze_probe_point(
        self,
        text: str,
        output_file: Optional[str] = None
    ) -> List[dict]:
        """
        Perform full RTRM analysis on a probe point across all trained layers.

        This is the primary SME-facing function. Given a probe point (a
        strategically selected input), it reconstructs the input from each
        layer and reports reconstruction quality.

        Args:
            text: Probe point text to analyze.
            output_file: Optional path to save results as text file.

        Returns:
            List of result dicts, each containing:
                - layer: Layer index.
                - reconstructed_text: Text reconstructed from that layer.
                - accuracy: Token-level accuracy.
        """
        print(f"\n{'=' * 80}")
        print("RTRM Analysis - Autoencoder Method")
        print(f"{'=' * 80}")
        print(f"\nOriginal Text:")
        print(f"  {text}")
        print(f"\n{'=' * 80}")

        results = []

        for layer_idx in sorted(self.layers_to_analyze):
            reconstructed_text, accuracy = self.reconstruct_text(text, layer_idx)

            results.append({
                'layer': layer_idx,
                'reconstructed_text': reconstructed_text,
                'accuracy': accuracy
            })

            print(f"\nLayer {layer_idx:2d} | Accuracy: {accuracy * 100:.1f}%")
            print(f"  Reconstruction: {reconstructed_text}")

        print(f"\n{'=' * 80}")

        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("RTRM Analysis Results\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(f"Original Text:\n{text}\n\n")
                f.write(f"{'=' * 80}\n\n")

                for result in results:
                    f.write(
                        f"Layer {result['layer']:2d} | "
                        f"Accuracy: {result['accuracy'] * 100:.1f}%\n"
                    )
                    f.write(f"  {result['reconstructed_text']}\n\n")

        return results

    def plot_reconstruction_quality(
        self,
        save_path: str = 'reconstruction_quality.png'
    ):
        """
        Plot final training loss across layers.

        Higher loss at later layers indicates more information has been
        discarded, making reconstruction more difficult.

        Args:
            save_path: Path to save the plot image.
        """
        if not self.training_history:
            print("No training history available. Train decoders first.")
            return

        layers = sorted(self.training_history.keys())
        losses = [
            self.training_history[layer]['train_loss'][-1]
            for layer in layers
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(layers, losses, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Reconstruction Loss', fontsize=12)
        plt.title('RTRM: Information Retention Across Layers', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(layers)

        plt.axhline(
            y=min(losses),
            color='g',
            linestyle='--',
            alpha=0.3,
            label='Best Reconstruction'
        )

        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()

    def _build_token_pools(
        self,
        training_texts: List[str],
        cache_dir: str = './rtrm_activations'
    ) -> Dict[int, dict]:
        """
        Build token embedding pools for cosine equivalence analysis.

        For each layer, extracts embeddings for every token in the training
        corpus and caches them for efficient similarity computation.

        Args:
            training_texts: Training texts to extract tokens from.
            cache_dir: Directory to cache token pools.

        Returns:
            Dict mapping layer_idx to dict containing:
                - embeddings: Normalized token embeddings [N, hidden_size].
                - tokens: List of token strings.
                - sources: List of source identifiers.
        """
        text_hash = hashlib.md5(''.join(training_texts).encode()).hexdigest()
        os.makedirs(cache_dir, exist_ok=True)

        token_pools = {}

        for layer_idx in self.layers_to_analyze:
            cache_file = os.path.join(
                cache_dir,
                f"token_pool_{text_hash}_layer_{layer_idx}.pkl"
            )

            if os.path.exists(cache_file):
                print(f"Loading cached token pool for layer {layer_idx}...")
                with open(cache_file, 'rb') as f:
                    token_pools[layer_idx] = pickle.load(f)
            else:
                print(f"Building token pool for layer {layer_idx}...")
                all_tokens = []
                all_embeddings = []
                all_sources = []

                for text_idx, train_text in enumerate(training_texts):
                    train_acts, train_ids, train_mask = \
                        self.extractor.extract_activations(train_text)
                    train_len = int(train_mask.sum().item())

                    for pos in range(train_len):
                        all_embeddings.append(train_acts[layer_idx][pos])
                        token_id = train_ids[pos].item()
                        token_text = self.extractor.tokenizer.decode([token_id])
                        all_tokens.append(token_text)
                        all_sources.append(f"text_{text_idx}_pos_{pos}")

                train_pool = torch.stack(all_embeddings)
                train_pool = F.normalize(train_pool, dim=1)

                token_pools[layer_idx] = {
                    'embeddings': train_pool,
                    'tokens': all_tokens,
                    'sources': all_sources
                }

                print(f"Saving token pool for layer {layer_idx}...")
                with open(cache_file, 'wb') as f:
                    pickle.dump(token_pools[layer_idx], f)

        print(f"Token pools ready for {len(token_pools)} layers")
        return token_pools

    def analyze_cosine_equivalence(
        self,
        probe_text: str,
        training_texts: List[str],
        threshold: float = 0.9,
        top_k: int = 3,
        cache_dir: str = './rtrm_activations'
    ) -> Dict[int, List[dict]]:
        """
        Find nearest training tokens to each probe token via cosine similarity.

        This implements the "bag-of-words" RTRM method: for each token position
        in the probe, find the most similar tokens from anywhere in the training
        corpus. This reveals when the network starts treating different tokens
        as equivalent.

        Args:
            probe_text: Text to analyze.
            training_texts: Training corpus to build token pool from.
            threshold: Minimum similarity threshold (informational only).
            top_k: Number of most similar tokens to return per position.
            cache_dir: Directory to cache token pools.

        Returns:
            Dict mapping layer_idx to list of per-position results.
            Each position result contains:
                - probe_token: The original probe token at this position.
                - matches: List of top_k dicts with token, similarity, source.
        """
        token_pools = self._build_token_pools(training_texts, cache_dir)

        probe_activations, probe_ids, probe_mask = \
            self.extractor.extract_activations(probe_text)
        probe_len = int(probe_mask.sum().item())

        results = {}

        for layer_idx in self.layers_to_analyze:
            pool_data = token_pools[layer_idx]
            train_pool = pool_data['embeddings']
            all_tokens = pool_data['tokens']
            all_sources = pool_data['sources']

            probe_vecs = probe_activations[layer_idx][:probe_len]
            probe_vecs = F.normalize(probe_vecs, dim=1)

            # Compute all pairwise similarities
            similarities = probe_vecs @ train_pool.T

            position_matches = []
            for pos in range(probe_len):
                probe_token = self.extractor.tokenizer.decode(
                    [probe_ids[pos].item()]
                )
                top_sims, top_indices = torch.topk(similarities[pos], k=top_k)

                matches = []
                for sim, idx in zip(top_sims.tolist(), top_indices.tolist()):
                    matches.append({
                        'token': all_tokens[idx],
                        'similarity': sim,
                        'source': all_sources[idx]
                    })

                position_matches.append({
                    'probe_token': probe_token,
                    'matches': matches
                })

            results[layer_idx] = position_matches

        return results


def main_example():
    """
    Example usage of the RTRM Autoencoder system.

    Demonstrates the complete workflow:
    1. Initialize system with GPT-2
    2. Train decoders on sample texts
    3. Analyze probe points
    4. Generate quality plot
    """
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models process information in layers.",
        "Natural language understanding requires context and semantics.",
        "Deep neural networks learn hierarchical representations.",
        "The cat sat on the mat and watched the birds.",
        "Artificial intelligence systems can analyze complex patterns.",
        "Text generation involves predicting the next word in sequence.",
        "Transformer models use attention mechanisms for processing.",
        "Understanding how AI works helps build better systems.",
        "Neural networks transform input data through multiple stages.",
    ]

    print("Initializing RTRM Autoencoder System...")
    rtrm = RTRMAutoencoder(model_name='gpt2')

    print("\nTraining decoders...")
    rtrm.train_decoders(
        texts=training_texts,
        epochs=700,
        batch_size=4,
        learning_rate=5e-4
    )

    probe_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence transforms modern technology.",
    ]

    for probe in probe_texts:
        rtrm.analyze_probe_point(
            probe,
            output_file=f'rtrm_analysis_{hash(probe)}.txt'
        )

    rtrm.plot_reconstruction_quality()


if __name__ == "__main__":
    main_example()
