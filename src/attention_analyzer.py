"""
src/attention_analyzer.py

Attention analysis and visualization toolkit for decoder-only Transformer LMs:
- Extract multi-head attention weights per layer.
- Generate attention heatmaps (query vs. key position).
- Compute Attention Entropy per head/layer (diffuse vs. focused attention).
- Compute Mean Attention Distance per head/layer (local vs. long-range heads).
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict


class AttentionAnalyzer:
    """
    Extracts and computes attention statistics from model forward pass.
    """
    def __init__(self, model: torch.nn.Module, tokenizer, device: str = "cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def extract_attentions(self, text: str) -> Dict:
        """
        Extracts attention weight tensors across all layers for a given input sentence.
        Returns tensor of shape (n_layer, n_head, seq_len, seq_len) and token list.
        """
        self.model.eval()
        token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        
        # Forward pass with attention extraction
        _, _, attentions = self.model(input_tensor, return_attentions=True)
        
        # Stack attentions across layers -> shape: (n_layer, B=1, n_head, T, T)
        stacked = torch.stack(attentions, dim=0).squeeze(1).cpu() # (n_layer, n_head, T, T)
        
        tokens = [self.tokenizer.decode([t]) for t in token_ids]
        return {
            "attentions": stacked,
            "tokens": tokens,
            "text": text
        }

    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> np.ndarray:
        """
        Computes attention entropy for each head and layer.
        Higher entropy = diffuse attention; lower entropy = focused attention.
        Input shape: (n_layer, n_head, T, T)
        Output shape: (n_layer, n_head)
        """
        # Clamp to avoid log(0)
        eps = 1e-10
        att_clamped = torch.clamp(attention_weights, min=eps)
        
        # Entropy per query position: H(q) = - sum_k p(k) * log(p(k))
        entropy_per_query = -torch.sum(att_clamped * torch.log(att_clamped), dim=-1) # (n_layer, n_head, T)
        
        # Average across query positions
        mean_entropy = entropy_per_query.mean(dim=-1).numpy() # (n_layer, n_head)
        return mean_entropy

    def compute_mean_attention_distance(self, attention_weights: torch.Tensor) -> np.ndarray:
        """
        Computes mean attention distance (average token distance attended to).
        Input shape: (n_layer, n_head, T, T)
        Output shape: (n_layer, n_head)
        """
        n_layer, n_head, T, _ = attention_weights.shape
        
        # Distance matrix D[i, j] = |i - j|
        pos = torch.arange(T, dtype=torch.float32)
        dist_matrix = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1)) # (T, T)
        
        # Multiply attention weight by distance: sum_j p(i, j) * |i - j|
        weighted_dist = torch.sum(attention_weights * dist_matrix, dim=-1) # (n_layer, n_head, T)
        
        # Average across query positions
        mean_dist = weighted_dist.mean(dim=-1).numpy() # (n_layer, n_head)
        return mean_dist

    def plot_heatmap(
        self,
        attn_analysis: Dict,
        layer_idx: int,
        head_idx: int,
        save_path: Path = None,
        title: str = None
    ):
        """
        Plots query vs key attention heatmap for a specific layer and head.
        """
        att = attn_analysis["attentions"][layer_idx, head_idx].numpy()
        tokens = attn_analysis["tokens"]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(att, cmap="viridis")
        fig.colorbar(cax)

        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)

        ax.set_xlabel("Key Position", fontsize=10)
        ax.set_ylabel("Query Position", fontsize=10)
        
        if title:
            ax.set_title(title, fontsize=12, pad=20)
        else:
            ax.set_title(f"Layer {layer_idx + 1}, Head {head_idx + 1} Attention Map", fontsize=12, pad=20)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"[AttentionAnalyzer] Heatmap saved -> {save_path}")
        plt.close()

    def plot_multi_head_grid(
        self,
        attn_analysis: Dict,
        layer_idx: int,
        head_indices: List[int] = [0, 1, 2, 3],
        save_path: Path = None,
        title: str = None
    ):
        """
        Plots a 2x2 grid of attention heatmaps for multiple heads in a specific layer.
        """
        tokens = attn_analysis["tokens"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, head_idx in enumerate(head_indices[:4]):
            att = attn_analysis["attentions"][layer_idx, head_idx].numpy()
            ax = axes[idx]
            cax = ax.matshow(att, cmap="viridis")
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticklabels(tokens, fontsize=6)

            ax.set_xlabel("Key Position", fontsize=8)
            ax.set_ylabel("Query Position", fontsize=8)
            ax.set_title(f"Head {head_idx + 1}", fontsize=10)

        if title:
            fig.suptitle(title, fontsize=13, weight="bold")
        else:
            fig.suptitle(f"Layer {layer_idx + 1} Multi-Head Attention Heatmaps", fontsize=13, weight="bold")

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[AttentionAnalyzer] Multi-head grid saved -> {save_path}")
        plt.close()

