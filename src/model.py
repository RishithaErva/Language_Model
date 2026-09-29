"""
src/model.py

Decoder-only (GPT-style) Transformer implementation built from scratch using primitive PyTorch layers.
Strictly adheres to assignment requirements:
- NO nn.Transformer* modules or HuggingFace model classes.
- Primitive layers: nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout.
- Multi-head causal self-attention implemented from first principles.
- Additive upper-triangular causal mask (-inf on future positions).
- Pre-LN architecture with residual connections.
- Parameter-saving weight tying option between input embedding and output projection.
- Empirical causal mask verification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTInputEmbeddings(nn.Module):
    """
    Input representation layer combining token embeddings and learned positional embeddings.
    """
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum supported sequence length {self.max_seq_len}")
        
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        token_embeds = self.wte(input_ids)       # (B, T, d_model)
        pos_embeds = self.wpe(pos_ids)           # (T, d_model)
        
        return self.drop(token_embeds + pos_embeds)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention from first principles.
    Projects X into Q, K, V, reshapes into h heads, applies scaled dot-product attention
    with an upper-triangular causal mask (-inf on future positions), and projects output.
    """
    def __init__(self, d_model: int, n_head: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, f"d_model ({d_model}) must be divisible by n_head ({n_head})"
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        # Combined linear projection for Query, Key, Value
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask: tril is 1 for allowed positions (<= t), 0 for future positions (> t)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, T, C = x.size()
        
        # Project and split into Q, K, V -> (B, T, d_model) each
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        
        # Reshape for multi-head attention: (B, h, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention per head: (B, h, T, T)
        # Scaling by 1/sqrt(d_k) prevents dot-product values from growing large, keeping softmax gradients stable
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        
        # Causal masking: replace future positions with -inf
        mask_slice = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask_slice == 0, float('-inf'))
        
        # Softmax over key dimension
        attn_weights = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_weights)
        
        # Weighted sum of values: (B, h, T, d_k)
        y = attn_probs @ v
        
        # Concatenate heads: (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        out = self.resid_dropout(self.c_proj(y))
        
        if return_attention:
            return out, attn_weights
        return out


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network with GELU non-linearity.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Standard Pre-LN Transformer decoder block:
    Pre-LN 1 -> MHA -> Residual -> Pre-LN 2 -> FFN -> Residual
    """
    def __init__(self, d_model: int, n_head: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if return_attention:
            attn_out, weights = self.attn(self.ln_1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, weights
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


class GPTLanguageModel(nn.Module):
    """
    Decoder-only GPT Language Model targeting ~25M parameters.
    """
    def __init__(
        self,
        vocab_size: int = 12000,
        d_model: int = 512,
        n_layer: int = 6,
        n_head: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights

        self.embeddings = GPTInputEmbeddings(vocab_size, d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, max_seq_len, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Parameter-saving weight tying
        if tie_weights:
            self.lm_head.weight = self.embeddings.wte.weight

        # Initialize weights custom GPT-style
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
        return_attentions: bool = False
    ):
        """
        Forward pass computing logits and optionally CrossEntropy loss or attention maps.
        """
        x = self.embeddings(input_ids)
        
        all_attentions = [] if return_attentions else None
        for block in self.blocks:
            if return_attentions:
                x, attn_weights = block(x, return_attention=True)
                all_attentions.append(attn_weights)
            else:
                x = block(x)
                
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for CrossEntropyLoss: (B * T, vocab_size) vs (B * T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_attentions:
            return logits, loss, all_attentions
        return logits, loss

    def count_parameters(self) -> dict:
        """
        Returns parameter breakdown.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb = sum(p.numel() for p in self.embeddings.parameters())
        blocks = sum(p.numel() for p in self.blocks.parameters())
        head = sum(p.numel() for p in self.lm_head.parameters()) if not self.tie_weights else 0
        
        return {
            "total_trainable": total,
            "embeddings": emb,
            "transformer_blocks": blocks,
            "lm_head": head,
            "weight_tying": self.tie_weights
        }

    @torch.no_grad()
    def verify_causal_masking(self) -> bool:
        """
        Empirically verifies that position t cannot see position t+1.
        Changes token at index t+1 and verifies that logits at position t remain IDENTICAL.
        """
        self.eval()
        seq = torch.randint(0, self.vocab_size, (1, 10))
        
        # Forward pass on original sequence
        logits1, _ = self.forward(seq)
        
        # Modify token at position t=5 (index 5)
        seq_modified = seq.clone()
        seq_modified[0, 5] = (seq[0, 5] + 1) % self.vocab_size
        
        # Forward pass on modified sequence
        logits2, _ = self.forward(seq_modified)
        
        # Check logits at positions 0 to 4 (position t < 5)
        diff_before_t = (logits1[0, :5, :] - logits2[0, :5, :]).abs().max().item()
        
        # Must be practically zero (within floating point precision)
        is_causal = diff_before_t < 1e-5
        return is_causal
