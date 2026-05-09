"""Lightweight graph refinement layer.

A small (2-layer) attention-based refiner that runs *on top* of
encoder token states using the existing grammar-graph edges. Output
shape matches input — drop-in residual addition before the existing
prediction heads.

Design choices
--------------

- We do **not** introduce a `torch_geometric` dependency. The refiner
  uses plain ``nn.MultiheadAttention`` with a per-head additive bias
  derived from edge-type embeddings; this is functionally equivalent
  to GATv2 with edge-type features and easier to ship.

- Hidden dim defaults to ``encoder_hidden // 2`` (768 → 384) so memory
  stays bounded.

- Two layers, residual + LayerNorm, dropout 0.15.

- A per-edge-type embedding feeds into the attention bias matrix, so
  long-range / dep / construction edges receive different prior
  weights — implementing item 5 ("edge-type attention bias") as a
  parameter of this module rather than a separate component.

Inputs at forward time:

  ``token_states`` (B, W, D)             — from the encoder, word-pooled
  ``edge_index``   (B, W, W)             — int64, edge type id per pair
                                            (0 = no edge; 1..K = edge types)
  ``token_mask``   (B, W)                — 1 = real token, 0 = pad

The refiner returns a residual-added refined state of the same shape.
Use ``GraphRefiner(...).enabled = False`` to ablate.
"""
from __future__ import annotations

from typing import Optional


def make_graph_refiner(*args, **kwargs):
    """Top-level factory so callers don't need to import torch eagerly."""
    return _make_lazy()(*args, **kwargs)


def _make_lazy():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _MHA_with_edge_bias(nn.Module):
        """Single-layer MHA where attention scores get an additive bias
        from a learned edge-type embedding."""

        def __init__(self, d_model: int, n_heads: int, n_edge_types: int,
                     dropout: float = 0.15):
            super().__init__()
            assert d_model % n_heads == 0
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.edge_bias = nn.Embedding(n_edge_types + 1, n_heads,
                                          padding_idx=0)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index, mask):
            B, W, D = x.shape
            q = self.q_proj(x).view(B, W, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, W, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, W, self.n_heads, self.head_dim).transpose(1, 2)
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (self.head_dim ** 0.5)

            # edge_index: (B, W, W) -> bias (B, W, W, H) -> (B, H, W, W)
            edge_b = self.edge_bias(edge_index).permute(0, 3, 1, 2)
            scores = scores + edge_b

            # Mask out padded positions (both q and k)
            if mask is not None:
                m = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,W)
                scores = scores.masked_fill(~m.bool(), float("-inf"))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
            out = out.transpose(1, 2).contiguous().view(B, W, D)
            return self.out_proj(out)

    class GraphRefiner(nn.Module):
        """Two-layer graph refiner with edge-type attention bias."""

        def __init__(self, d_in: int, n_edge_types: int = 8,
                     n_layers: int = 2, n_heads: int = 4,
                     hidden_ratio: float = 0.5, dropout: float = 0.15):
            super().__init__()
            d_hidden = max(16, int(d_in * hidden_ratio))
            self.proj_in = nn.Linear(d_in, d_hidden)
            self.proj_out = nn.Linear(d_hidden, d_in)
            self.layers = nn.ModuleList([
                _MHA_with_edge_bias(d_hidden, n_heads, n_edge_types, dropout)
                for _ in range(n_layers)
            ])
            self.norms = nn.ModuleList([
                nn.LayerNorm(d_hidden) for _ in range(n_layers)
            ])
            self.ffn = nn.Sequential(
                nn.Linear(d_hidden, d_hidden * 2), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden * 2, d_hidden),
            )
            self.ffn_norm = nn.LayerNorm(d_hidden)
            self.enabled = True

        def forward(self, token_states, edge_index=None, token_mask=None):
            """Refine token states using graph edges.

            If ``edge_index`` is None, falls back to all-pairs no-bias
            attention so the module still works for ablation.
            """
            if not self.enabled or token_states is None:
                return token_states
            B, W, D = token_states.shape
            if edge_index is None:
                edge_index = torch.zeros(B, W, W, dtype=torch.long,
                                          device=token_states.device)
            if token_mask is None:
                token_mask = torch.ones(B, W, dtype=torch.long,
                                          device=token_states.device)

            x = self.proj_in(token_states)
            for mha, norm in zip(self.layers, self.norms):
                x = norm(x + mha(x, edge_index, token_mask))
            x = self.ffn_norm(x + self.ffn(x))
            x = self.proj_out(x)
            # Residual back into the encoder space
            return token_states + x

    return GraphRefiner


# Edge-type id constants — keep aligned with the grammar-graph builder.
EDGE_TYPES = {
    "dep":                  1,
    "agreement":            2,
    "construction_member":  3,
    "clause_member":        4,
    "governor":             5,
    "overlap":              6,
    "discourse_link":       7,
    "coref":                8,
}
