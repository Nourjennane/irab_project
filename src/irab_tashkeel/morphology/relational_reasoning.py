"""Phase 3.1 — relation-aware self-attention.

Layered on top of Phase 3-A (the production checkpoint). Takes the per-word
encoder feature ``pooled_irab`` and a per-sentence dep tree (parent indices
+ DEPREL per word) and produces a richer relational feature ``pooled_rel``
that the iʿrāb decoders consume.

Design choice (vs Phase 2 input-side conditioning, Phases 5/6 output-side
hierarchy): the relational signal here is **structurally injected** through
attention biased by dep edge type, not learned through a head whose
representation can drift. The attention mechanism reads the dep structure
once (as bias on attention scores) and lets the iʿrāb gradient teach the
attention to weight neighbours useful for case/role/marker prediction.

Identity initialisation: the attention's output projection is initialised
to zero, so step 0 yields ``pooled_rel = pooled_irab`` byte-exact. Gradient
pressure from L_irab is what teaches the attention + per-edge-type bias
to deviate from no-op.

Edge type vocabulary (76 entries):
    0:   <no_edge>           (no direct dep edge between i and j)
    1:   <self>              (i == j)
    2..38:  "j governs i with DEPREL d"   (i is the dependent of j)
    39..75: "i governs j with DEPREL d"   (j is the dependent of i)

Each edge type maps to a scalar bias added to the attention score. Total
new params: ~600K (single-head attention with d=768 + 76-entry embedding).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dep_schema import N_DEPREL


# Edge type vocabulary
N_EDGE_TYPES = 1 + 1 + N_DEPREL + N_DEPREL   # = 76 with N_DEPREL=37? wait.
# Let me recount: N_DEPREL = 38 (37 standard + <unk>). So edge types = 1 + 1 + 38 + 38 = 78.
N_EDGE_TYPES = 2 + 2 * N_DEPREL


class RelationAwareSelfAttention(nn.Module):
    """Single-head self-attention with per-edge-type scalar bias on attention scores.

    Args:
        hidden_size: encoder hidden dim (e.g. 768).
        n_deprel: number of DEPREL labels (default 38 from dep_schema).
    """

    def __init__(self, hidden_size: int, n_deprel: int = N_DEPREL):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_deprel = n_deprel
        self.n_edge_types = 2 + 2 * n_deprel

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.edge_bias = nn.Embedding(self.n_edge_types, 1)  # scalar per edge type

        self._identity_init()

    def _identity_init(self) -> None:
        """Init so step 0 output projection is zero → output = pooled_irab via residual."""
        # Standard small-noise init for Q/K/V (training will shape them).
        for lin in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        # Output projection: ZERO weight + bias so attention contribution is zero at step 0.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Edge bias: zero (no per-edge-type score modulation at step 0).
        nn.init.zeros_(self.edge_bias.weight)

    @staticmethod
    def build_edge_type_matrix(
        head_indices: torch.LongTensor,    # (B, W) — 1-based UD parent index, 0 = root
        deprel_ids: torch.LongTensor,      # (B, W) — DEPREL id per word
        word_mask: torch.LongTensor,        # (B, W) — 1 for valid words, 0 for pad
    ) -> torch.LongTensor:
        """For each (i, j) pair, return an edge type id ∈ [0, N_EDGE_TYPES).

        - 0: <no_edge>           (no direct dep edge between i and j)
        - 1: <self>              (i == j)
        - 2 + d (d ∈ 0..N_DEPREL-1):  "j governs i with deprel d"
            i.e. i's parent is j and i has deprel d
        - 2 + N_DEPREL + d:      "i governs j with deprel d"
            i.e. j's parent is i and j has deprel d
        """
        B, W = head_indices.shape
        device = head_indices.device

        # head_indices is 1-based; convert to 0-based: parent_idx[i] = head_indices[i] - 1, or -1 if root.
        parent = head_indices - 1   # (B, W); -1 means root

        # For each (i, j), is j the parent of i?  parent[i] == j ?
        # Build (B, W, W) tensors of i and j indices
        i_arange = torch.arange(W, device=device).unsqueeze(0).unsqueeze(-1)   # (1, W, 1)
        j_arange = torch.arange(W, device=device).unsqueeze(0).unsqueeze(0)    # (1, 1, W)

        # j_is_parent[b, i, j] = (parent[b, i] == j)
        parent_b_i = parent.unsqueeze(-1)                                       # (B, W, 1)
        j_is_parent = (parent_b_i == j_arange)                                  # (B, W, W)

        # i_is_parent[b, i, j] = (parent[b, j] == i)
        parent_b_j = parent.unsqueeze(1)                                        # (B, 1, W)
        i_is_parent = (parent_b_j == i_arange)                                  # (B, W, W)

        # Self diagonal
        i_eq_j = (i_arange == j_arange).expand(B, -1, -1)                       # (B, W, W)

        # Build edge type matrix
        edge_type = torch.zeros((B, W, W), dtype=torch.long, device=device)
        edge_type = torch.where(i_eq_j, torch.full_like(edge_type, 1), edge_type)

        # When j_is_parent: edge type = 2 + deprel(i)
        deprel_i = deprel_ids.unsqueeze(-1).expand(-1, -1, W)                   # (B, W, W)
        edge_type = torch.where(
            j_is_parent & ~i_eq_j,
            2 + deprel_i,
            edge_type,
        )
        # When i_is_parent: edge type = 2 + N_DEPREL + deprel(j)
        deprel_j = deprel_ids.unsqueeze(1).expand(-1, W, -1)                    # (B, W, W)
        edge_type = torch.where(
            i_is_parent & ~i_eq_j & ~j_is_parent,
            2 + N_DEPREL + deprel_j,
            edge_type,
        )
        return edge_type

    def forward(
        self,
        x: torch.Tensor,                    # (B, W, hidden)
        head_indices: torch.LongTensor,     # (B, W) 1-based UD parent index
        deprel_ids: torch.LongTensor,       # (B, W) DEPREL id
        word_mask: torch.LongTensor,        # (B, W)
    ) -> torch.Tensor:
        """Returns ``pooled_rel = x + out_proj(attn_output)``.

        At identity init, ``out_proj`` = 0, so output = x byte-exact.
        """
        B, W, hidden = x.shape

        Q = self.q_proj(x)   # (B, W, hidden)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Attention scores (B, W, W)
        scale = 1.0 / math.sqrt(hidden)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

        # Add per-edge-type bias
        edge_types = self.build_edge_type_matrix(head_indices, deprel_ids, word_mask)
        edge_bias = self.edge_bias(edge_types).squeeze(-1)   # (B, W, W)
        attn_scores = attn_scores + edge_bias

        # Mask out attention to padded positions
        # word_mask is (B, W) → (B, 1, W) broadcasts for the j dimension
        key_mask = word_mask.unsqueeze(1).to(torch.bool)   # (B, 1, W)
        attn_scores = attn_scores.masked_fill(~key_mask, float("-inf"))
        # Also mask query rows for padded i — set their output to zero post-attn.

        attn_probs = F.softmax(attn_scores, dim=-1)
        # If a query row is fully masked (all keys = pad), attn_probs is NaN; replace with 0.
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

        attn_output = torch.matmul(attn_probs, V)            # (B, W, hidden)
        attn_output = self.out_proj(attn_output)              # zero at init

        # Residual: pooled_rel = pooled_irab + (zero at init, learned later)
        out = x + attn_output

        # Zero out pad-position outputs for cleanliness
        out_mask = word_mask.unsqueeze(-1).to(out.dtype)
        return out * out_mask
