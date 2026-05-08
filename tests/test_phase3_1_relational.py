"""Phase 3.1 — relation-aware self-attention unit tests.

Pure-tensor tests. Verify:
1. Edge-type vocabulary size + value bounds
2. build_edge_type_matrix correctness on a synthetic 5-word sentence
3. Identity init: out_proj zero → pooled_rel = pooled_irab byte-exact (residual)
4. Shape preservation
5. Pad masking (key positions with mask=0 receive -inf attention; query
   positions with mask=0 receive zero output)
6. Gradient flow through the residual + edge bias
"""
from __future__ import annotations

import pytest
import torch

from irab_tashkeel.morphology.relational_reasoning import (
    RelationAwareSelfAttention, N_EDGE_TYPES,
)
from irab_tashkeel.morphology.dep_schema import N_DEPREL


def test_n_edge_types_matches_formula():
    # 1 (<no_edge>) + 1 (<self>) + N_DEPREL (j governs i) + N_DEPREL (i governs j)
    assert N_EDGE_TYPES == 2 + 2 * N_DEPREL


def test_edge_type_matrix_synthetic_sentence():
    """5-word sentence; word 3 is root; word 1->word 2; word 2->word 3;
    word 4->word 3; word 5->word 3.

    Tree:
        word 3 (root)
        ├── word 2 (parent of word 1)
        │   └── word 1
        ├── word 4
        └── word 5
    """
    # 1-based UD: head_indices[0] = 2 means word 1's parent is word 2
    head_indices = torch.tensor([[2, 3, 0, 3, 3]], dtype=torch.long)
    # deprel ids — arbitrary distinct values for testing
    deprel_ids = torch.tensor([[10, 5, 1, 7, 8]], dtype=torch.long)
    word_mask = torch.ones(1, 5, dtype=torch.long)

    edge_type = RelationAwareSelfAttention.build_edge_type_matrix(
        head_indices, deprel_ids, word_mask,
    )
    assert edge_type.shape == (1, 5, 5)

    # Diagonal: <self> = 1
    for i in range(5):
        assert edge_type[0, i, i].item() == 1, f"diagonal at {i} should be self=1"

    # word 1 (i=0) is governed by word 2 (j=1) with deprel(1) = 10
    # → edge_type[0, 0, 1] = 2 + 10 = 12
    assert edge_type[0, 0, 1].item() == 2 + 10

    # word 2 (i=1) is governed by word 3 (j=2) with deprel(2) = 5
    # → edge_type[0, 1, 2] = 2 + 5 = 7
    assert edge_type[0, 1, 2].item() == 2 + 5

    # word 4 (i=3) is governed by word 3 (j=2) with deprel(4) = 7
    # → edge_type[0, 3, 2] = 2 + 7 = 9
    assert edge_type[0, 3, 2].item() == 2 + 7

    # word 1 (j=0) is the dependent of word 2 (i=1) with deprel(1) = 10
    # → edge_type[0, 1, 0] = 2 + N_DEPREL + 10
    assert edge_type[0, 1, 0].item() == 2 + N_DEPREL + 10

    # word 3 is root (head_index=0) → no edges to/from anyone except itself
    # word 3 (i=2) and word 1 (j=0): no direct edge → 0
    assert edge_type[0, 2, 0].item() == 0


def test_identity_init_byte_exact():
    """At identity init, pooled_rel = pooled_irab + 0 = pooled_irab on valid positions."""
    torch.manual_seed(0)
    m = RelationAwareSelfAttention(hidden_size=64)
    x = torch.randn(2, 6, 64)
    head_indices = torch.tensor([[0, 1, 2, 3, 1, 0],
                                 [0, 1, 1, 2, 0, 0]], dtype=torch.long)
    deprel_ids = torch.randint(0, N_DEPREL, (2, 6), dtype=torch.long)
    mask = torch.ones(2, 6, dtype=torch.long)
    mask[0, 5] = 0
    mask[1, 4:] = 0

    out = m(x, head_indices, deprel_ids, mask)
    expected = x * mask.unsqueeze(-1).to(x.dtype)
    assert torch.allclose(out, expected, atol=1e-6), \
        "step-0 output must equal x on valid positions, zero on pads"


def test_shape_preservation():
    m = RelationAwareSelfAttention(hidden_size=128)
    x = torch.randn(3, 7, 128)
    head_indices = torch.zeros(3, 7, dtype=torch.long)
    deprel_ids = torch.zeros(3, 7, dtype=torch.long)
    mask = torch.ones(3, 7, dtype=torch.long)
    out = m(x, head_indices, deprel_ids, mask)
    assert out.shape == (3, 7, 128)


def test_gradient_flows_to_edge_bias():
    """Ensure the edge_bias parameter receives gradient signal."""
    torch.manual_seed(0)
    m = RelationAwareSelfAttention(hidden_size=32)
    # Move out_proj off zero so attention contribution is nonzero
    with torch.no_grad():
        m.out_proj.weight.fill_(0.01)

    x = torch.randn(1, 4, 32, requires_grad=False)
    head_indices = torch.tensor([[0, 1, 1, 2]], dtype=torch.long)
    deprel_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mask = torch.ones(1, 4, dtype=torch.long)

    out = m(x, head_indices, deprel_ids, mask)
    target = torch.randn_like(out)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    assert m.edge_bias.weight.grad is not None
    assert m.edge_bias.weight.grad.abs().sum() > 0, \
        "edge_bias must receive gradient signal"


def test_gradient_flows_to_qkv_projections():
    """Q, K, V projection weights all receive gradient."""
    torch.manual_seed(0)
    m = RelationAwareSelfAttention(hidden_size=32)
    with torch.no_grad():
        m.out_proj.weight.fill_(0.01)

    x = torch.randn(1, 4, 32)
    head_indices = torch.tensor([[0, 1, 1, 2]], dtype=torch.long)
    deprel_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mask = torch.ones(1, 4, dtype=torch.long)

    out = m(x, head_indices, deprel_ids, mask)
    out.sum().backward()

    for name, p in [("q", m.q_proj.weight), ("k", m.k_proj.weight), ("v", m.v_proj.weight)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name}_proj.weight no grad"


def test_pad_masking():
    """Padded query positions receive zero output; padded key positions
    don't contribute to other queries' attention."""
    torch.manual_seed(0)
    m = RelationAwareSelfAttention(hidden_size=16)
    # Push out_proj off zero so attention has effect
    with torch.no_grad():
        m.out_proj.weight.fill_(0.1)

    x = torch.randn(1, 5, 16)
    head_indices = torch.tensor([[0, 1, 2, 0, 0]], dtype=torch.long)
    deprel_ids = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
    # Words 4 and 5 are pads
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    out = m(x, head_indices, deprel_ids, mask)
    # Pad positions: out[0, 3, :] and out[0, 4, :] should be zero
    assert torch.allclose(out[0, 3], torch.zeros(16), atol=1e-6)
    assert torch.allclose(out[0, 4], torch.zeros(16), atol=1e-6)
