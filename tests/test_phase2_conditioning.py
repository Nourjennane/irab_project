"""Phase 2 — conditioning module unit tests.

Verifies the load-bearing design-doc invariants WITHOUT spinning up the
full 296M encoder + 50-sentence smoke loop. Each test is fast (< 1s) and
checks one of:

1. **Identity init.** FiLM γ=1, β=0; additive b=0; concat MLP final layer
   zero. At step 0, h' = h regardless of m.
2. **Gradient flow (joint).** Gradients from a downstream loss reach
   the conditioning module's parameters AND the morph signal m's
   producer (when ``detached=False``).
3. **No gradient flow (detached).** When ``detached=True``, the
   conditioning module trains but no gradient reaches m's producer.
4. **Shape invariants.** h' has the same shape as h; mask zeros out
   pad positions.
5. **Factory wiring.** ``build_conditioning("film")`` returns FiLM,
   ``"additive"`` → AdditiveBias, ``"concat_embed"`` → ConcatEmbed.
   Unknown mechanism raises.
6. **Model integration (no encoder).** MorphAugmentedStructuredModel
   with a FiLM conditioning module produces the SAME iʿrāb logits at
   step 0 as one with conditioning=None, given an identical encoder
   forward.

Tests #1-#5 are pure-tensor and run in CPU. Test #6 requires the
encoder load (skipped if HF_HUB_OFFLINE blocks download AND no local
cache).
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from irab_tashkeel.morphology.conditioning import (
    AdditiveBiasConditioning,
    ConcatEmbedConditioning,
    FiLMConditioning,
    MORPH_TOTAL_DIM,
    build_conditioning,
)


def _make_h(B=2, W=5, D=768, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, W, D, generator=g)


def _make_m(B=2, W=5, seed=1):
    """Per-word soft conditioning signal: 7 head softmaxes concatenated."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(B, W, MORPH_TOTAL_DIM, generator=g)
    head_sizes = [3, 4, 4, 4, 3, 5, 3]
    starts = [0]
    for s in head_sizes[:-1]:
        starts.append(starts[-1] + s)
    out = torch.empty_like(raw)
    for start, sz in zip(starts, head_sizes):
        out[..., start:start + sz] = torch.softmax(raw[..., start:start + sz], dim=-1)
    return out


def _make_mask(B=2, W=5):
    mask = torch.ones(B, W, dtype=torch.long)
    mask[:, -1] = 0  # last word always padded — pad-zeroing must zero h'
    return mask


# ---------------------------------------------------------------------------
# 1. Identity init
# ---------------------------------------------------------------------------

def test_film_identity_init():
    cond = FiLMConditioning(hidden_size=768)
    h = _make_h()
    m = _make_m()
    mask = _make_mask()
    h_prime = cond(h, m, mask)
    expected = h * mask.unsqueeze(-1).to(h.dtype)
    assert torch.allclose(h_prime, expected, atol=1e-6), (
        "FiLM at step 0 must reproduce h on valid positions and zero on pads"
    )


def test_additive_identity_init():
    cond = AdditiveBiasConditioning(hidden_size=768)
    h = _make_h(seed=2)
    m = _make_m(seed=3)
    mask = _make_mask()
    h_prime = cond(h, m, mask)
    expected = h * mask.unsqueeze(-1).to(h.dtype)
    assert torch.allclose(h_prime, expected, atol=1e-6)


def test_concat_embed_identity_init():
    cond = ConcatEmbedConditioning(hidden_size=768, embed_dim=16)
    h = _make_h(seed=4)
    m = _make_m(seed=5)
    mask = _make_mask()
    h_prime = cond(h, m, mask)
    expected = h * mask.unsqueeze(-1).to(h.dtype)
    assert torch.allclose(h_prime, expected, atol=1e-6), (
        "ConcatEmbed residual init must give h' = h on valid positions"
    )


# ---------------------------------------------------------------------------
# 2-3. Gradient flow (joint vs detached)
# ---------------------------------------------------------------------------

class _MorphProducer(nn.Module):
    """Stub for whatever produces morph soft probs upstream."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, MORPH_TOTAL_DIM)

    def forward(self, dummy):
        logits = self.proj(dummy)
        # Softmax per slice — same as morph_model does at runtime.
        out = torch.empty_like(logits)
        idx = 0
        for sz in [3, 4, 4, 4, 3, 5, 3]:
            out[..., idx:idx + sz] = torch.softmax(logits[..., idx:idx + sz], dim=-1)
            idx += sz
        return out


def _train_film_one_step_off_identity(cond: FiLMConditioning) -> None:
    """At identity init, ∂h'/∂m = W_γ ⊙ h + W_β = 0 for both — so morph
    gradient is trivially zero whether detached or not. The joint-vs-detached
    distinction is observable only AFTER FiLM has moved off the identity.

    This helper takes one SGD step that drives ``W_γ.weight`` and ``W_β.weight``
    away from zero; subsequent ``backward()`` calls then test the
    detached/joint distinction meaningfully.
    """
    h = torch.randn(2, 5, 768)
    m = torch.randn(2, 5, MORPH_TOTAL_DIM)
    mask = torch.ones(2, 5, dtype=torch.long)
    # Asymmetric loss → non-zero gradient on W_γ.weight and W_β.weight
    target = torch.randn_like(h)
    loss = ((cond(h, m, mask) - target) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        for p in cond.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad
                p.grad = None


def test_film_joint_gradient_flow():
    """Joint training: gradients reach the morph producer through FiLM
    (after FiLM has moved off identity init)."""
    cond = FiLMConditioning(hidden_size=768, detached=False)
    _train_film_one_step_off_identity(cond)

    producer = _MorphProducer()
    dummy = torch.randn(2, 5, 8)
    h = torch.randn(2, 5, 768)
    mask = torch.ones(2, 5, dtype=torch.long)
    m = producer(dummy)
    h_prime = cond(h, m, mask)
    h_prime.sum().backward()

    assert cond.gamma_proj.weight.grad is not None
    assert cond.gamma_proj.weight.grad.abs().sum() > 0
    # The actual joint-vs-detached test:
    assert producer.proj.weight.grad is not None
    assert producer.proj.weight.grad.abs().sum() > 0, (
        "joint mode must propagate gradient through morph producer"
    )


def test_film_detached_blocks_morph_gradient():
    """Detached: conditioning trains but no gradient reaches the morph producer
    (post-identity-init, where the joint case would receive gradient)."""
    cond = FiLMConditioning(hidden_size=768, detached=True)
    _train_film_one_step_off_identity(cond)

    producer = _MorphProducer()
    dummy = torch.randn(2, 5, 8)
    h = torch.randn(2, 5, 768)
    mask = torch.ones(2, 5, dtype=torch.long)
    m = producer(dummy)
    h_prime = cond(h, m, mask)
    h_prime.sum().backward()

    assert cond.gamma_proj.weight.grad is not None
    assert cond.gamma_proj.weight.grad.abs().sum() > 0
    # Producer must NOT receive gradient
    assert producer.proj.weight.grad is None or producer.proj.weight.grad.abs().sum() == 0, (
        "detached mode must block gradient flow to morph producer"
    )


# ---------------------------------------------------------------------------
# 4. Shape invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mechanism", ["film", "additive", "concat_embed"])
def test_shape_preservation(mechanism):
    cond = build_conditioning(mechanism, hidden_size=768)
    h = _make_h(B=3, W=7)
    m = _make_m(B=3, W=7, seed=42)
    mask = torch.ones(3, 7, dtype=torch.long)
    h_prime = cond(h, m, mask)
    assert h_prime.shape == h.shape


@pytest.mark.parametrize("mechanism", ["film", "additive", "concat_embed"])
def test_pad_zeroing(mechanism):
    cond = build_conditioning(mechanism, hidden_size=768)
    # Train one step away from identity init so the projections have non-zero output
    opt = torch.optim.SGD(cond.parameters(), lr=0.1)
    h = _make_h()
    m = _make_m()
    mask = torch.ones(2, 5, dtype=torch.long)
    h_prime = cond(h, m, mask)
    h_prime.sum().backward()
    opt.step()
    # Now run with mask=0 for last word and verify output is zero there
    mask_padded = mask.clone()
    mask_padded[:, -1] = 0
    h_prime2 = cond(h, m, mask_padded)
    assert torch.allclose(h_prime2[:, -1, :], torch.zeros_like(h_prime2[:, -1, :]))


# ---------------------------------------------------------------------------
# 5. Factory wiring
# ---------------------------------------------------------------------------

def test_build_conditioning_returns_correct_class():
    assert isinstance(build_conditioning("film", 768), FiLMConditioning)
    assert isinstance(build_conditioning("additive", 768), AdditiveBiasConditioning)
    assert isinstance(build_conditioning("concat_embed", 768), ConcatEmbedConditioning)
    # Case-insensitive
    assert isinstance(build_conditioning("FiLM", 768), FiLMConditioning)


def test_build_conditioning_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown conditioning mechanism"):
        build_conditioning("transformer_block", 768)


# ---------------------------------------------------------------------------
# 6. Model integration — at-init byte-identity of iʿrāb logits.
# ---------------------------------------------------------------------------
#  Skipped when the encoder is not cached locally (avoids hitting HF in tests).

def _has_encoder_cached() -> bool:
    cache_root = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    target = "models--UBC-NLP--AraT5v2-base-1024"
    for root, dirs, _ in os.walk(cache_root):
        if target in dirs:
            return True
    return False


def _morph_model_importable() -> bool:
    try:
        from irab_tashkeel.morphology.morph_model import MorphAugmentedStructuredModel  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not (_has_encoder_cached() and _morph_model_importable()),
    reason="AraT5v2-base encoder not cached or transformers stack unimportable",
)
def test_model_integration_step0_byte_identity():
    """At-init, MorphAugmentedStructuredModel(conditioning=film) produces the
    same iʿrāb logits as MorphAugmentedStructuredModel(conditioning=None)."""
    from irab_tashkeel.morphology.morph_model import MorphAugmentedStructuredModel

    try:
        torch.manual_seed(0)
        m_phase1 = MorphAugmentedStructuredModel(
            encoder_name="UBC-NLP/AraT5v2-base-1024",
            enable_morph_heads=True,
            conditioning_mechanism=None,
        )
        torch.manual_seed(0)
        m_phase2 = MorphAugmentedStructuredModel(
            encoder_name="UBC-NLP/AraT5v2-base-1024",
            enable_morph_heads=True,
            conditioning_mechanism="film",
        )
    except (ImportError, OSError) as e:
        pytest.skip(f"encoder load failed in this env (likely transitive TF dep): {e}")
    # Copy iʿrāb head + morph head + encoder weights so the only difference
    # is the conditioning module (which is identity-init).
    m_phase2.load_state_dict(m_phase1.state_dict(), strict=False)
    m_phase1.eval(); m_phase2.eval()

    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
    attention_mask = torch.ones_like(input_ids)
    word_starts = torch.tensor([[0, 3, 6]])
    word_ends = torch.tensor([[3, 6, 9]])
    word_mask = torch.ones((1, 3), dtype=torch.long)

    with torch.no_grad():
        out1 = m_phase1(
            input_ids=input_ids, attention_mask=attention_mask,
            word_starts=word_starts, word_ends=word_ends, word_mask=word_mask,
        )
        out2 = m_phase2(
            input_ids=input_ids, attention_mask=attention_mask,
            word_starts=word_starts, word_ends=word_ends, word_mask=word_mask,
        )

    # Pad position (none here, all word_mask=1) zeroing is trivial; the test
    # is the valid-position iʿrāb logits.
    for k in ("case_logits", "role_logits", "marker_logits"):
        assert torch.allclose(out1[k], out2[k], atol=1e-5), (
            f"{k} differs at step 0 between Phase 1 and Phase 2 (FiLM identity init)"
        )
