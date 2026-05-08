"""Phase 2 — soft morphology conditioning modules.

Three mechanisms that take per-word encoder features ``h ∈ R^{B×W×768}`` and
per-word morph signal ``m ∈ R^{B×W×M}`` (the concat of softmax probabilities
across the seven Phase 1 morph heads, M = 26) and produce conditioned
features ``h' ∈ R^{B×W×768}`` that the iʿrāb decoders consume:

* :class:`FiLMConditioning`        — multiplicative + additive gating (γ ⊙ h + β)
* :class:`AdditiveBiasConditioning` — additive only (h + b)
* :class:`ConcatEmbedConditioning` — discrete argmax → embedding → MLP

All three implement the same forward signature
``forward(h, morph_probs, word_mask) -> h'`` so the model can swap them via
config. All three identity-initialise (FiLM γ=1/β=0, additive b≈0, concat
MLP with residual) so step-0 behaviour is byte-identical to Phase 1 and
Phase 2 can only *add* signal as training proceeds.

The conditioning module never owns a loss of its own. It is trained purely
by the gradient from ``L_irab`` flowing back through it (and, when
``detached=False``, on through the morph heads).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import (
    MORPH_FEATURES, N_GENDER, N_NUMBER, N_DEFINITE,
    N_PERSON, N_ASPECT, N_MOOD, N_VOICE,
)


_MORPH_HEAD_SIZES = {
    "gender": N_GENDER, "number": N_NUMBER, "definite": N_DEFINITE,
    "person": N_PERSON, "aspect": N_ASPECT, "mood": N_MOOD, "voice": N_VOICE,
}

# Canonical concat order. Frozen — must match the order used at build time
# in MorphAugmentedStructuredModel.forward when assembling ``m``.
MORPH_CONCAT_ORDER: List[str] = list(MORPH_FEATURES)
MORPH_TOTAL_DIM: int = sum(_MORPH_HEAD_SIZES[f] for f in MORPH_CONCAT_ORDER)
assert MORPH_TOTAL_DIM == 26, f"Phase 2 expects M=26 (got {MORPH_TOTAL_DIM})"


class FiLMConditioning(nn.Module):
    """Feature-wise linear modulation: h' = γ(m) ⊙ h + β(m).

    Identity-initialised so that γ ≈ 1 and β ≈ 0 at step 0. With this init,
    h' = h regardless of m, so the iʿrāb heads see the unmodulated Phase 1
    representation initially. Gradient pressure from L_irab is what teaches
    γ, β to deviate from identity.

    Args:
        hidden_size: encoder hidden dim (e.g. 768 for AraT5v2-base).
        morph_dim:   total concat dim of morph soft probabilities (26 by default).
        detached:    if True, ``m`` is detached before entering the projections,
                     so gradients do NOT flow back through the morph heads.
    """

    def __init__(self, hidden_size: int, morph_dim: int = MORPH_TOTAL_DIM, detached: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.morph_dim = morph_dim
        self.detached = bool(detached)
        self.gamma_proj = nn.Linear(morph_dim, hidden_size, bias=True)
        self.beta_proj = nn.Linear(morph_dim, hidden_size, bias=True)
        self._identity_init()

    def _identity_init(self) -> None:
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(
        self,
        h: torch.Tensor,
        morph_probs: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        m = morph_probs.detach() if self.detached else morph_probs
        gamma = self.gamma_proj(m)
        beta = self.beta_proj(m)
        out = gamma * h + beta
        # Pad-position safety: zero-out h' on masked positions so downstream
        # heads cannot pick up garbage from initialised projections.
        mask = word_mask.unsqueeze(-1).to(out.dtype)
        return out * mask


class AdditiveBiasConditioning(nn.Module):
    """h' = h + W_b · m. Strictly weaker than FiLM (no multiplicative gate).

    Reported as a control to isolate whether the gain comes from the
    *information* in m (additive is enough) or from the *interaction* with h
    (FiLM is needed).
    """

    def __init__(self, hidden_size: int, morph_dim: int = MORPH_TOTAL_DIM, detached: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.morph_dim = morph_dim
        self.detached = bool(detached)
        self.bias_proj = nn.Linear(morph_dim, hidden_size, bias=False)
        nn.init.zeros_(self.bias_proj.weight)  # identity init: b = 0

    def forward(
        self,
        h: torch.Tensor,
        morph_probs: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        m = morph_probs.detach() if self.detached else morph_probs
        b = self.bias_proj(m)
        out = h + b
        mask = word_mask.unsqueeze(-1).to(out.dtype)
        return out * mask


class ConcatEmbedConditioning(nn.Module):
    """h' = MLP([h ; embed(argmax(m_per_head))]).

    Discrete control: takes argmax over each per-head softmax slice, looks up
    a learned embedding (one table per head), concatenates the seven
    embeddings to ``h``, and projects back to hidden_size via a residual MLP.

    Loses the soft-probability information, so reported as a control: tests
    whether the soft signal matters or whether discrete morph identity is
    enough.
    """

    def __init__(
        self,
        hidden_size: int,
        morph_dim: int = MORPH_TOTAL_DIM,
        detached: bool = False,
        embed_dim: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.morph_dim = morph_dim
        self.detached = bool(detached)
        self.embed_dim = embed_dim
        # One embedding table per head, sized to that head's K
        self.embeds = nn.ModuleList([
            nn.Embedding(_MORPH_HEAD_SIZES[f], embed_dim) for f in MORPH_CONCAT_ORDER
        ])
        # Slice boundaries within m for argmax
        self._slice_starts: List[int] = []
        s = 0
        for f in MORPH_CONCAT_ORDER:
            self._slice_starts.append(s)
            s += _MORPH_HEAD_SIZES[f]
        # MLP: hidden + 7 * embed_dim → hidden
        in_dim = hidden_size + len(MORPH_CONCAT_ORDER) * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # Residual init: zero the final layer so output ≈ h at step 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        morph_probs: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        m = morph_probs.detach() if self.detached else morph_probs
        # Argmax per slice
        embeds: List[torch.Tensor] = []
        for i, f in enumerate(MORPH_CONCAT_ORDER):
            start = self._slice_starts[i]
            end = start + _MORPH_HEAD_SIZES[f]
            slice_logits = m[..., start:end]
            arg = slice_logits.argmax(dim=-1)
            embeds.append(self.embeds[i](arg))
        z = torch.cat(embeds, dim=-1)
        x = torch.cat([h, z], dim=-1)
        delta = self.mlp(x)
        out = h + delta  # residual: at step 0 delta=0 → h' = h
        mask = word_mask.unsqueeze(-1).to(out.dtype)
        return out * mask


def build_conditioning(
    mechanism: str,
    hidden_size: int,
    morph_dim: int = MORPH_TOTAL_DIM,
    detached: bool = False,
) -> nn.Module:
    """Factory used by the trainer to instantiate by config name.

    ``mechanism`` ∈ {"film", "additive", "concat_embed"}. Anything else raises.
    The "none" case is handled by the model (no conditioning module instantiated)
    rather than by this factory.
    """
    m = mechanism.lower()
    if m == "film":
        return FiLMConditioning(hidden_size, morph_dim, detached=detached)
    if m == "additive":
        return AdditiveBiasConditioning(hidden_size, morph_dim, detached=detached)
    if m == "concat_embed":
        return ConcatEmbedConditioning(hidden_size, morph_dim, detached=detached)
    raise ValueError(
        f"Unknown conditioning mechanism: {mechanism!r}. "
        f"Expected one of: film, additive, concat_embed."
    )
