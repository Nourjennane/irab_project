"""Morph-augmented structured model — Phase 1.

Subclasses :class:`irab_tashkeel.structured.model.StructuredIrabModel`
without modifying it. Adds 7 optional morphology heads (gender, number,
definiteness, person, aspect, mood, voice) and a flag-guarded forward path.

Design rules:
* If ``enable_morph_heads=False``, behaviour is byte-identical to the parent
  class — same forward, same loss, same return dict. Used for ablation.
* Each morph head is independently toggleable through ``morph_heads_enabled``
  (a set of feature names). Off-by-default heads have their loss masked to
  zero AND their logits are not even allocated.
* Per-feature loss weights are individually configurable via
  ``morph_loss_weights`` (dict feature → float). Default uniform 0.3 for all
  enabled morph features (per Phase 1 spec).
* Soft hierarchy: morph head outputs DO NOT feed back into the i'rāb heads
  in Phase 1. That conditioning lands in Phase 2.

The class also exposes a ``predict_morph()`` method returning per-head
softmax confidence + argmax, parallel to the existing ``predict()``.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..structured.dataset import IGNORE
from ..structured.model import StructuredIrabModel
from .schema import (
    MORPH_FEATURES, N_GENDER, N_NUMBER, N_DEFINITE, N_PERSON,
    N_ASPECT, N_MOOD, N_VOICE,
)


_MORPH_HEAD_SIZES: Dict[str, int] = {
    "gender": N_GENDER, "number": N_NUMBER, "definite": N_DEFINITE,
    "person": N_PERSON, "aspect": N_ASPECT, "mood": N_MOOD, "voice": N_VOICE,
}


class MorphAugmentedStructuredModel(StructuredIrabModel):
    """Rev 2 model + 7 optional morphology heads.

    Args:
        ...all parent args...
        enable_morph_heads: master switch. When False, this class is byte-
            identical to :class:`StructuredIrabModel`.
        morph_heads_enabled: which specific morph heads to instantiate.
            Default: all seven. Disabling individual heads at construction
            saves memory + parameters.
        morph_loss_weights: per-feature loss weights. Default uniform 0.3.
    """

    def __init__(
        self,
        *args,
        enable_morph_heads: bool = False,
        morph_heads_enabled: Optional[Set[str]] = None,
        morph_loss_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_morph_heads = bool(enable_morph_heads)
        # Resolve which heads are active
        if self.enable_morph_heads:
            active = set(morph_heads_enabled) if morph_heads_enabled else set(MORPH_FEATURES)
        else:
            active = set()
        self.morph_heads_enabled: Set[str] = active

        # Default uniform weight 0.3 for every active morph head
        weights = {f: 0.3 for f in MORPH_FEATURES}
        if morph_loss_weights:
            weights.update(morph_loss_weights)
        self.morph_loss_weights: Dict[str, float] = weights

        # Allocate heads only for the active set
        self.morph_heads = nn.ModuleDict()
        for f in MORPH_FEATURES:
            if f in active:
                self.morph_heads[f] = nn.Linear(self.hidden_size, _MORPH_HEAD_SIZES[f])

    # -- helper: pool encoder hidden states (delegate to parent) --
    def _encode_and_pool(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        word_starts: torch.LongTensor,
        word_ends: torch.LongTensor,
        word_mask: torch.LongTensor,
    ) -> torch.Tensor:
        from ..structured.model import _word_first_pool, _word_mean_pool
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = enc_out.last_hidden_state
        if self.pooling_strategy == "first":
            pooled = _word_first_pool(hidden, word_starts, word_mask)
        else:
            pooled = _word_mean_pool(hidden, word_starts, word_ends, word_mask)
        return self.dropout(pooled)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        word_starts: torch.LongTensor,
        word_ends: torch.LongTensor,
        word_mask: torch.LongTensor,
        case_labels: Optional[torch.LongTensor] = None,
        role_labels: Optional[torch.LongTensor] = None,
        marker_labels: Optional[torch.LongTensor] = None,
        pos_labels: Optional[torch.LongTensor] = None,
        gender_labels: Optional[torch.LongTensor] = None,
        number_labels: Optional[torch.LongTensor] = None,
        definite_labels: Optional[torch.LongTensor] = None,
        person_labels: Optional[torch.LongTensor] = None,
        aspect_labels: Optional[torch.LongTensor] = None,
        mood_labels: Optional[torch.LongTensor] = None,
        voice_labels: Optional[torch.LongTensor] = None,
        has_irab: Optional[torch.LongTensor] = None,
        has_morph: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        # If morph heads are entirely disabled, fall through to the parent's
        # forward unchanged — this is the rev 2 path.
        if not self.enable_morph_heads:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_starts=word_starts,
                word_ends=word_ends,
                word_mask=word_mask,
                case_labels=case_labels, role_labels=role_labels,
                marker_labels=marker_labels, pos_labels=pos_labels,
                return_dict=return_dict,
            )

        # Phase 1 augmented path: re-implement the parent forward to compute
        # i'rāb losses + morph losses on the same encoder pass.
        pooled = self._encode_and_pool(input_ids, attention_mask,
                                       word_starts, word_ends, word_mask)

        from ..structured.schema import N_CASE, N_ROLE, N_MARKER, N_POS

        case_logits   = self.case_head(pooled)
        role_logits   = self.role_head(pooled)
        marker_logits = self.marker_head(pooled)
        pos_logits    = self.pos_head(pooled)

        morph_logits: Dict[str, torch.Tensor] = {}
        for f in self.morph_heads_enabled:
            morph_logits[f] = self.morph_heads[f](pooled)

        out: Dict[str, torch.Tensor] = {
            "case_logits":   case_logits,
            "role_logits":   role_logits,
            "marker_logits": marker_logits,
            "pos_logits":    pos_logits,
            "word_mask":     word_mask,
        }
        for f, l in morph_logits.items():
            out[f"{f}_logits"] = l

        if case_labels is None:
            out["loss"] = None
            return out if return_dict else out

        # i'rāb losses (rev-2-identical)
        wc, wr, wm, wp = self.loss_weights
        ls = self.label_smoothing

        def _safe_ce(logits, labels, **ce_kwargs):
            """CE that returns 0.0 when every label is ignored (avoid NaN)."""
            if labels is None:
                return logits.new_zeros(())
            valid = (labels != IGNORE)
            if not valid.any():
                return logits.new_zeros(())
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE,
                **ce_kwargs,
            )

        loss_case   = _safe_ce(case_logits,   case_labels,   label_smoothing=ls)
        if self.use_crf_role and self.role_crf is not None:
            loss_role = self.role_crf(role_logits, role_labels, word_mask)
        else:
            role_w = self.role_class_weights if self._has_role_weights else None
            loss_role = _safe_ce(role_logits, role_labels, label_smoothing=ls, weight=role_w)
        loss_marker = _safe_ce(marker_logits, marker_labels, label_smoothing=ls)
        loss_pos    = _safe_ce(pos_logits,    pos_labels,    label_smoothing=ls)

        # Morph losses: each head's CE is independent; the dataset masks the
        # labels to IGNORE on examples without morph annotations.
        morph_losses: Dict[str, torch.Tensor] = {}
        morph_loss_inputs = {
            "gender": gender_labels, "number": number_labels,
            "definite": definite_labels, "person": person_labels,
            "aspect": aspect_labels, "mood": mood_labels, "voice": voice_labels,
        }
        for f in self.morph_heads_enabled:
            morph_losses[f] = _safe_ce(morph_logits[f], morph_loss_inputs[f])

        # Total loss: i'rāb part + weighted morph part
        total = (wc * loss_case + wr * loss_role
                 + wm * loss_marker + wp * loss_pos)
        for f, l in morph_losses.items():
            total = total + self.morph_loss_weights.get(f, 0.0) * l

        out["loss"] = total
        out["case_loss"]   = loss_case
        out["role_loss"]   = loss_role
        out["marker_loss"] = loss_marker
        out["pos_loss"]    = loss_pos
        for f, l in morph_losses.items():
            out[f"{f}_loss"] = l
        return out
