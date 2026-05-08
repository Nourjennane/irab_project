"""Phase 3 — dep-aware structured model.

Subclasses :class:`MorphAugmentedStructuredModel` (Phase 1) and adds an
optional dep-feature input augmentation. Per-word UD dep features
(DEPREL one-hot embedding + HEAD direction + HEAD distance bucket +
governor's POS embedding) are concatenated to the encoder pooled feature
``h`` and projected back to 768 dim before the iʿrāb decoders consume it.

Critical design choice (vs Phase 2): dep features are STATIC INPUTS
computed offline by Stanza/UD parser, NOT learned heads with joint
gradient flow. The dep-feature embedding tables ARE learnable (small,
~30K params total), but the *source* of the dep signal is offline
parsing — not a head we train. The Phase 2 joint-training-dynamics
issue (morph head representation drifts under joint training, iʿrāb
heads chase the moving target) does not apply.

Identity initialisation: when dep features are present but ``dep_proj``'s
weight is initialised so the projection of the concat is approximately
``h`` (zero on the dep slice, identity on the h slice), the iʿrāb heads
at step 0 see the same input as Phase 1. Gradient pressure from
``L_irab`` is what teaches the dep slice to contribute.
"""
from __future__ import annotations

from typing import Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import MORPH_CONCAT_ORDER
from .dep_schema import (
    DEPREL_EMB_DIM, GOV_POS_EMB_DIM, HEAD_DIR_EMB_DIM, HEAD_DIST_EMB_DIM,
    DEP_FEATURE_DIM_TOTAL, N_DEPREL, N_HEAD_DIR, N_HEAD_DIST,
)
from .relational_reasoning import RelationAwareSelfAttention
from .morph_model import MorphAugmentedStructuredModel
from .schema import MORPH_FEATURES
from ..structured.dataset import IGNORE
from ..structured.schema import N_POS


class DepFeatureEncoder(nn.Module):
    """Per-word dep feature embedding stack.

    Takes per-word integer ids for (deprel, head_dir, head_dist, gov_pos)
    and emits a concatenated embedding of size :data:`DEP_FEATURE_DIM_TOTAL`.
    """

    def __init__(self):
        super().__init__()
        self.deprel_embed = nn.Embedding(N_DEPREL, DEPREL_EMB_DIM)
        self.head_dir_embed = nn.Embedding(N_HEAD_DIR, HEAD_DIR_EMB_DIM)
        self.head_dist_embed = nn.Embedding(N_HEAD_DIST, HEAD_DIST_EMB_DIM)
        self.gov_pos_embed = nn.Embedding(N_POS, GOV_POS_EMB_DIM)
        # Initialise embeddings near-zero so identity init of dep_proj
        # (zero on the dep slice) genuinely starts at no-op. Final
        # near-zero contribution after concat → projection.
        for emb in (self.deprel_embed, self.head_dir_embed,
                    self.head_dist_embed, self.gov_pos_embed):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        deprel_ids: torch.LongTensor,
        head_dir_ids: torch.LongTensor,
        head_dist_ids: torch.LongTensor,
        gov_pos_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Returns (B, W, DEP_FEATURE_DIM_TOTAL)."""
        a = self.deprel_embed(deprel_ids)
        b = self.head_dir_embed(head_dir_ids)
        c = self.head_dist_embed(head_dist_ids)
        d = self.gov_pos_embed(gov_pos_ids)
        return torch.cat([a, b, c, d], dim=-1)


class DepAwareStructuredModel(MorphAugmentedStructuredModel):
    """Rev 2 + Phase 1 morph heads + Phase 3 optional dep-feature input.

    Args extending parent:
        enable_dep_features: master switch. When False, behaviour is
            byte-identical to :class:`MorphAugmentedStructuredModel`.
            When True the model expects per-word dep ids in the batch.
    """

    def __init__(
        self,
        *args,
        enable_dep_features: bool = False,
        # Phase 5: hierarchical case decoder (output-side conditioning).
        enable_case_hierarchy: bool = False,
        case_hierarchy_detached: bool = False,
        # Phase 6: hierarchical marker decoder (output-side; conditions marker on case + role).
        enable_marker_hierarchy: bool = False,
        marker_hierarchy_detached: bool = False,
        # Phase 3.1: relational reasoning expansion. None = Phase 3-A baseline; "attn" = §3.1.
        enable_relational_reasoning: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_dep_features = bool(enable_dep_features)
        if self.enable_dep_features:
            if self.conditioning is not None:
                # Phase 3 explicitly avoids stacking dep features on top of
                # a Phase 2 conditioning module — that would re-introduce
                # the joint-dynamics issue Phase 3 was designed to sidestep.
                raise ValueError(
                    "enable_dep_features=True is incompatible with "
                    "conditioning_mechanism set. Phase 3 stays on the "
                    "Phase 1 baseline (rev 2 + 7 morph heads), no "
                    "conditioning module."
                )
            self.dep_feature_encoder = DepFeatureEncoder()
            # h ⊕ dep_features  →  hidden_size
            in_dim = self.hidden_size + DEP_FEATURE_DIM_TOTAL
            self.dep_proj = nn.Linear(in_dim, self.hidden_size, bias=True)
            self._identity_init_dep_proj()

        # Phase 5: optional hierarchical case decoder. role_softmax (N_role)
        # → small linear → case_bias (N_case). Initialised to zero so step 0
        # is byte-equivalent to the no-hierarchy path.
        self.enable_case_hierarchy = bool(enable_case_hierarchy)
        self.case_hierarchy_detached = bool(case_hierarchy_detached)
        if self.enable_case_hierarchy:
            if not self.enable_dep_features:
                raise ValueError(
                    "enable_case_hierarchy=True requires enable_dep_features=True. "
                    "Phase 5 is designed to layer on top of Phase 3-A (the current "
                    "production checkpoint), not on Phase 1 alone."
                )
            self.role_to_case_bias = nn.Linear(
                self._n_role, self.case_head.out_features, bias=False,
            )
            with torch.no_grad():
                self.role_to_case_bias.weight.zero_()

        # Phase 3.1: relational reasoning. Operates between dep_proj and the iʿrāb heads.
        self.enable_relational_reasoning = enable_relational_reasoning
        if self.enable_relational_reasoning:
            if not self.enable_dep_features:
                raise ValueError(
                    "enable_relational_reasoning requires enable_dep_features=True. "
                    "Phase 3.1 layers on Phase 3-A."
                )
            if self.enable_relational_reasoning == "attn":
                self.relational_layer = RelationAwareSelfAttention(
                    hidden_size=self.hidden_size,
                )
            else:
                raise ValueError(
                    f"unknown enable_relational_reasoning={self.enable_relational_reasoning!r}; "
                    f"expected 'attn' or None"
                )

        # Phase 6: optional hierarchical marker decoder.
        # softmax([case_logits; role_logits]) → small linear → marker_bias (N_marker).
        self.enable_marker_hierarchy = bool(enable_marker_hierarchy)
        self.marker_hierarchy_detached = bool(marker_hierarchy_detached)
        if self.enable_marker_hierarchy:
            if not self.enable_dep_features:
                raise ValueError(
                    "enable_marker_hierarchy=True requires enable_dep_features=True."
                )
            n_case = self.case_head.out_features
            self.case_role_to_marker_bias = nn.Linear(
                n_case + self._n_role, self.marker_head.out_features, bias=False,
            )
            with torch.no_grad():
                self.case_role_to_marker_bias.weight.zero_()

    def _identity_init_dep_proj(self) -> None:
        """Initialise dep_proj so that, with random near-zero dep embeddings,
        ``dep_proj([h ; ε]) ≈ h`` at step 0.

        Layout: ``dep_proj.weight`` has shape ``(hidden_size,
        hidden_size + DEP_FEATURE_DIM_TOTAL)``. The first ``hidden_size``
        columns operate on ``h`` — set to identity. The trailing
        ``DEP_FEATURE_DIM_TOTAL`` columns operate on the dep features —
        set to zero. Bias = 0. Step 0 output is exactly ``h``.
        """
        with torch.no_grad():
            self.dep_proj.weight.zero_()
            self.dep_proj.bias.zero_()
            for i in range(self.hidden_size):
                self.dep_proj.weight[i, i] = 1.0

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
        # Phase 3 dep features
        deprel_ids: Optional[torch.LongTensor] = None,
        head_dir_ids: Optional[torch.LongTensor] = None,
        head_dist_ids: Optional[torch.LongTensor] = None,
        gov_pos_ids: Optional[torch.LongTensor] = None,
        # Phase 3.1: raw HEAD index per word (1-based UD; 0 = root).
        # Optional — only needed when enable_relational_reasoning is set.
        head_indices: Optional[torch.LongTensor] = None,
        has_dep: Optional[torch.LongTensor] = None,
        has_irab: Optional[torch.LongTensor] = None,
        has_morph: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        # When dep features are disabled OR not provided in this batch,
        # fall through to the parent's forward unchanged.
        dep_provided = (
            self.enable_dep_features
            and deprel_ids is not None
            and head_dir_ids is not None
            and head_dist_ids is not None
            and gov_pos_ids is not None
        )
        if not self.enable_dep_features:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask,
                word_starts=word_starts, word_ends=word_ends, word_mask=word_mask,
                case_labels=case_labels, role_labels=role_labels,
                marker_labels=marker_labels, pos_labels=pos_labels,
                gender_labels=gender_labels, number_labels=number_labels,
                definite_labels=definite_labels, person_labels=person_labels,
                aspect_labels=aspect_labels, mood_labels=mood_labels,
                voice_labels=voice_labels,
                has_irab=has_irab, has_morph=has_morph,
                return_dict=return_dict,
            )

        # Phase 3 augmented path: re-implement the parent forward, splicing in
        # the dep-feature input augmentation for the iʿrāb heads.
        pooled = self._encode_and_pool(input_ids, attention_mask,
                                       word_starts, word_ends, word_mask)

        # Compute morph logits FIRST (Phase 1 still active)
        morph_logits: Dict[str, torch.Tensor] = {}
        for f in self.morph_heads_enabled:
            morph_logits[f] = self.morph_heads[f](pooled)

        # Build dep-augmented features. CRITICAL: when enable_dep_features=True,
        # we ALWAYS run dep_proj (even if no dep tensors are supplied), so the
        # iʿrāb heads see the same transformation pipeline at inference as they
        # did during training on has_dep=False examples. Without this, the
        # inference path skips dep_proj entirely and the iʿrāb heads operate
        # out-of-distribution.
        B, W = pooled.size(0), pooled.size(1)
        if dep_provided:
            dep_emb = self.dep_feature_encoder(
                deprel_ids=deprel_ids, head_dir_ids=head_dir_ids,
                head_dist_ids=head_dist_ids, gov_pos_ids=gov_pos_ids,
            )
            if has_dep is not None:
                # has_dep is (B,); broadcast to (B, 1, 1) so it scales (B, W, M_dep)
                gate = has_dep.view(-1, 1, 1).to(dep_emb.dtype)
                dep_emb = dep_emb * gate
        else:
            # No dep tensors supplied → use zeros. Matches the has_dep=False
            # training path (where dep_emb is masked to zero and dep_proj
            # still runs).
            dep_emb = pooled.new_zeros(B, W, self.dep_feature_encoder.deprel_embed.embedding_dim
                                       + self.dep_feature_encoder.head_dir_embed.embedding_dim
                                       + self.dep_feature_encoder.head_dist_embed.embedding_dim
                                       + self.dep_feature_encoder.gov_pos_embed.embedding_dim)
        h_aug = torch.cat([pooled, dep_emb], dim=-1)
        pooled_irab = self.dep_proj(h_aug)

        # Phase 3.1: relational reasoning over the dep tree.
        # The relational layer reads the per-sentence head_indices + deprel_ids
        # to compute per-edge-type bias on attention; output is residual-summed
        # to pooled_irab (zero-init out_proj keeps step 0 byte-equivalent).
        if (self.enable_relational_reasoning is not None
                and dep_provided
                and head_indices is not None):
            pooled_irab = self.relational_layer(
                x=pooled_irab,
                head_indices=head_indices,
                deprel_ids=deprel_ids,
                word_mask=word_mask,
            )

        # Compute role first so the (optional) hierarchical case decoder
        # can consume role_softmax. Phase 5 layered on top of Phase 3.
        role_logits   = self.role_head(pooled_irab)
        marker_logits = self.marker_head(pooled_irab)
        # POS stays unconditioned — it is encoder-level (parallel auxiliary), not iʿrāb-level.
        pos_logits    = self.pos_head(pooled)

        case_logits = self.case_head(pooled_irab)
        if self.enable_case_hierarchy:
            role_softmax = F.softmax(role_logits, dim=-1)
            if self.case_hierarchy_detached:
                role_softmax = role_softmax.detach()
            case_logits = case_logits + self.role_to_case_bias(role_softmax)

        # Phase 6: marker conditioned on case + role
        if self.enable_marker_hierarchy:
            case_softmax = F.softmax(case_logits, dim=-1)
            role_softmax_m = F.softmax(role_logits, dim=-1)
            if self.marker_hierarchy_detached:
                case_softmax = case_softmax.detach()
                role_softmax_m = role_softmax_m.detach()
            cr = torch.cat([case_softmax, role_softmax_m], dim=-1)
            marker_logits = marker_logits + self.case_role_to_marker_bias(cr)

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

        # i'rāb losses (rev-2-identical) — copied from MorphAugmentedStructuredModel
        wc, wr, wm, wp = self.loss_weights
        ls = self.label_smoothing

        def _safe_ce(logits, labels, **ce_kwargs):
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

        loss_case = _safe_ce(case_logits, case_labels, label_smoothing=ls)
        if self.use_crf_role and self.role_crf is not None:
            loss_role = self.role_crf(role_logits, role_labels, word_mask)
        else:
            role_w = self.role_class_weights if self._has_role_weights else None
            loss_role = _safe_ce(role_logits, role_labels, label_smoothing=ls, weight=role_w)
        loss_marker = _safe_ce(marker_logits, marker_labels, label_smoothing=ls)
        loss_pos    = _safe_ce(pos_logits,    pos_labels,    label_smoothing=ls)

        morph_losses: Dict[str, torch.Tensor] = {}
        morph_loss_inputs = {
            "gender": gender_labels, "number": number_labels,
            "definite": definite_labels, "person": person_labels,
            "aspect": aspect_labels, "mood": mood_labels, "voice": voice_labels,
        }
        for f in self.morph_heads_enabled:
            morph_losses[f] = _safe_ce(morph_logits[f], morph_loss_inputs[f])

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
