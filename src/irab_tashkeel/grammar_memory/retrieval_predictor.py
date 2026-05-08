"""Phase R — RetrievalAugmentedPredictor.

Wraps :class:`StructuredPredictor` (Phase 3-A) and adds inference-time
retrieval-bias on the iʿrāb decoder logits. The Phase 3-A checkpoint
is unchanged; retrieval is purely an inference wrapper.

Pipeline per test sentence:
1. Phase 3-A forward → raw (case_logits, role_logits, marker_logits)
2. Initial argmax for construction detection (uses predicted role labels
   to detect iḍāfa, in addition to surface-form particles)
3. Construction detection (surface + predicted-role)
4. For each detected construction span:
   - Build query signature
   - Retrieve top-k analogues from grammar memory
   - Aggregate per-word soft prior over (case, role, marker)
   - Apply additive logit bias: logits += λ · log(prior + ε)
5. Final argmax → SentenceIrab + RetrievalTrace
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..inference.structured_predictor import (
    StructuredPredictor, StructuredPredictorConfig,
)
from ..structured.schema import (
    CASE_LABELS, MARKER_LABELS, POS_LABELS, ROLE_LABELS,
    CASE_TO_ID, MARKER_TO_ID, ROLE_TO_ID, ID_TO_CASE, ID_TO_MARKER, ID_TO_ROLE,
    ARABIC_ROLE_FORMS,
)
from ..structured.word_irab import WordIrab, SentenceIrab
from ..inference.template_renderer import render_word
from .memory import GrammarMemory, RetrievalHit
from .signature import (
    ALL_FAMILIES, ConstructionInstance, build_signature,
    detect_constructions_in_record,
)


@dataclass
class RetrievalSpanTrace:
    """Trace data for one construction span."""
    span: Tuple[int, int]
    construction: str
    particle_group: str
    particle_surface: str
    n_hits: int
    top_hits: List[Dict] = field(default_factory=list)
    bias_applied: bool = False


@dataclass
class RetrievalTrace:
    """Per-sentence retrieval explanation hook."""
    span_traces: List[RetrievalSpanTrace] = field(default_factory=list)
    total_constructions_detected: int = 0
    total_hits: int = 0


def _initial_pred_to_record(words: List[str], role_pred: np.ndarray,
                             id_to_role: Dict[int, str]) -> Dict:
    """Build a records-like dict using INITIAL predicted roles, suitable
    for ``detect_constructions_in_record`` (which expects per-word
    'role' fields to detect iḍāfa)."""
    items = []
    for i, w in enumerate(words):
        items.append({
            "word": w,
            "role": id_to_role.get(int(role_pred[i]), ""),
        })
    return {"sentence": " ".join(words), "items": items, "source": "_query"}


def _aggregate_priors(
    span_len: int,
    hits: List[RetrievalHit],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-word soft prior over (case, role, marker) from hits.

    For each within-span word position i (0..span_len-1), look at the
    corresponding word in each hit's items list. Weight by hit.score.
    """
    n_case = len(CASE_LABELS)
    n_role = len(ROLE_LABELS)
    n_marker = len(MARKER_LABELS)

    prior_case = np.full((span_len, n_case), 1e-6, dtype=np.float32)
    prior_role = np.full((span_len, n_role), 1e-6, dtype=np.float32)
    prior_marker = np.full((span_len, n_marker), 1e-6, dtype=np.float32)

    for h in hits:
        score = max(h.score, 0.0)   # clip negative cosines
        for i, item in enumerate(h.instance.items):
            if i >= span_len:
                break
            ci = CASE_TO_ID.get(item.get("case", ""), -1)
            ri = ROLE_TO_ID.get(item.get("role", ""), -1)
            mi = MARKER_TO_ID.get(item.get("marker", ""), -1)
            if ci >= 0:
                prior_case[i, ci] += score
            if ri >= 0:
                prior_role[i, ri] += score
            if mi >= 0:
                prior_marker[i, mi] += score

    # L1-normalise per-row (each row a soft probability distribution)
    prior_case   = prior_case   / prior_case.sum(axis=-1, keepdims=True)
    prior_role   = prior_role   / prior_role.sum(axis=-1, keepdims=True)
    prior_marker = prior_marker / prior_marker.sum(axis=-1, keepdims=True)
    return prior_case, prior_role, prior_marker


class RetrievalAugmentedPredictor:
    """Wraps StructuredPredictor with inference-time retrieval bias.

    Args:
        base_predictor: a constructed StructuredPredictor (Phase 3-A).
        memory: a loaded GrammarMemory.
        lambda_: bias multiplier on log(prior). λ=0 → byte-equivalent
            to base_predictor.
        k: top-k retrieval per construction span.
    """

    def __init__(
        self,
        base_predictor: StructuredPredictor,
        memory: GrammarMemory,
        lambda_: float = 0.3,
        k: int = 5,
        retrieval_alpha: float = 0.3,
        require_particle_group: bool = True,
    ):
        self.base = base_predictor
        self.memory = memory
        self.lambda_ = float(lambda_)
        self.k = int(k)
        self.alpha = float(retrieval_alpha)
        self.require_particle_group = bool(require_particle_group)

    @torch.no_grad()
    def predict_sentence(
        self, sentence: str,
    ) -> Tuple[SentenceIrab, RetrievalTrace]:
        """Predict iʿrāb for ``sentence`` with retrieval-augmented bias."""
        enc = self.base._encode_sentence(sentence)
        if enc is None:
            return SentenceIrab(sentence=sentence, items=[]), RetrievalTrace()

        model = self.base.model
        # 1. Phase 3-A forward + extract per-word pooled_irab
        from irab_tashkeel.structured.model import _word_first_pool
        enc_out = model.encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        hidden = enc_out.last_hidden_state
        pooled = _word_first_pool(hidden, enc["word_starts"], enc["word_mask"])
        # Apply dep_proj path with zero dep_emb (matches Phase 3 inference fix)
        if getattr(model, "enable_dep_features", False):
            B, W = pooled.shape[:2]
            dep_emb = pooled.new_zeros(
                B, W,
                model.dep_feature_encoder.deprel_embed.embedding_dim
                + model.dep_feature_encoder.head_dir_embed.embedding_dim
                + model.dep_feature_encoder.head_dist_embed.embedding_dim
                + model.dep_feature_encoder.gov_pos_embed.embedding_dim,
            )
            h_aug = torch.cat([pooled, dep_emb], dim=-1)
            pooled_irab = model.dep_proj(h_aug)
        else:
            pooled_irab = pooled

        # Apply iʿrāb heads to get raw logits
        case_logits = model.case_head(pooled_irab)
        role_logits = model.role_head(pooled_irab)
        marker_logits = model.marker_head(pooled_irab)
        pos_logits = model.pos_head(pooled)

        # 2. Initial argmax (for construction detection; iḍāfa needs predicted role)
        role_probs = F.softmax(role_logits, dim=-1)
        role_conf, role_pred = role_probs[0].max(dim=-1)
        words = enc["words"]

        # 3. Detect constructions using surface (particles) + initial roles (iḍāfa)
        id_to_role = self.base.id_to_role
        record_for_detection = _initial_pred_to_record(
            words, role_pred.cpu().numpy(), id_to_role,
        )
        constructions = detect_constructions_in_record(record_for_detection)

        trace = RetrievalTrace(total_constructions_detected=len(constructions))

        # 4. For each construction: retrieve + apply bias
        for span_desc in constructions:
            start, end = span_desc["span"]
            start = max(0, min(start, len(words)))
            end = max(start, min(end, len(words)))
            if end <= start:
                continue
            span_emb = pooled_irab[0, start:end].mean(dim=0).cpu().numpy()
            query = build_signature(record_for_detection, span_desc, sentence_idx=-1)
            hits = self.memory.retrieve(
                query=query,
                query_embedding=span_emb,
                k=self.k,
                alpha=self.alpha,
                require_particle_group=self.require_particle_group,
            )

            span_trace = RetrievalSpanTrace(
                span=(start, end),
                construction=query.construction,
                particle_group=query.particle_group,
                particle_surface=query.particle_surface,
                n_hits=len(hits),
                top_hits=[{
                    "instance_id": h.instance.instance_id,
                    "sentence": h.instance.sentence,
                    "score": round(h.score, 4),
                    "cosine": round(h.cosine, 4),
                    "sym_overlap": round(h.sym_overlap, 4),
                } for h in hits[:3]],
                bias_applied=False,
            )

            if hits and self.lambda_ > 0:
                span_len = end - start
                prior_case, prior_role, prior_marker = _aggregate_priors(span_len, hits)
                # Apply additive bias: logits[start:end, :] += λ * log(prior + ε)
                eps = 1e-6
                bias_case = self.lambda_ * np.log(prior_case + eps)
                bias_role = self.lambda_ * np.log(prior_role + eps)
                bias_marker = self.lambda_ * np.log(prior_marker + eps)
                # In-place additive bias (logits is (1, W, n_class))
                case_logits[0, start:end] += torch.tensor(
                    bias_case, dtype=case_logits.dtype, device=case_logits.device,
                )
                role_logits[0, start:end] += torch.tensor(
                    bias_role, dtype=role_logits.dtype, device=role_logits.device,
                )
                marker_logits[0, start:end] += torch.tensor(
                    bias_marker, dtype=marker_logits.dtype, device=marker_logits.device,
                )
                span_trace.bias_applied = True
                trace.total_hits += len(hits)
            trace.span_traces.append(span_trace)

        # 5. Final argmax + build SentenceIrab
        case_probs = F.softmax(case_logits, dim=-1)
        marker_probs = F.softmax(marker_logits, dim=-1)
        pos_probs = F.softmax(pos_logits, dim=-1)
        # role: re-argmax after bias
        role_probs_final = F.softmax(role_logits, dim=-1)
        case_conf, case_pred = case_probs[0].max(dim=-1)
        role_conf_final, role_pred_final = role_probs_final[0].max(dim=-1)
        marker_conf, marker_pred = marker_probs[0].max(dim=-1)
        pos_conf, pos_pred = pos_probs[0].max(dim=-1)

        items: List[WordIrab] = []
        id_to_case = ID_TO_CASE
        id_to_marker = ID_TO_MARKER
        for i, w in enumerate(words):
            ci = int(case_pred[i].item())
            ri = int(role_pred_final[i].item())
            mi = int(marker_pred[i].item())
            pi = int(pos_pred[i].item())
            wi = WordIrab(
                word=w,
                case=id_to_case.get(ci, ""),
                role=id_to_role.get(ri, ""),
                marker=id_to_marker.get(mi, ""),
                pos=POS_LABELS[pi] if pi < len(POS_LABELS) else "",
                case_conf=float(case_conf[i].item()),
                role_conf=float(role_conf_final[i].item()),
                marker_conf=float(marker_conf[i].item()),
                pos_conf=float(pos_conf[i].item()),
                irab_prose=None,
            )
            wi.irab_prose = render_word(wi)
            items.append(wi)

        return SentenceIrab(sentence=sentence, items=items), trace
