"""Phase R2 — StructuralReasoningPredictor.

Wraps :class:`StructuredPredictor` (Phase 3-A) and adds inference-time
retrieval-guided STRUCTURAL REASONING via per-construction reasoners.

Pipeline per test sentence:
1. Phase 3-A forward → raw (case_logits, role_logits, marker_logits)
   + initial argmax for construction detection.
2. Construction detection (surface particles + predicted iḍāfa role).
3. For each detected construction span:
    a. Build query signature.
    b. Retrieve top-k analogues from grammar memory.
    c. Run the family-specific structural reasoner.
    d. Apply three-tier confidence gating:
        - confidence ≥ τ_high  → OVERRIDE Phase 3-A predictions
                                  (set logits to one-hot at consensus
                                  label with a large logit value).
        - confidence ≥ τ_med   → STRONG bias (λ_strong × log(consensus prior)).
        - confidence < τ_med   → FALLBACK (no change).
4. Final argmax → SentenceIrab + StructuralReasoningTrace.

The override is implemented as a logit-replacement (set the predicted
class to a large positive logit, others to a large negative) rather
than as a discrete prediction so we keep the (logit → softmax → argmax
+ confidence) shape identical to the rest of the pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..inference.structured_predictor import StructuredPredictor
from ..structured.schema import (
    CASE_LABELS, MARKER_LABELS, POS_LABELS, ROLE_LABELS,
    CASE_TO_ID, MARKER_TO_ID, ROLE_TO_ID, ID_TO_CASE, ID_TO_MARKER, ID_TO_ROLE,
)
from ..structured.word_irab import WordIrab, SentenceIrab
from ..inference.template_renderer import render_word
from .memory import GrammarMemory, RetrievalHit
from .signature import (
    ALL_FAMILIES, ConstructionInstance, build_signature,
    detect_constructions_in_record,
)
from .structural_reasoner import (
    REASONER_REGISTRY, ReasoningOutput, get_reasoner,
)


# ---------------------------------------------------------------------------
# Trace dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SpanReasoningTrace:
    span: Tuple[int, int]
    family: str
    particle_group: str
    particle_surface: str
    n_hits: int
    confidence: float
    consensus_rate: float
    rule: str
    tier: str                     # "override" / "strong_bias" / "fallback" / "no_reasoner" / "no_retrievals"
    predicted: List[Dict]
    trace_text: str = ""


@dataclass
class StructuralReasoningTrace:
    span_traces: List[SpanReasoningTrace] = field(default_factory=list)
    n_constructions_detected: int = 0
    n_overrides: int = 0
    n_strong_bias: int = 0
    n_fallback: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _initial_pred_to_record(
    words: List[str],
    role_pred: np.ndarray,
    id_to_role: Dict[int, str],
) -> Dict:
    """Build a records-like dict from an initial Phase 3-A role argmax."""
    items = []
    for i, w in enumerate(words):
        items.append({
            "word": w,
            "role": id_to_role.get(int(role_pred[i]), ""),
        })
    return {"sentence": " ".join(words), "items": items, "source": "_query"}


def _override_logits_at_position(
    logits: torch.Tensor,
    pos: int,
    target_label_id: int,
    high_logit: float = 8.0,
    low_logit: float = -8.0,
) -> None:
    """In-place: set logits[0, pos, :] to a one-hot-like distribution where
    target_label_id has high_logit and all others have low_logit.

    After softmax this yields ~p(target) ≈ 1 - 6e-8 for typical class counts.
    """
    n = logits.shape[-1]
    new_row = torch.full((n,), low_logit, dtype=logits.dtype, device=logits.device)
    new_row[target_label_id] = high_logit
    logits[0, pos, :] = new_row


def _add_log_prior_bias_at_position(
    logits: torch.Tensor,
    pos: int,
    target_label_id: int,
    consensus_rate: float,
    lambda_strong: float,
) -> None:
    """In-place: add a *positive when consensus is high* bias to the target class.

    R2-v2 fix #1: the original formula λ · log(p) was negative for p < 1, which
    actively demoted the target class instead of boosting it. Replace with the
    log-odds formulation λ · log(p / (1 - p)), which:
      * is 0 at p = 0.5 (no bias when consensus is split),
      * is positive for p > 0.5 (boost the consensus winner),
      * is negative for p < 0.5 (rare in practice — argmax winners always have ≥ 1/n_class).

    Clamping prevents log-of-zero blow-up.
    """
    p = max(min(float(consensus_rate), 1.0 - 1e-3), 1e-3)
    bias = float(lambda_strong * math.log(p / (1.0 - p)))
    logits[0, pos, target_label_id] += bias


# Roles that indicate a particle/connector. Phase 3-A predicts these for words
# like لا (negation), قد (perfective particle), helper-particles between kana
# and the actual ism. R2-v2 fix #3: when Phase 3-A predicts harf_* at a span
# position, skip override there — surface-position alignment is misidentifying
# the position as ism/khabar when it's actually a helper particle.
HARF_ROLES = {
    "harf_jarr", "harf_atf", "harf_other",
    "harf_nafy", "harf_nasb", "harf_tahqiq",
}


def _per_field_skip(
    base_role: str,
    base_conf: float,
    target_role: str,
    base_conf_threshold: float,
) -> bool:
    """R2-v2 fix #2 + #3: decide whether to skip override at this position+field.

    Skip if:
      * Phase 3-A is confident (conf ≥ threshold), OR
      * Phase 3-A predicts harf_* role and the override target is non-harf
        (helper-particle protection).
    """
    if base_conf >= base_conf_threshold:
        return True
    if base_role in HARF_ROLES and target_role not in HARF_ROLES:
        return True
    return False


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class StructuralReasoningPredictor:
    """Wraps StructuredPredictor + GrammarMemory with structural reasoners.

    Args:
        base_predictor: a constructed StructuredPredictor (Phase 3-A).
        memory: a loaded GrammarMemory.
        tau_high: confidence threshold for symbolic override.
        tau_med: confidence threshold for strong-bias mode.
        lambda_strong: multiplier for strong-bias log-prior addition.
        retrieval_k: top-k retrieval per span.
        retrieval_alpha: symbolic vs vector mix in retrieval scoring.
        require_particle_group: if True, retrievals must match particle group.
        enabled_families: optional whitelist of construction families
            (others fall back to Phase 3-A). Defaults to all reasoners
            in the registry.
    """

    def __init__(
        self,
        base_predictor: StructuredPredictor,
        memory: GrammarMemory,
        tau_high: float = 0.75,
        tau_med: float = 0.50,
        lambda_strong: float = 1.5,
        retrieval_k: int = 5,
        retrieval_alpha: float = 0.3,
        require_particle_group: bool = True,
        enabled_families: Optional[List[str]] = None,
        # R2-v2 fix #2: trust Phase 3-A when its per-field confidence is at
        # least this. Above the threshold, we don't override even if consensus
        # is strong — Phase 3-A's calibration carries useful signal.
        base_conf_skip: float = 0.7,
        # R2-v2 fix #4: when the reasoner provides a canonical_case rule, it
        # overrides as if rate=1.0 (forced one-hot to the canonical label).
        # Set False to fall back to consensus-only behaviour.
        use_canonical_case: bool = True,
    ):
        self.base = base_predictor
        self.memory = memory
        self.tau_high = float(tau_high)
        self.tau_med = float(tau_med)
        self.lambda_strong = float(lambda_strong)
        self.k = int(retrieval_k)
        self.alpha = float(retrieval_alpha)
        self.require_particle_group = bool(require_particle_group)
        self.enabled_families = (
            set(enabled_families) if enabled_families is not None
            else set(REASONER_REGISTRY.keys())
        )
        self.base_conf_skip = float(base_conf_skip)
        self.use_canonical_case = bool(use_canonical_case)

    @torch.no_grad()
    def predict_sentence(
        self, sentence: str,
    ) -> Tuple[SentenceIrab, StructuralReasoningTrace]:
        """Predict iʿrāb with retrieval-guided structural reasoning."""
        enc = self.base._encode_sentence(sentence)
        if enc is None:
            return SentenceIrab(sentence=sentence, items=[]), StructuralReasoningTrace()

        model = self.base.model
        # 1. Phase 3-A forward — call model.forward() directly so we get
        #    BYTE-IDENTICAL logits to StructuredPredictor.predict_sentence
        #    (eliminates the manual-forward drift bug that produced the
        #    inna_marker −9.1 / idafa fully −2.1 Gazelle collateral). The
        #    only difference between Phase 3-A and R2-v2.1 is now solely
        #    the construction-span override applied AFTER the forward.
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            word_starts=enc["word_starts"],
            word_ends=enc["word_ends"],
            word_mask=enc["word_mask"],
            return_dict=True,
        )
        case_logits   = out["case_logits"]
        role_logits   = out["role_logits"]
        marker_logits = out["marker_logits"]
        pos_logits    = out["pos_logits"]
        # Reconstruct pooled_irab for retrieval span embeddings. Mirrors the
        # forward path but without the iʿrāb heads — this is the same
        # representation used at memory-build time.
        from irab_tashkeel.structured.model import _word_first_pool
        enc_out = model.encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        hidden = enc_out.last_hidden_state
        pooled = _word_first_pool(hidden, enc["word_starts"], enc["word_mask"])
        if getattr(model, "enable_dep_features", False):
            B, W = pooled.shape[:2]
            dep_emb_dim = (
                model.dep_feature_encoder.deprel_embed.embedding_dim
                + model.dep_feature_encoder.head_dir_embed.embedding_dim
                + model.dep_feature_encoder.head_dist_embed.embedding_dim
                + model.dep_feature_encoder.gov_pos_embed.embedding_dim
            )
            dep_emb = pooled.new_zeros(B, W, dep_emb_dim)
            h_aug = torch.cat([pooled, dep_emb], dim=-1)
            pooled_irab = model.dep_proj(h_aug)
        else:
            pooled_irab = pooled

        # 2. Initial argmax for construction detection
        role_probs = F.softmax(role_logits, dim=-1)
        _, role_pred_initial = role_probs[0].max(dim=-1)
        words = enc["words"]
        id_to_role = self.base.id_to_role
        record_for_detection = _initial_pred_to_record(
            words, role_pred_initial.cpu().numpy(), id_to_role,
        )

        # 3. Detect constructions
        constructions = detect_constructions_in_record(record_for_detection)
        struct_trace = StructuralReasoningTrace(n_constructions_detected=len(constructions))

        # R2-v2 fix #2 + #3: snapshot Phase 3-A's per-position predictions and
        # per-field confidences BEFORE applying any override. The override-skip
        # logic needs to read these (base role -> harf protection, base conf
        # -> trust threshold) without seeing already-modified logits.
        base_case_probs = F.softmax(case_logits, dim=-1)
        base_role_probs = F.softmax(role_logits, dim=-1)
        base_marker_probs = F.softmax(marker_logits, dim=-1)
        base_case_conf, base_case_pred = base_case_probs[0].max(dim=-1)
        base_role_conf, base_role_pred = base_role_probs[0].max(dim=-1)
        base_marker_conf, base_marker_pred = base_marker_probs[0].max(dim=-1)

        def _base_role_at(pos: int) -> str:
            return id_to_role.get(int(base_role_pred[pos].item()), "")

        # 4. Per construction: retrieve + reason + apply gating
        for span_desc in constructions:
            family = span_desc["construction"]
            start, end = span_desc["span"]
            start = max(0, min(start, len(words)))
            end = max(start, min(end, len(words)))
            if end <= start:
                continue

            # 4a. Skip families without a reasoner (e.g. iḍāfa) or disabled
            if family not in self.enabled_families or get_reasoner(family) is None:
                struct_trace.span_traces.append(SpanReasoningTrace(
                    span=(start, end), family=family,
                    particle_group=span_desc.get("particle_group", ""),
                    particle_surface=span_desc.get("particle_surface", ""),
                    n_hits=0, confidence=0.0, consensus_rate=0.0,
                    rule="", tier="no_reasoner", predicted=[],
                ))
                struct_trace.n_fallback += 1
                continue

            # 4b. Retrieve
            span_emb = pooled_irab[0, start:end].mean(dim=0).cpu().numpy()
            query = build_signature(record_for_detection, span_desc, sentence_idx=-1)
            hits = self.memory.retrieve(
                query=query,
                query_embedding=span_emb,
                k=self.k,
                alpha=self.alpha,
                require_particle_group=self.require_particle_group,
            )

            # 4c. Reason
            reasoner = get_reasoner(family)
            ro: ReasoningOutput = reasoner.reason(
                query_span=[],   # not used by current consensus reasoner
                retrieved=hits,
                query_words=words,
                span=(start, end),
                particle_group=span_desc.get("particle_group", ""),
                particle_surface=span_desc.get("particle_surface", ""),
            )

            # 4d. Apply gating — R2-v2 PER-FIELD decision logic.
            #
            # For each (position i, field f in {case, role, marker}):
            #   Skip if Phase 3-A is confident (base_conf ≥ base_conf_skip)  [fix #2]
            #   Skip if base role is harf_* and target field is non-harf      [fix #3]
            #   For case: if reasoner provides canonical_case[i]              [fix #4]
            #             → force one-hot to canonical (highest confidence)
            #   Else if consensus_rate[f] ≥ tau_high → one-hot override
            #   Else if consensus_rate[f] ≥ tau_med  → log-odds bias          [fix #1]
            #   Else → no change
            #
            # The single "tier" label is the strongest action taken on any
            # field of any position in this span (for trace visibility).
            span_max_action = "fallback"   # "fallback" < "strong_bias" < "override"

            def _bump_tier(action: str):
                # Track the strongest action applied this span.
                nonlocal span_max_action
                order = {"fallback": 0, "strong_bias": 1, "override": 2}
                if order.get(action, 0) > order.get(span_max_action, 0):
                    span_max_action = action

            if not ro.valid:
                struct_trace.n_fallback += 1
            else:
                for i in range(ro.span_len):
                    word_idx = start + i
                    if word_idx >= len(words):
                        break
                    pred = ro.predicted[i]
                    cons = ro.consensus_per_pos[i]
                    base_role_str = _base_role_at(word_idx)

                    # --- CASE field ---
                    # R2-v2.1 fix: canonical case is applied ONLY when consensus
                    # agrees with it. If consensus_winner ≠ canonical, the query
                    # is structurally non-canonical (e.g., khabar is a nested
                    # sentence with raf rather than a single noun with nasb) —
                    # fall through to consensus, don't force the rule.
                    consensus_case = pred.get("case")
                    canonical_case = (ro.canonical_case[i]
                                       if i < len(ro.canonical_case) else None)
                    use_canonical = (
                        self.use_canonical_case
                        and canonical_case is not None
                        and consensus_case is not None
                        and consensus_case == canonical_case
                    )
                    case_target = canonical_case if use_canonical else consensus_case
                    case_target_role_proxy = pred.get("role") or base_role_str
                    if case_target:
                        skip_case = _per_field_skip(
                            base_role=base_role_str,
                            base_conf=float(base_case_conf[word_idx].item()),
                            target_role=case_target_role_proxy,
                            base_conf_threshold=self.base_conf_skip,
                        )
                        if not skip_case:
                            cid = CASE_TO_ID.get(case_target, -1)
                            if cid >= 0:
                                if use_canonical:
                                    # Canonical agrees with consensus → strong override.
                                    _override_logits_at_position(case_logits, word_idx, cid)
                                    _bump_tier("override")
                                elif cons.get("case_rate", 0) >= self.tau_high:
                                    _override_logits_at_position(case_logits, word_idx, cid)
                                    _bump_tier("override")
                                elif cons.get("case_rate", 0) >= self.tau_med:
                                    _add_log_prior_bias_at_position(
                                        case_logits, word_idx, cid,
                                        cons.get("case_rate", 0.0), self.lambda_strong,
                                    )
                                    _bump_tier("strong_bias")

                    # --- ROLE field ---
                    role_target = pred.get("role")
                    if role_target:
                        skip_role = _per_field_skip(
                            base_role=base_role_str,
                            base_conf=float(base_role_conf[word_idx].item()),
                            target_role=role_target,
                            base_conf_threshold=self.base_conf_skip,
                        )
                        if not skip_role:
                            rid = ROLE_TO_ID.get(role_target, -1)
                            if rid >= 0:
                                if cons.get("role_rate", 0) >= self.tau_high:
                                    _override_logits_at_position(role_logits, word_idx, rid)
                                    _bump_tier("override")
                                elif cons.get("role_rate", 0) >= self.tau_med:
                                    _add_log_prior_bias_at_position(
                                        role_logits, word_idx, rid,
                                        cons.get("role_rate", 0.0), self.lambda_strong,
                                    )
                                    _bump_tier("strong_bias")

                    # --- MARKER field ---
                    marker_target = pred.get("marker")
                    if marker_target:
                        skip_marker = _per_field_skip(
                            base_role=base_role_str,
                            base_conf=float(base_marker_conf[word_idx].item()),
                            target_role=role_target or base_role_str,
                            base_conf_threshold=self.base_conf_skip,
                        )
                        if not skip_marker:
                            mid = MARKER_TO_ID.get(marker_target, -1)
                            if mid >= 0:
                                if cons.get("marker_rate", 0) >= self.tau_high:
                                    _override_logits_at_position(marker_logits, word_idx, mid)
                                    _bump_tier("override")
                                elif cons.get("marker_rate", 0) >= self.tau_med:
                                    _add_log_prior_bias_at_position(
                                        marker_logits, word_idx, mid,
                                        cons.get("marker_rate", 0.0), self.lambda_strong,
                                    )
                                    _bump_tier("strong_bias")

                # Update span-level counters from the strongest action applied
                if span_max_action == "override":
                    struct_trace.n_overrides += 1
                elif span_max_action == "strong_bias":
                    struct_trace.n_strong_bias += 1
                else:
                    struct_trace.n_fallback += 1
            tier = span_max_action if ro.valid else "no_retrievals"

            struct_trace.span_traces.append(SpanReasoningTrace(
                span=(start, end), family=family,
                particle_group=span_desc.get("particle_group", ""),
                particle_surface=span_desc.get("particle_surface", ""),
                n_hits=ro.n_hits, confidence=ro.confidence,
                consensus_rate=ro.consensus_rate,
                rule=ro.rule, tier=tier,
                predicted=ro.predicted,
                trace_text=ro.reasoning_trace,
            ))

        # 5. Final decode + build SentenceIrab
        case_probs = F.softmax(case_logits, dim=-1)
        role_probs_final = F.softmax(role_logits, dim=-1)
        marker_probs = F.softmax(marker_logits, dim=-1)
        pos_probs = F.softmax(pos_logits, dim=-1)
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

        return SentenceIrab(sentence=sentence, items=items), struct_trace
