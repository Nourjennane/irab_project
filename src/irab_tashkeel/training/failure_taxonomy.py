"""Step-16 failure-mode taxonomy — heuristic inference for sampling.

Maps a :class:`Sentence` to a set of T-codes (T01..T18) based on its
schema_v2 metadata. Used by the hard-failure sampler to upweight
sentences that exhibit the failure modes we want to train on.

The codes match ``docs/error_analysis_v2/ceiling_analysis.md``:

  T01 morphology_failure
  T02 local_syntax_failure
  T03 long_range_dependency_failure
  T04 nested_clause_failure
  T05 semantic_ambiguity
  T06 discourse_context_failure
  T07 parser_alignment_failure
  T08 annotation_sparsity
  T09 rare_construction_collapse
  T10 confidence_calibration_pathology
  T11 retrieval_mismatch
  T12 evaluator_limitation
  T13 implicit_governor_failure
  T14 omitted_element_reasoning
  T15 coordination_ambiguity
  T16 clause_attachment_ambiguity
  T17 semantic_role_confusion
  T18 construction_overlap
"""
from __future__ import annotations

from typing import List, Set

from ..data_v2.schema_v2 import Sentence


# Per-code default weights (priority items from the recovery patch).
DEFAULT_TAG_WEIGHTS = {
    "T01": 1.0,
    "T02": 1.0,
    "T03": 3.0,   # long-range
    "T04": 3.0,   # nested clause
    "T05": 4.0,   # semantic ambiguity
    "T06": 1.5,
    "T09": 2.0,
    "T13": 1.5,
    "T14": 1.5,
    "T15": 3.0,   # coordination ambiguity
    "T16": 4.0,   # clause attachment
    "T17": 2.0,
    "T18": 5.0,   # construction overlap
}


def tag_sentence(s: Sentence) -> Set[str]:
    """Heuristically tag ``s`` with applicable T-codes."""
    tags: Set[str] = set()

    # T03 long-range dependency: sentence_length large or dep depth high
    if (s.curriculum.sentence_length_tokens or s.n_tokens) >= 25:
        tags.add("T03")
    if s.curriculum.dependency_depth >= 5:
        tags.add("T03")

    # T04 nested clause
    if s.curriculum.clause_depth >= 2:
        tags.add("T04")

    # T05 semantic ambiguity
    if s.curriculum.semantic_pressure_score >= 2:
        tags.add("T05")

    # T13 implicit governor — heuristic: very short sentence with construction
    if s.n_tokens <= 4 and len(s.constructions) > 0:
        tags.add("T13")

    # T14 omitted element — sentences with construction but missing
    # an expected member (best-effort: family with < expected token_indices)
    for c in s.constructions:
        if c.family in ("kana_sisters", "inna_sisters") and len(c.token_indices) < 3:
            tags.add("T14")

    # T15 coordination ambiguity — multiple `cc`/`harf_atf` markers
    n_atf = sum(1 for t in s.tokens if t.role.value == "harf_atf")
    if n_atf >= 2:
        tags.add("T15")

    # T16 clause attachment — embedded clause + ambiguous construction
    if s.curriculum.clause_depth >= 2 and s.curriculum.ambiguity_score > 0.2:
        tags.add("T16")

    # T17 semantic role confusion — high ambiguity at role level
    if s.curriculum.ambiguity_score > 0.3:
        tags.add("T17")

    # T18 construction overlap — two or more constructions sharing a token
    if len(s.constructions) >= 2:
        spans = [set(c.token_indices) for c in s.constructions]
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                if spans[i] & spans[j]:
                    tags.add("T18")
                    break

    # T08 annotation sparsity — known issue baseline; not a target
    if s.completeness.fields_complete_pct < 0.5:
        tags.add("T08")

    # Default fallback: light T01/T02 weight
    if not tags:
        tags.add("T02")

    return tags


def sentence_weight(s: Sentence, tag_weights: dict = None) -> float:
    """Compute the sampling weight for a sentence as max of its T-code weights.

    Using max (rather than sum) stops a sentence with many tags
    dominating; we still want the sampler to favour hard sentences but
    not collapse the distribution.
    """
    if tag_weights is None:
        tag_weights = DEFAULT_TAG_WEIGHTS
    tags = tag_sentence(s)
    if not tags:
        return 1.0
    return max(tag_weights.get(t, 1.0) for t in tags)


def stage_tag_weights(stage_id: int) -> dict:
    """Per-stage scaling of T-code weights.

    Stages 1-2: mostly normal — pull weights toward 1.0.
    Stages 3-7: progressively harder — amplify hard-failure tags.
    """
    base = DEFAULT_TAG_WEIGHTS
    if stage_id <= 2:
        return {k: 1.0 + 0.25 * (v - 1.0) for k, v in base.items()}
    elif stage_id <= 4:
        return {k: 1.0 + 0.6 * (v - 1.0) for k, v in base.items()}
    else:
        return base.copy()
