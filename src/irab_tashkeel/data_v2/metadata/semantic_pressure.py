"""Semantic-pressure scoring for schema_v2 sentences.

Per-sentence score 0..3 estimating how much the sentence's
grammatical decisions depend on semantics rather than surface
syntax. Used by the curriculum scheduler (Step 7) to time the
introduction of semantic-supervision in training, and by eval_v2
to stratify metrics by semantic pressure.

Scoring rubric (from docs/error_analysis_v2/semantic_pressure.md
in the Step 16 taxonomy):

    0 — pure syntax       (case/marker only depend on dep tree)
    1 — syntax-leaning    (long-range dep but clear governor)
    2 — semantic-leaning  (semantic-role ambiguity, attachment ambiguity)
    3 — semantic-required (omitted element, discourse, hal vs naat)
"""
from __future__ import annotations

from typing import Set

from ..schema_v2 import Sentence


# Constructions that systematically introduce semantic ambiguity
SEMANTIC_AMBIGUITY_CONSTRUCTIONS: Set[str] = {
    "istithna",        # munqaṭiʿ vs muttaṣil semantic distinction
    "mawsool",         # antecedent reference can be semantic
    "quranic_proxy",   # Quranic context often semantic
}

# Roles that frequently require semantic disambiguation
SEMANTIC_AMBIGUOUS_ROLES: Set[str] = {
    "hal",             # circumstantial accusative — vs naat (descriptive)
    "naat",            # adjective modifier — vs hal
    "tamyeez",         # specifier — semantic class
    "badal",           # apposition — vs naat
    "mafoul_other",    # paronymous/locative/comitative — semantic distinction
}

# Roles that signal omitted-element reasoning
OMITTED_ELEMENT_ROLES: Set[str] = {
    "other",           # frozen-baseline catch-all for ḍamīr mustatir / omitted elements
}


def score_sentence(sentence: Sentence) -> int:
    """Return a 0..3 semantic-pressure score for ``sentence``."""
    n_tokens = sentence.n_tokens
    if n_tokens == 0:
        return 0

    # Layer 1: any token with a semantically-ambiguous role → ≥ 2
    ambiguous_role_tokens = sum(
        1 for t in sentence.tokens
        if t.role.value in SEMANTIC_AMBIGUOUS_ROLES
    )
    omitted_role_tokens = sum(
        1 for t in sentence.tokens
        if t.role.value in OMITTED_ELEMENT_ROLES
    )

    # Layer 2: discourse links → 3
    if sentence.discourse_links:
        return 3

    # Layer 3: omitted elements → 3
    if omitted_role_tokens > 0:
        return 3

    # Layer 4: ambiguous constructions → 2
    has_ambiguous_construction = any(
        c.family in SEMANTIC_AMBIGUITY_CONSTRUCTIONS
        for c in sentence.constructions
    )
    if has_ambiguous_construction or ambiguous_role_tokens > 0:
        return 2

    # Layer 5: long-range dep (max head distance ≥ 4) → 1
    max_head_dist = 0
    for t in sentence.tokens:
        if t.dep_head_idx is None or t.dep_head_idx < 0:
            continue
        max_head_dist = max(max_head_dist, abs(t.dep_head_idx - t.index))
    if max_head_dist >= 4:
        return 1

    return 0
