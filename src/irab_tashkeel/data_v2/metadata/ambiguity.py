"""Ambiguity score heuristic for schema_v2 sentences.

Returns a float in [0, 1] estimating how many competing parses the
sentence admits. Based on:

- per-construction ``ambiguity_score`` (set by detector confidence)
- presence of ``alternative_analyses`` per construction
- attachment-ambiguous dep labels (acl, nmod, amod) with low parser conf
- coordination structures (multiple conj children)
- presence of pronouns (often discourse-ambiguous)

This is a deliberately rough heuristic — fine-grained ambiguity
analysis happens at the eval_v2 stage with full alt-parse
enumeration. The score here is for curriculum scheduling and
sentence selection, not for paper-grade claims.
"""
from __future__ import annotations

from ..schema_v2 import Sentence

ATTACHMENT_AMBIGUOUS = {"acl", "nmod", "amod", "appos"}
COORDINATION = {"conj", "cc"}


def score_sentence(sentence: Sentence) -> float:
    """Return a 0..1 ambiguity score."""
    n_tokens = max(sentence.n_tokens, 1)
    components: list[float] = []

    # Component 1: max construction ambiguity
    if sentence.constructions:
        max_amb = max(c.ambiguity_score for c in sentence.constructions)
        components.append(max_amb)

    # Component 2: density of alternative analyses
    alt_count = sum(len(c.alternative_analyses) for c in sentence.constructions)
    components.append(min(alt_count / max(len(sentence.constructions), 1), 1.0))

    # Component 3: density of attachment-ambiguous deps
    n_amb_dep = sum(
        1 for t in sentence.tokens
        if t.dep_label.value in ATTACHMENT_AMBIGUOUS
    )
    components.append(min(n_amb_dep / n_tokens, 1.0))

    # Component 4: density of coordination
    n_coord = sum(
        1 for t in sentence.tokens
        if t.dep_label.value in COORDINATION
    )
    components.append(min(n_coord / n_tokens, 1.0))

    # Component 5: pronoun density
    n_pronouns = sum(
        1 for t in sentence.tokens
        if t.morph.pos.value == "pronoun" or t.pos.value == "pronoun"
    )
    components.append(min(n_pronouns / n_tokens, 1.0))

    if not components:
        return 0.0
    return sum(components) / len(components)
