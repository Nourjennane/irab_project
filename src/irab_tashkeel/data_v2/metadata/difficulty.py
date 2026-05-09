"""Curriculum difficulty + structural-depth metadata for schema_v2.

Computes :class:`CurriculumMetadata` for a :class:`Sentence`,
returning a value 1..7 that maps to the curriculum stages defined
in ``src/irab_tashkeel/curriculum/README.md``.

Stage rubric:

    1 — pure morphology, no constructions, dep_depth ≤ 2
    2 — local syntax, dep_depth ≤ 3, no nested clauses
    3 — simple constructions (single iḍāfa, single kāna sister, etc.)
    4 — nested constructions (iḍāfa chain, embedded clauses)
    5 — semantic interactions (semantic-pressure ≥ 2)
    6 — discourse-sensitive structures (discourse_links present
        OR pronoun antecedents)
    7 — Quranic / classical complexity (Quranic register +
        omitted elements + multi-construction overlap)

Difficulty assignment is monotonic in the underlying signals;
the highest qualifying stage wins.
"""
from __future__ import annotations

from typing import Set

from ..schema_v2 import (
    AnnotationCompleteness, CurriculumMetadata, Domain, Sentence,
)
from . import semantic_pressure, ambiguity


# ---------------------------------------------------------------------------
# Structural-depth helpers
# ---------------------------------------------------------------------------

def dep_depth(sentence: Sentence) -> int:
    """Max depth of the dependency tree (0 = root, 1 = root's children, etc.).

    Robust to malformed dep_head_idx; returns 0 if no dep info.
    """
    n = sentence.n_tokens
    if n == 0: return 0
    children: dict[int, list[int]] = {-2: []}
    for t in sentence.tokens:
        head = t.dep_head_idx
        if head is None or head < -2: head = -1
        children.setdefault(head, []).append(t.index)

    # Walk from root (head_idx=-2) downward
    max_depth = 0
    stack = [(idx, 1) for idx in children.get(-2, [])]
    visited: Set[int] = set()
    iterations = 0
    while stack and iterations < 10000:
        iterations += 1
        idx, d = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        max_depth = max(max_depth, d)
        for c in children.get(idx, []):
            if c not in visited:
                stack.append((c, d + 1))
    return max_depth


def clause_depth(sentence: Sentence) -> int:
    if not sentence.clauses:
        return 0
    return max(c.depth for c in sentence.clauses)


def construction_overlap_count(sentence: Sentence) -> int:
    """How many tokens are covered by ≥ 2 constructions."""
    if not sentence.constructions:
        return 0
    cover_count: dict[int, int] = {}
    for c in sentence.constructions:
        for i in c.token_indices:
            cover_count[i] = cover_count.get(i, 0) + 1
    return sum(1 for v in cover_count.values() if v >= 2)


# ---------------------------------------------------------------------------
# Stage assignment
# ---------------------------------------------------------------------------

def difficulty_level(sentence: Sentence) -> int:
    n_constructions = len(sentence.constructions)
    n_overlap = construction_overlap_count(sentence)
    dd = dep_depth(sentence)
    cd = clause_depth(sentence)
    sp = semantic_pressure.score_sentence(sentence)

    # Stage 7: Quranic / classical complexity
    if (sentence.metadata.domain == Domain.QURANIC.value
        and (n_overlap > 0 or sp >= 3)):
        return 7
    if (sentence.metadata.domain == Domain.CLASSICAL.value
        and (n_overlap > 0 or sp >= 3)):
        return 7

    # Stage 6: discourse-sensitive
    if sentence.discourse_links or sp >= 3:
        return 6

    # Stage 5: semantic interactions
    if sp >= 2:
        return 5

    # Stage 4: nested constructions
    if cd >= 2 or n_overlap >= 1 or n_constructions >= 2:
        return 4

    # Stage 3: simple constructions
    if n_constructions >= 1:
        return 3

    # Stage 2: local syntax
    if dd >= 2:
        return 2

    # Stage 1: pure morphology
    return 1


# ---------------------------------------------------------------------------
# Top-level helper
# ---------------------------------------------------------------------------

def populate_metadata(sentence: Sentence) -> Sentence:
    """In-place: compute and attach curriculum metadata.

    Called by post-loader passes; loaders themselves do NOT
    populate this (keeping loaders deterministic + reusable).
    """
    n_tokens = sentence.n_tokens
    dd = dep_depth(sentence)
    cd = clause_depth(sentence)
    n_overlap = construction_overlap_count(sentence)
    sp = semantic_pressure.score_sentence(sentence)
    amb = ambiguity.score_sentence(sentence)

    sentence.curriculum = CurriculumMetadata(
        difficulty_level=difficulty_level(sentence),
        dependency_depth=dd,
        clause_depth=cd,
        construction_count=len(sentence.constructions),
        nested_construction_count=n_overlap,
        ambiguity_score=amb,
        semantic_pressure_score=sp,
        discourse_complexity=float(len(sentence.discourse_links)),
        sentence_length_tokens=n_tokens,
        nested_clause_count=sum(1 for c in sentence.clauses if c.depth > 0),
    )
    return sentence


def populate_all(sentences) -> int:
    """Apply metadata population to an iterable of sentences. Returns count."""
    n = 0
    for s in sentences:
        populate_metadata(s)
        n += 1
    return n
