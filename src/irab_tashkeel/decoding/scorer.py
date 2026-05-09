"""Sentence-level parse scorers for structured decoding.

Given a :class:`SentenceCandidate` and the surrounding context
(detected constructions, dep graph, etc.), produce a *grammar
consistency score* in [0, 1] that the reranker (Step 8 reranker)
combines with the per-token log-prob score.

Scorers
-------

- :func:`construction_consistency` — penalise candidates that
  contradict canonical construction case rules (kana ism→raf,
  khabar→nasb, idafa head→jarr, etc.)
- :func:`dep_role_consistency` — penalise role labels that don't
  fit the dep relation (e.g., a token with deprel=obj predicted
  as `mubtada`)
- :func:`agreement_consistency` — penalise predictions that break
  morphological agreement constraints (e.g., naat must agree with
  its head noun on gender/number)

All scorers return a value in [0, 1]; the reranker combines them
with weighting + the per-token log-prob.

These scorers are *deterministic and stateless* — no learnable
weights. They encode known grammar rules. The reranker
introduces tunable weights at the *combination* level, which
remains inference-time.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..data_v2.schema_v2 import Construction, Sentence
from .candidates import SentenceCandidate, TokenCandidate


# ===========================================================================
# Canonical construction case rules
# ===========================================================================

CANONICAL_CONSTRUCTION_CASE: Dict[str, Dict[int, str]] = {
    # family → {position_within_span: expected_case}
    "kana_sisters": {0: "mabni", 1: "raf",  2: "nasb"},
    "inna_sisters": {0: "mabni", 1: "nasb", 2: "raf"},
    "idafa":        {1: "jarr"},     # second token (mudaaf_ilayh) → jarr
    "idafa_multi":  {},               # too variable; use role-based check
}


# ===========================================================================
# Construction consistency
# ===========================================================================

def construction_consistency(
    candidate: SentenceCandidate, sentence: Sentence,
) -> float:
    """Fraction of canonical case slots respected by the candidate.

    Returns 1.0 if no constructions present (vacuously true), else
    the ratio of (correct case-slot match) / (total case-slot
    constraints).
    """
    constraints = 0
    matches = 0
    by_idx = {tc.token_index: tc for tc in candidate.tokens}

    for c in sentence.constructions:
        rules = CANONICAL_CONSTRUCTION_CASE.get(c.family, {})
        if not rules:
            continue
        for offset_in_span, expected_case in rules.items():
            if offset_in_span >= len(c.token_indices):
                continue
            tok_idx = c.token_indices[offset_in_span]
            tc = by_idx.get(tok_idx)
            if tc is None:
                continue
            constraints += 1
            if tc.case == expected_case:
                matches += 1

    if constraints == 0:
        return 1.0
    return matches / constraints


# ===========================================================================
# Dep-role consistency
# ===========================================================================

DEPREL_TO_CANONICAL_ROLES: Dict[str, set] = {
    # UD DEPREL → set of plausible iʿrāb roles for the dependent
    "nsubj":     {"mubtada", "fail", "ism_kana", "ism_inna"},
    "nsubj:pass": {"naib_fail"},
    "obj":       {"mafoul_bih", "mafoul_other", "khabar_kana"},
    "iobj":      {"mafoul_bih"},
    "amod":      {"naat"},
    "nmod":      {"mudaaf_ilayh"},
    "case":      {"harf_jarr"},
    "cc":        {"harf_atf"},
    "advmod":    {"dharf"},
    "discourse": {"harf_other"},
    "appos":     {"badal", "matuf"},
    "vocative":  {"munada"},
}


def dep_role_consistency(
    candidate: SentenceCandidate, sentence: Sentence,
) -> float:
    """Fraction of dep-relations whose role prediction is plausible.

    Returns 1.0 when no deps populated (all UD-PADT records have
    deps; distill_v2 ones don't always); else ratio of plausible /
    total dep-equipped tokens.
    """
    by_idx = {tc.token_index: tc for tc in candidate.tokens}
    n_total = 0
    n_plausible = 0

    for t in sentence.tokens:
        if not t.dep_label.is_present:
            continue
        deprel = t.dep_label.value
        plausible_roles = DEPREL_TO_CANONICAL_ROLES.get(deprel)
        if plausible_roles is None:
            continue
        tc = by_idx.get(t.index)
        if tc is None or tc.role is None:
            continue
        n_total += 1
        if tc.role in plausible_roles:
            n_plausible += 1

    if n_total == 0:
        return 1.0
    return n_plausible / n_total


# ===========================================================================
# Agreement consistency
# ===========================================================================

def agreement_consistency(
    candidate: SentenceCandidate, sentence: Sentence,
) -> float:
    """Fraction of declared agreement relations that are case-compatible.

    For each construction's ``agreement_relations``, check whether
    the candidate assigns the same case to both endpoints. (Full
    agreement also checks gender/number, but the candidate only
    carries case/role/marker — gender comes from the sentence's
    morph layer, not from the candidate.)
    """
    by_idx = {tc.token_index: tc for tc in candidate.tokens}
    total = 0
    consistent = 0
    for c in sentence.constructions:
        for (a, b, axes) in c.agreement_relations:
            ta = by_idx.get(a); tb = by_idx.get(b)
            if ta is None or tb is None:
                continue
            total += 1
            # Construction-level agreement_relations declared by the
            # detector are case-share constraints (the detector only
            # fires axes={"gender","number"} when both populated;
            # here we approximate with case-share as a proxy).
            if ta.case == tb.case:
                consistent += 1
    if total == 0:
        return 1.0
    return consistent / total
