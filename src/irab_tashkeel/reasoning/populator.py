"""Populate :class:`schema_v2.ReasoningStep` records on a Sentence.

For each detected construction in ``sentence.constructions``, look
up the canonical template (``templates.get_template``), substitute
runtime placeholders, and append a :class:`ReasoningStep` to the
sentence.

This is a *deterministic* populator that fills reasoning steps
from the rule-based templates. A future LLM-driven populator
will produce richer per-sentence rationales (see
:mod:`reasoning.ingestor` for that).
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..data_v2.schema_v2 import Construction, ReasoningStep, Sentence
from .templates import ReasoningTemplate, get_template


# ===========================================================================
# Placeholder substitution
# ===========================================================================

def _safe_format(template_str: str, **kwargs) -> str:
    """Format ``template_str`` with kwargs, leaving missing placeholders intact."""
    if not template_str:
        return ""
    try:
        return template_str.format(**kwargs)
    except (KeyError, IndexError):
        # Fall back to a partial render — replace only known keys
        out = template_str
        for k, v in kwargs.items():
            out = out.replace("{" + k + "}", str(v))
        return out


def _substitute_for_construction(
    template: ReasoningTemplate, sentence: Sentence, c: Construction,
) -> str:
    """Substitute the standard placeholders for a kāna/inna/idāfa/etc.
    construction. Returns the rendered justification string."""
    if not template.justification:
        return ""

    n = sentence.n_tokens
    particle_surface = c.particle_surface or ""
    ism_word = ""
    khabar_word = ""
    head_noun = ""
    dependent_noun = ""
    target_word = ""

    if c.token_indices:
        if len(c.token_indices) >= 2:
            ism_idx = c.token_indices[1]
            if 0 <= ism_idx < n:
                ism_word = sentence.tokens[ism_idx].surface
        if len(c.token_indices) >= 3:
            khabar_idx = c.token_indices[2]
            if 0 <= khabar_idx < n:
                khabar_word = sentence.tokens[khabar_idx].surface

    # iḍāfa-specific
    if c.family in ("idafa", "idafa_multi") and c.children_indices:
        head_idx = c.head_idx if c.head_idx is not None else c.token_indices[0]
        if 0 <= head_idx < n:
            head_noun = sentence.tokens[head_idx].surface
        dep_idx = c.children_indices[0]
        if 0 <= dep_idx < n:
            dependent_noun = sentence.tokens[dep_idx].surface

    # istithna-specific (target = the word AFTER the particle)
    if c.family == "istithna" and c.children_indices:
        target_idx = c.children_indices[0]
        if 0 <= target_idx < n:
            target_word = sentence.tokens[target_idx].surface

    return _safe_format(
        template.justification,
        particle_surface=particle_surface,
        ism_word=ism_word,
        khabar_word=khabar_word,
        head_noun=head_noun,
        dependent_noun=dependent_noun,
        target_word=target_word,
        inherited_case="—",        # unknown without dep info
    )


# ===========================================================================
# Public API
# ===========================================================================

def populate_reasoning_for_sentence(sentence: Sentence) -> int:
    """Populate ``sentence.reasoning_steps`` for each construction.

    Returns the number of reasoning steps appended. Existing
    reasoning_steps are *not* overwritten — appended only.
    """
    if not sentence.constructions:
        return 0

    n_added = 0
    base_step_idx = max((rs.step_idx for rs in sentence.reasoning_steps),
                         default=-1) + 1
    for i, c in enumerate(sentence.constructions):
        tmpl = get_template(c.family, c.subgroup)
        if tmpl is None:
            continue
        rendered = _substitute_for_construction(tmpl, sentence, c)
        sentence.reasoning_steps.append(ReasoningStep(
            step_idx=base_step_idx + i,
            applies_to_type="construction",
            applies_to_id=c.construction_id,
            justification=rendered,
            derivation_chain=list(tmpl.derivation_chain),
            transformation_logic=tmpl.transformation_logic,
            semantic_disambiguation=tmpl.semantic_disambiguation,
            discourse_notes=tmpl.discourse_notes,
            confidence=tmpl.confidence,
            source="bronze_template",
        ))
        n_added += 1

    if n_added > 0:
        sentence.completeness.has_reasoning = True
    return n_added


def populate_reasoning_pass(sentences: Iterable[Sentence]) -> int:
    """In-place: populate reasoning steps across a corpus."""
    total = 0
    for s in sentences:
        total += populate_reasoning_for_sentence(s)
    return total
