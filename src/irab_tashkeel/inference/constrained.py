"""Constrained decoding for i'rāb generation.

Two regimes available:

  1. **Postprocessor (default, no extra deps).**
     Parse the model's free generation via the structural extractor, then
     reassemble into a canonical template using only taxonomy terms. If the
     generated string contains junk or hallucinations, this snaps it back to
     well-formed Arabic.

  2. **Outlines/XGrammar grammar mask** (optional).
     For decoder-only HF models, an EBNF-style grammar can mask logits at
     generation time. Provide a `build_outlines_regex()` helper; the actual
     wrapping is done by the caller (we don't add Outlines as a hard dep).

The postprocessor is what to use unless you've already pip-installed Outlines
or vLLM with grammar support.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable, List, Optional

from ..evaluation.structural import (
    CASES, MARKERS, POS_TERMS, ROLES, IrabAnalysis, extract,
)


# Canonical Arabic phrase per case, used when reassembling.
_CASE_TEMPLATE = {
    "marfu":  "مرفوع وعلامة رفعه {marker}",
    "mansub": "منصوب وعلامة نصبه {marker}",
    "majrur": "مجرور وعلامة جره {marker}",
    "majzum": "مجزوم وعلامة جزمه {marker}",
    "mabni":  "مبني",
}

_DEFAULT_MARKER = {
    "marfu":  "الضمة الظاهرة",
    "mansub": "الفتحة الظاهرة",
    "majrur": "الكسرة الظاهرة",
    "majzum": "السكون",
}


def _snap(value: Optional[str], vocabulary: Iterable[str], cutoff: float = 0.6) -> Optional[str]:
    """Snap an extracted term to the nearest entry in `vocabulary`.

    Returns the value unchanged if it's already in the vocabulary, otherwise
    the closest match above `cutoff`, otherwise None.
    """
    if value is None:
        return None
    vocab = list(vocabulary)
    if value in vocab:
        return value
    matches = difflib.get_close_matches(value, vocab, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def reassemble(analysis: IrabAnalysis) -> str:
    """Build a canonical i'rāb string from extracted fields, snapping to taxonomy."""
    pos = _snap(analysis.pos, POS_TERMS)
    role = _snap(analysis.role, ROLES)
    marker = _snap(analysis.marker, MARKERS)
    case = analysis.case  # already a closed-set label from extract()

    parts: List[str] = []
    if pos and pos != role:
        parts.append(pos)
    if role:
        parts.append(role)

    if case in _CASE_TEMPLATE:
        if case == "mabni":
            parts.append(_CASE_TEMPLATE["mabni"])
        else:
            mk = marker or _DEFAULT_MARKER[case]
            parts.append(_CASE_TEMPLATE[case].format(marker=mk))

    if not parts:
        return analysis.pos or analysis.role or "كلمة"
    return " ".join(parts)


def constrain(irab_text: str) -> str:
    """One-shot: take a free generation, return a canonical taxonomy-clean string."""
    return reassemble(extract(irab_text))


# ---------------------------------------------------------------------------
# Outlines/XGrammar (optional) — return a regex over the closed vocab.
# The caller wires it into Outlines.generate / vLLM with this regex.
# ---------------------------------------------------------------------------
def build_irab_regex() -> str:
    """Return a single regex matching the canonical i'rāb-line shape.

        <pos>?  <role>?  <case_clause>?
            where case_clause = "<word_for_case> وعلامة <kasra/...> <marker>"
    """
    pos = "(?:" + "|".join(re.escape(t) for t in POS_TERMS) + ")"
    role = "(?:" + "|".join(re.escape(t) for t in ROLES) + ")"
    marker = "(?:" + "|".join(re.escape(t) for t in MARKERS) + ")"
    case_words = "|".join(re.escape(w) for words in CASES.values() for w in words)
    case_clause = rf"(?:{case_words})(?:[؀-ۿ\s]*?{marker})?"
    return rf"{pos}?\s*{role}?\s*{case_clause}?\s*"


def build_irab_lines_regex(max_words: int = 64) -> str:
    """Regex matching a multi-line per-word output with at most `max_words` lines."""
    line = rf"[؀-ۿ]+\s*[:：]\s*{build_irab_regex()}"
    return rf"(?:{line}\n?){{1,{max_words}}}"
