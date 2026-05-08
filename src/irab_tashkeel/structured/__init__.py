"""Structured i'rāb prediction (Phase 3 / v1 rebuild).

Treats per-word i'rāb as multi-head classification over a small canonical schema
instead of free-form Arabic prose generation. Sister module to the existing
:mod:`irab_tashkeel.evaluation.structural` extractor: the extractor parses
prose into the same structured fields this module predicts, so eval is on the
same surface.
"""

from .schema import (
    CASE_LABELS, ROLE_LABELS, MARKER_LABELS, POS_LABELS,
    CASE_TO_ID, ROLE_TO_ID, MARKER_TO_ID, POS_TO_ID,
    ID_TO_CASE, ID_TO_ROLE, ID_TO_MARKER, ID_TO_POS,
    canonicalize_case, canonicalize_role, canonicalize_marker, canonicalize_pos, derive_pos,
    arabic_normalize,
    N_CASE, N_ROLE, N_MARKER, N_POS,
)
from .word_irab import WordIrab, SentenceIrab

__all__ = [
    "CASE_LABELS", "ROLE_LABELS", "MARKER_LABELS", "POS_LABELS",
    "CASE_TO_ID", "ROLE_TO_ID", "MARKER_TO_ID", "POS_TO_ID",
    "ID_TO_CASE", "ID_TO_ROLE", "ID_TO_MARKER", "ID_TO_POS",
    "canonicalize_case", "canonicalize_role", "canonicalize_marker",
    "canonicalize_pos", "derive_pos", "arabic_normalize",
    "N_CASE", "N_ROLE", "N_MARKER", "N_POS",
    "WordIrab", "SentenceIrab",
]
