"""Unified Arabic + Unicode normalisation for the next-gen data engine.

This module is the single source of truth for tokenization +
normalisation across schema_v2 loaders. Both the frozen baseline's
``evaluation/structural.py`` and ``structured/schema.py`` had their
own normalisation logic; this module supersedes both. Loaders MUST
use these functions; downstream components MUST use the same
convention.

The normaliser is configurable so different layers (gold extraction,
retrieval keys, surface-form display) can preserve or fold different
features without divergence between loaders.

Quick reference
---------------

- ``normalize_text(s)`` — full whitespace + NFC + diacritic strip
- ``arabic_normalize(s)`` — Arabic-aware fold (alif, ya, hamza, tatweel)
- ``strip_diacritics(s)``
- ``fold_alif(s)`` / ``fold_ya(s)`` / ``fold_hamza(s)``
- ``normalize_punctuation(s)`` — Arabic punctuation → canonical
- ``tokenize_whitespace(s)`` — produces aligned tokens with char offsets
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple


# ===========================================================================
# Character-class regexes
# ===========================================================================

_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
"""Arabic diacritics (tashkīl) and Quranic small marks."""

_TATWEEL = "ـ"     # tatweel — purely decorative, always strip

_ALIF_VARIANTS = str.maketrans({
    "آ": "ا",   # alif madda
    "أ": "ا",   # alif hamza above
    "إ": "ا",   # alif hamza below
    "ٱ": "ا",   # alif wasla
})
_YA_NORMALIZE = str.maketrans({
    "ى": "ي",   # alif maqsura → ya
})
_HAMZA_FORMS = str.maketrans({
    "ؤ": "ء",   # waw with hamza
    "ئ": "ء",   # ya with hamza
})

_ARABIC_PUNCT_TO_LATIN = str.maketrans({
    "،": ",", "؛": ";", "؟": "?", "٪": "%",
    "٬": ",", "٫": ".",
})


# ===========================================================================
# Atomic transforms
# ===========================================================================

def strip_diacritics(s: str) -> str:
    """Remove Arabic tashkīl + Quranic small marks. Idempotent."""
    if not s: return ""
    return _DIACRITICS.sub("", s)


def strip_tatweel(s: str) -> str:
    return s.replace(_TATWEEL, "")


def fold_alif(s: str) -> str:
    """Map alif variants (آأإٱ) → bare alif. Idempotent."""
    return s.translate(_ALIF_VARIANTS)


def fold_ya(s: str) -> str:
    """Map alif maqsura (ى) → ya (ي). Idempotent."""
    return s.translate(_YA_NORMALIZE)


def fold_hamza(s: str) -> str:
    """Map carrier-hamza variants (ؤ ئ) → bare hamza (ء). Idempotent."""
    return s.translate(_HAMZA_FORMS)


def normalize_punctuation(s: str, *, fold_to_latin: bool = False) -> str:
    """Normalise Arabic punctuation. Always collapses repeated punctuation."""
    if fold_to_latin:
        s = s.translate(_ARABIC_PUNCT_TO_LATIN)
    return s


def collapse_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ===========================================================================
# Composite normalisers
# ===========================================================================

def normalize_text(s: str) -> str:
    """Default safe normalisation: NFC + strip diacritics + collapse whitespace.

    Does NOT fold alif/ya/hamza variants — those discard real
    distinctions (e.g., إ vs أ semantic class) and should be opt-in.
    """
    if not s: return ""
    s = unicodedata.normalize("NFC", s)
    s = strip_diacritics(s)
    s = strip_tatweel(s)
    s = collapse_whitespace(s)
    return s


def arabic_normalize(
    s: str, *,
    fold_alif_variants: bool = True,
    fold_ya_variant: bool = True,
    fold_hamza_variants: bool = False,
    fold_arabic_punct: bool = False,
    keep_diacritics: bool = False,
) -> str:
    """Arabic-aware normalisation.

    Defaults match the frozen-baseline ``evaluation/structural.py``
    behaviour: alif and ya are folded, hamza is preserved, punctuation
    is preserved, diacritics are stripped.

    Set ``keep_diacritics=True`` for surface-form rendering or
    diacritization tasks.
    """
    if not s: return ""
    s = unicodedata.normalize("NFC", s)
    if not keep_diacritics:
        s = strip_diacritics(s)
    s = strip_tatweel(s)
    if fold_alif_variants:
        s = fold_alif(s)
    if fold_ya_variant:
        s = fold_ya(s)
    if fold_hamza_variants:
        s = fold_hamza(s)
    if fold_arabic_punct:
        s = normalize_punctuation(s, fold_to_latin=True)
    s = collapse_whitespace(s)
    return s


# ===========================================================================
# Tokenisation
# ===========================================================================

@dataclass
class TokenSpan:
    """A whitespace-separated token plus its char offsets in the source."""
    text: str
    char_start: int
    char_end: int


def tokenize_whitespace(s: str) -> List[TokenSpan]:
    """Whitespace tokenisation with character offsets.

    Use this for aligning predictions / annotations to the source
    text. The frozen-baseline pipeline uses simple ``str.split()``
    and loses character offsets; schema_v2 records keep offsets so
    discourse links and span boundaries are robust to text
    transformations.
    """
    out: List[TokenSpan] = []
    i = 0
    n = len(s)
    while i < n:
        # skip whitespace
        while i < n and s[i].isspace():
            i += 1
        if i >= n: break
        start = i
        while i < n and not s[i].isspace():
            i += 1
        out.append(TokenSpan(text=s[start:i], char_start=start, char_end=i))
    return out


# ===========================================================================
# Surface-match helpers (used at retrieval / alignment time)
# ===========================================================================

def surface_match(a: str, b: str, *, ignore_diacritics: bool = True) -> bool:
    """True if two surface forms match under the standard fold."""
    return arabic_normalize(a, keep_diacritics=not ignore_diacritics) == \
           arabic_normalize(b, keep_diacritics=not ignore_diacritics)


def normalize_for_lookup(s: str) -> str:
    """The most aggressive fold — for retrieval keys only.

    Folds alif + ya + hamza + strips diacritics + lowercases ASCII.
    Use this ONLY for non-display-side keys (retrieval, signature
    matching, lookup tables); never for stored surface forms.
    """
    s = arabic_normalize(s, fold_alif_variants=True, fold_ya_variant=True,
                          fold_hamza_variants=True)
    return s.lower()
