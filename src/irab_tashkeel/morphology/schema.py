"""Canonical morphology label schema for Phase 1.

Maps Universal Dependencies (UD-PADT) FEATS columns to a small, stable set of
per-feature canonical labels. Seven feature heads are defined; each follows the
same `<value>` / ``"und"`` (undefined) idiom — ``"und"`` is BOTH a real label
class AND the catch-all bucket for words where the feature does not apply
(e.g. *aspect* for nouns or *gender* for prepositions).

Why not use raw UD strings? Two reasons:
1. The model learns canonical IDs, not raw strings; we want a stable mapping
   that is checked once and unit-tested rather than re-derived from FEATS at
   every forward.
2. UD's value space contains rare values (``Number=Tri``, ``Number=Coll``,
   ``Definite=Spec``) that are negligibly populated in PADT and would inflate
   the head's class count. We collapse them to ``"und"`` deliberately.

------------------------------------------------------------------
EXACT CoNLL-U FEATS → canonical schema mapping (Phase 1, frozen):

  Gender:    Masc → "m"     Fem → "f"      else → "und"
  Number:    Sing → "sg"    Dual → "dual"  Plur → "pl"
             (Tri/Coll → "und"; very rare in PADT)
  Definite:  Def → "def"    Ind → "indef"  Cons → "cons"
             (Spec → "und")
  Person:    1 → "1"        2 → "2"        3 → "3"
             (else → "und")
  Aspect:    Imp → "imp"    Perf → "perf"  else → "und"
             (replaces a naive "tense" head; UD-PADT uses Aspect)
  Mood:      Ind → "ind"    Imp → "imp_mood"  Sub → "sub"  Jus → "jus"
             (else → "und"; "imp_mood" disambiguated from Aspect=Imp)
  Voice:     Act → "act"    Pass → "pass"  else → "und"

UPOS → canonical 6-class POS (used by existing rev 2 POS head; not a Phase 1
addition):
  NOUN, PROPN              → "noun"
  VERB, AUX                → "verb"
  ADP, CCONJ, SCONJ, PART  → "particle"
  DET                      → "particle"   (Arabic ال)
  PRON                     → "pronoun"
  ADJ                      → "adjective"
  PUNCT                    → "punctuation"
  ADV                      → "particle"   (no adverb class in current schema)
  NUM, INTJ, SYM, X        → "noun"       (numerics & foreign tokens default
                                          to noun in absence of a better bin)

Undefined-label policy: ``"und"`` is a real predicted class in every head.
At training time it is also the catch-all for words where the feature does
not apply (e.g. tense for nouns); during dataloader build, those positions
get the canonical "und" label (NOT ``ignore_index=-100``). We do this so the
head learns to *predict* "und" for non-applicable words — that's the right
behaviour at inference time.

When labels are entirely absent (e.g. distill_v2 examples with no morph
information at all), the dataloader sets the **per-head presence flag** to 0
and the loss is masked at the example level, not the word level.

This schema is FROZEN once Phase 1 ships; future morph extensions add new
heads rather than modifying these label sets.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-feature label sets (canonical, stable, ID-mapped)
# ---------------------------------------------------------------------------
GENDER_LABELS:    List[str] = ["m", "f", "und"]
NUMBER_LABELS:    List[str] = ["sg", "dual", "pl", "und"]
DEFINITE_LABELS:  List[str] = ["def", "indef", "cons", "und"]
PERSON_LABELS:    List[str] = ["1", "2", "3", "und"]
ASPECT_LABELS:    List[str] = ["imp", "perf", "und"]
MOOD_LABELS:      List[str] = ["ind", "imp_mood", "sub", "jus", "und"]
VOICE_LABELS:     List[str] = ["act", "pass", "und"]

GENDER_TO_ID:    Dict[str, int] = {l: i for i, l in enumerate(GENDER_LABELS)}
NUMBER_TO_ID:    Dict[str, int] = {l: i for i, l in enumerate(NUMBER_LABELS)}
DEFINITE_TO_ID:  Dict[str, int] = {l: i for i, l in enumerate(DEFINITE_LABELS)}
PERSON_TO_ID:    Dict[str, int] = {l: i for i, l in enumerate(PERSON_LABELS)}
ASPECT_TO_ID:    Dict[str, int] = {l: i for i, l in enumerate(ASPECT_LABELS)}
MOOD_TO_ID:      Dict[str, int] = {l: i for i, l in enumerate(MOOD_LABELS)}
VOICE_TO_ID:     Dict[str, int] = {l: i for i, l in enumerate(VOICE_LABELS)}

ID_TO_GENDER    = {v: k for k, v in GENDER_TO_ID.items()}
ID_TO_NUMBER    = {v: k for k, v in NUMBER_TO_ID.items()}
ID_TO_DEFINITE  = {v: k for k, v in DEFINITE_TO_ID.items()}
ID_TO_PERSON    = {v: k for k, v in PERSON_TO_ID.items()}
ID_TO_ASPECT    = {v: k for k, v in ASPECT_TO_ID.items()}
ID_TO_MOOD      = {v: k for k, v in MOOD_TO_ID.items()}
ID_TO_VOICE     = {v: k for k, v in VOICE_TO_ID.items()}

N_GENDER, N_NUMBER, N_DEFINITE, N_PERSON, N_ASPECT, N_MOOD, N_VOICE = (
    len(GENDER_LABELS), len(NUMBER_LABELS), len(DEFINITE_LABELS),
    len(PERSON_LABELS), len(ASPECT_LABELS), len(MOOD_LABELS), len(VOICE_LABELS),
)

# Canonical feature name → (label_list, to_id). Source of truth for iterating
# over morph heads in the model + dataset + eval.
MORPH_FEATURES: List[str] = [
    "gender", "number", "definite", "person", "aspect", "mood", "voice",
]


# ---------------------------------------------------------------------------
# UD UPOS → canonical 6-class POS (matches existing rev 2 POS head)
# ---------------------------------------------------------------------------
UPOS_TO_CANONICAL_POS: Dict[str, str] = {
    "NOUN":      "noun",
    "PROPN":     "noun",
    "VERB":      "verb",
    "AUX":       "verb",
    "ADP":       "particle",
    "CCONJ":     "particle",
    "SCONJ":     "particle",
    "PART":      "particle",
    "DET":       "particle",
    "PRON":      "pronoun",
    "ADJ":       "adjective",
    "PUNCT":     "punctuation",
    "ADV":       "particle",
    "NUM":       "noun",
    "INTJ":      "noun",
    "SYM":       "noun",
    "X":         "noun",
}


# ---------------------------------------------------------------------------
# CoNLL-U FEATS parser
# ---------------------------------------------------------------------------
def parse_feats(feats_str: Optional[str]) -> Dict[str, str]:
    """Parse a CoNLL-U FEATS column into a {key: value} dict.

    UD-PADT FEATS column format:
        ``Gender=Fem|Number=Sing|Person=3|VerbForm=Fin|Voice=Act``
    Empty / underscore is returned as an empty dict.
    """
    if not feats_str or feats_str == "_":
        return {}
    out: Dict[str, str] = {}
    for piece in feats_str.split("|"):
        if "=" in piece:
            k, v = piece.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# Per-feature canonicalization
# ---------------------------------------------------------------------------
_CANON: Dict[str, Dict[str, str]] = {
    "gender": {
        "Masc": "m", "Fem": "f",
    },
    "number": {
        "Sing": "sg", "Dual": "dual", "Plur": "pl",
        # Tri / Coll → "und" (deliberately collapsed; <0.1% of PADT)
    },
    "definite": {
        "Def": "def", "Ind": "indef", "Cons": "cons",
        # Spec → "und" (rare)
    },
    "person": {
        "1": "1", "2": "2", "3": "3",
    },
    "aspect": {
        "Imp": "imp", "Perf": "perf",
    },
    "mood": {
        "Ind": "ind", "Imp": "imp_mood", "Sub": "sub", "Jus": "jus",
    },
    "voice": {
        "Act": "act", "Pass": "pass",
    },
}

# CoNLL-U FEATS key for each canonical morph feature.  Some UD names map
# 1-to-1 (e.g. ``Number``→``number``); a few are renamed for clarity (we keep
# canonical lowercase + drop the verb-only ``Aspect``/``VerbForm``/``Voice``
# names slightly).
_FEATS_KEY: Dict[str, str] = {
    "gender":   "Gender",
    "number":   "Number",
    "definite": "Definite",
    "person":   "Person",
    "aspect":   "Aspect",
    "mood":     "Mood",
    "voice":    "Voice",
}


def canonicalize_morph_feature(feature: str, feats: Dict[str, str]) -> str:
    """Map a UD FEATS dict to the canonical label for one feature.

    Returns the canonical string label (e.g. ``"m"`` / ``"f"`` / ``"und"``).
    The caller is responsible for converting to an integer id via the
    appropriate ``*_TO_ID`` map.
    """
    if feature not in _FEATS_KEY:
        raise ValueError(f"Unknown morph feature: {feature}")
    raw = feats.get(_FEATS_KEY[feature])
    if raw is None:
        return "und"
    table = _CANON[feature]
    return table.get(raw, "und")
