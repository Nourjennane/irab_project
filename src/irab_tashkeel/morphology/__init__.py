"""Phase 1 — morphology module.

Auxiliary, opt-in feature heads (gender, number, definiteness, person, aspect,
mood, voice) trained jointly with the rev 2 i'rāb heads via masked multi-task
learning over UD Arabic-PADT + the existing distill_v2 corpus. Strict design
rules:

* **Rev 2 stays untouched.** Phase 1 lives entirely in this package + a
  flag-guarded branch in ``training/structured/train.py``. Default config has
  morph heads disabled, in which case the model + trainer are byte-identical
  to rev 2.
* **Soft hierarchy only.** Phase 1 heads are independent — no conditioning
  yet. Soft conditioning lands in Phase 2.
* **Modular and ablatable.** Each morph head is independently toggleable; per-
  head loss weights are individually configurable.

See ``docs/roadmap/phase1_morphology.md`` for the architecture, masking
strategy, ablations, and findings.
"""

from .schema import (
    GENDER_LABELS, NUMBER_LABELS, DEFINITE_LABELS, PERSON_LABELS,
    ASPECT_LABELS, MOOD_LABELS, VOICE_LABELS, UPOS_TO_CANONICAL_POS,
    GENDER_TO_ID, NUMBER_TO_ID, DEFINITE_TO_ID, PERSON_TO_ID,
    ASPECT_TO_ID, MOOD_TO_ID, VOICE_TO_ID,
    ID_TO_GENDER, ID_TO_NUMBER, ID_TO_DEFINITE, ID_TO_PERSON,
    ID_TO_ASPECT, ID_TO_MOOD, ID_TO_VOICE,
    N_GENDER, N_NUMBER, N_DEFINITE, N_PERSON, N_ASPECT, N_MOOD, N_VOICE,
    MORPH_FEATURES, parse_feats, canonicalize_morph_feature,
)
from .word_morph import WordMorph, SentenceMorph

__all__ = [
    "GENDER_LABELS", "NUMBER_LABELS", "DEFINITE_LABELS", "PERSON_LABELS",
    "ASPECT_LABELS", "MOOD_LABELS", "VOICE_LABELS", "UPOS_TO_CANONICAL_POS",
    "GENDER_TO_ID", "NUMBER_TO_ID", "DEFINITE_TO_ID", "PERSON_TO_ID",
    "ASPECT_TO_ID", "MOOD_TO_ID", "VOICE_TO_ID",
    "ID_TO_GENDER", "ID_TO_NUMBER", "ID_TO_DEFINITE", "ID_TO_PERSON",
    "ID_TO_ASPECT", "ID_TO_MOOD", "ID_TO_VOICE",
    "N_GENDER", "N_NUMBER", "N_DEFINITE", "N_PERSON", "N_ASPECT",
    "N_MOOD", "N_VOICE", "MORPH_FEATURES",
    "parse_feats", "canonicalize_morph_feature",
    "WordMorph", "SentenceMorph",
]
