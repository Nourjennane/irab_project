"""Step-6 of the supervision phase — explicit construction supervision.

Schema for annotating, per construction:

  - governor_token       — the ʿāmil (regent) that imposes case/role
  - governed_span        — the tokens whose case is set by the governor
  - attachment_target    — for nested cases, which outer construction
                            this one attaches into
  - construction_scope   — full token range of the construction
                            (may be wider than governed_span if it
                            includes setup particles, copulas, etc.)

This is *not* automatic — it requires a human grammarian.

Particularly high-leverage for:

  - kāna sisters         (governor = kāna; ism = raf, khabar = nasb)
  - inna sisters         (governor = inna; ism = nasb, khabar = raf)
  - istithnāʾ            (governor = ʾillā; mustathnā = nasb, etc.)
  - mawṣūl               (governor = the relative pronoun)
  - nested iḍāfa         (the inner head is itself the governor of
                            the next layer)

Stored alongside `data_v2/annotated/<source>/all.jsonl` as a
parallel file `governor_supervision.jsonl` keyed by sentence_id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GovernorAnnotation:
    """One construction's governor supervision."""
    sentence_id:        str
    construction_id:    str                     # echoes schema_v2 construction id
    family:             str                     # kana_sisters / inna_sisters / ...
    governor_token:     int                     # absolute token index
    governed_span:      List[int] = field(default_factory=list)
    construction_scope: List[int] = field(default_factory=list)
    attachment_target:  Optional[str] = None    # construction_id of outer ctx
    confidence:         float = 1.0
    annotator_id:       str = ""
    notes:              str = ""

    def to_dict(self) -> Dict:
        return {
            "sentence_id":        self.sentence_id,
            "construction_id":    self.construction_id,
            "family":             self.family,
            "governor_token":     self.governor_token,
            "governed_span":      list(self.governed_span),
            "construction_scope": list(self.construction_scope),
            "attachment_target":  self.attachment_target,
            "confidence":         self.confidence,
            "annotator_id":       self.annotator_id,
            "notes":              self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GovernorAnnotation":
        return cls(
            sentence_id=d["sentence_id"],
            construction_id=d["construction_id"],
            family=d["family"],
            governor_token=int(d["governor_token"]),
            governed_span=[int(x) for x in d.get("governed_span", [])],
            construction_scope=[int(x) for x in d.get("construction_scope", [])],
            attachment_target=d.get("attachment_target"),
            confidence=float(d.get("confidence", 1.0)),
            annotator_id=d.get("annotator_id", ""),
            notes=d.get("notes", ""),
        )


@dataclass
class SentenceGovernorAnnotation:
    """All governor annotations for one sentence."""
    sentence_id:    str
    annotations:    List[GovernorAnnotation] = field(default_factory=list)
    annotation_pass: int = 1
    annotation_date: str = ""

    def for_construction(self, construction_id: str
                          ) -> Optional[GovernorAnnotation]:
        for a in self.annotations:
            if a.construction_id == construction_id:
                return a
        return None


# ===========================================================================
# Canonical governance rules (deterministic seed for the annotation tool)
# ===========================================================================

# A starting policy table: which token in a construction is most
# likely the governor. The annotator can override; this provides
# pre-population to speed up annotation.
DEFAULT_GOVERNOR_HEURISTIC = {
    "kana_sisters":  "first_token",       # the kāna verb
    "inna_sisters":  "first_token",       # the inna particle
    "idafa":         "head_idx",          # the iḍāfa head (mudaaf)
    "idafa_multi":   "head_idx",
    "istithna":      "first_particle",    # ʾillā / siwā / etc.
    "mawsool":       "first_token",       # the relative pronoun
    "munada":        "first_particle",    # the yā particle
}


def heuristic_governor_token(family: str, token_indices: List[int],
                              head_idx: Optional[int]) -> int:
    """Return a heuristic guess for the governor token in a span."""
    rule = DEFAULT_GOVERNOR_HEURISTIC.get(family, "first_token")
    if rule == "head_idx" and head_idx is not None and head_idx >= 0:
        return head_idx
    if rule in ("first_token", "first_particle"):
        return min(token_indices) if token_indices else 0
    return min(token_indices) if token_indices else 0
