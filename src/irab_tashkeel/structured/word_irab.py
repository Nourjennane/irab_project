"""Dataclasses for structured per-word i'rāb records.

These are the in-memory representations the new structured predictor consumes
and emits. They are deliberately decoupled from the model so utilities (eval
harness, qualitative renderer, retrieval) can import them without pulling in
torch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WordIrab:
    """One word's structured analysis + optional confidence + provenance."""

    word: str
    case: Optional[str] = None      # one of CASE_LABELS or None
    role: Optional[str] = None      # one of ROLE_LABELS or None
    marker: Optional[str] = None    # one of MARKER_LABELS or None
    pos: Optional[str] = None       # one of POS_LABELS or None

    # Per-head softmax confidence (max-prob in [0,1]); only populated by the
    # neural predictor, not by data-prep.
    case_conf: Optional[float] = None
    role_conf: Optional[float] = None
    marker_conf: Optional[float] = None
    pos_conf: Optional[float] = None

    # If symbolic constraints fired on this word, list their names (e.g.
    # "prep_to_jarr", "inna_ism_to_nasb"). Empty if none fired.
    constraints_fired: List[str] = field(default_factory=list)

    # The rendered Arabic prose (template_renderer fills this in).  None until
    # rendered.
    irab_prose: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Drop None confidences for compactness when serialized.
        for k in ("case_conf", "role_conf", "marker_conf", "pos_conf"):
            if d.get(k) is None:
                d.pop(k, None)
        if not d.get("constraints_fired"):
            d.pop("constraints_fired", None)
        if d.get("irab_prose") is None:
            d.pop("irab_prose", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WordIrab":
        return cls(
            word=d["word"],
            case=d.get("case"),
            role=d.get("role"),
            marker=d.get("marker"),
            pos=d.get("pos"),
            case_conf=d.get("case_conf"),
            role_conf=d.get("role_conf"),
            marker_conf=d.get("marker_conf"),
            pos_conf=d.get("pos_conf"),
            constraints_fired=list(d.get("constraints_fired") or []),
            irab_prose=d.get("irab_prose"),
        )

    def has_full_label(self) -> bool:
        """True if all four prediction axes are present (training filter)."""
        return self.case is not None and self.role is not None and self.marker is not None and self.pos is not None

    def min_confidence(self) -> Optional[float]:
        confs = [c for c in (self.case_conf, self.role_conf, self.marker_conf, self.pos_conf) if c is not None]
        return min(confs) if confs else None


@dataclass
class SentenceIrab:
    """A sentence + its per-word structured analyses."""

    sentence: str
    items: List[WordIrab] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"sentence": self.sentence, "items": [w.to_dict() for w in self.items]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SentenceIrab":
        return cls(
            sentence=d["sentence"],
            items=[WordIrab.from_dict(it) for it in d.get("items", [])],
        )

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json_line(cls, line: str) -> "SentenceIrab":
        return cls.from_dict(json.loads(line))
