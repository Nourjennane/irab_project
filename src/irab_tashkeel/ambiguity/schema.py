"""Ambiguity-corpus schema.

`AmbiguityExample` extends the binary "one valid parse" assumption
with explicit alternative analyses, governor candidates, and
attachment candidates. The schema is independent from `schema_v2`'s
core `Sentence`; it *references* a sentence by id and adds the
ambiguity layer on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AmbiguityKind(str, Enum):
    IDAFA_ATTACHMENT       = "idafa_attachment"
    PREPOSITION_VS_IDAFA   = "preposition_vs_idafa"
    COORDINATION_SCOPE     = "coordination_scope"
    LATENT_GOVERNOR        = "latent_governor"
    NESTED_ATTACHMENT      = "nested_attachment"
    OMITTED_ELEMENT        = "omitted_element"
    SEMANTIC_ROLE_OVERLAP  = "semantic_role_overlap"


@dataclass
class TokenAnalysis:
    """One (case, role, marker) triple for a single token plus
    optional governor index for attachment supervision."""
    case:    Optional[str] = None
    role:    Optional[str] = None
    marker:  Optional[str] = None
    governor_token: Optional[int] = None      # absolute index, None if N/A
    note:    str = ""


@dataclass
class AmbiguityExample:
    """A single annotated ambiguity. The same sentence may have
    multiple AmbiguityExamples covering different ambiguous spans.
    """
    ambiguity_id:     str
    sentence_id:      str
    ambiguity_kind:   AmbiguityKind
    span_tokens:      List[int]                    # tokens this ambiguity covers

    # Primary (most-likely / canonical) analysis — per token in span
    primary_analysis: Dict[int, TokenAnalysis] = field(default_factory=dict)

    # Alternative valid analyses — list of complete reanalyses
    secondary_analyses: List[Dict[int, TokenAnalysis]] = field(default_factory=list)

    # Candidate governors that any of the analyses might attach to
    governor_candidates: List[int] = field(default_factory=list)

    # Candidate attachment heads (the construction this token can attach into)
    attachment_candidates: List[int] = field(default_factory=list)

    confidence_difficulty: float = 0.5            # 0..1; higher = harder
    reasoning_note:        str = ""
    annotator_id:          str = ""
    confidence:            float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        def _ta(t: TokenAnalysis) -> Dict:
            return {"case": t.case, "role": t.role, "marker": t.marker,
                    "governor_token": t.governor_token, "note": t.note}
        return {
            "ambiguity_id":     self.ambiguity_id,
            "sentence_id":      self.sentence_id,
            "ambiguity_kind":   self.ambiguity_kind.value,
            "span_tokens":      list(self.span_tokens),
            "primary_analysis": {str(k): _ta(v) for k, v in self.primary_analysis.items()},
            "secondary_analyses": [
                {str(k): _ta(v) for k, v in d.items()}
                for d in self.secondary_analyses
            ],
            "governor_candidates":   list(self.governor_candidates),
            "attachment_candidates": list(self.attachment_candidates),
            "confidence_difficulty": self.confidence_difficulty,
            "reasoning_note":        self.reasoning_note,
            "annotator_id":          self.annotator_id,
            "confidence":            self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "AmbiguityExample":
        def _ta(x: Dict) -> TokenAnalysis:
            return TokenAnalysis(
                case=x.get("case"), role=x.get("role"),
                marker=x.get("marker"),
                governor_token=x.get("governor_token"),
                note=x.get("note", ""),
            )
        return cls(
            ambiguity_id=d["ambiguity_id"],
            sentence_id=d["sentence_id"],
            ambiguity_kind=AmbiguityKind(d["ambiguity_kind"]),
            span_tokens=[int(t) for t in d.get("span_tokens", [])],
            primary_analysis={int(k): _ta(v)
                              for k, v in (d.get("primary_analysis") or {}).items()},
            secondary_analyses=[
                {int(k): _ta(v) for k, v in (alt or {}).items()}
                for alt in d.get("secondary_analyses", [])
            ],
            governor_candidates=[int(x) for x in d.get("governor_candidates", [])],
            attachment_candidates=[int(x) for x in d.get("attachment_candidates", [])],
            confidence_difficulty=float(d.get("confidence_difficulty", 0.5)),
            reasoning_note=d.get("reasoning_note", ""),
            annotator_id=d.get("annotator_id", ""),
            confidence=float(d.get("confidence", 1.0)),
        )
