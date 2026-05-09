"""Step-5 of the supervision phase — semantic ambiguity schema.

Captures cases where iʿrāb is genuinely ambiguous (multiple valid
analyses) or where a single label requires disambiguating semantic
context. The current single-label evaluator is too binary for these
cases — a token marked "naat" in gold may be analysable as "haal" by
a competent grammarian, and the model should not be penalised for
choosing the alternate analysis if the construction permits both.

Eight semantic-ambiguity patterns covered:

  - hal_vs_naat                        (حال vs نعت)
  - mafoul_bih_vs_mafoul_mutlaq        (مفعول به vs مفعول مطلق)
  - idafa_attachment_ambiguity         (which noun is the مضاف إليه?)
  - clause_attachment_ambiguity        (which clause attaches to which?)
  - coordination_ambiguity             (what does the ʿaṭf connect?)
  - omitted_governor                   (the عامل is implicit)
  - implicit_subject                   (the فاعل is implicit)
  - implicit_predicate                 (the خبر is implicit)

Each annotation can carry a list of `AlternativeAnalysis` records.
The evaluator can consume `alternative_valid_analyses` and treat any
of them as a correct answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AmbiguityKind(str, Enum):
    HAL_VS_NAAT                  = "hal_vs_naat"
    MAFOUL_BIH_VS_MUTLAQ         = "mafoul_bih_vs_mafoul_mutlaq"
    IDAFA_ATTACHMENT             = "idafa_attachment_ambiguity"
    CLAUSE_ATTACHMENT            = "clause_attachment_ambiguity"
    COORDINATION                 = "coordination_ambiguity"
    OMITTED_GOVERNOR             = "omitted_governor"
    IMPLICIT_SUBJECT             = "implicit_subject"
    IMPLICIT_PREDICATE           = "implicit_predicate"


@dataclass
class AlternativeAnalysis:
    """One valid analysis for an ambiguous span.

    ``per_token`` maps the absolute token index to the (case, role,
    marker) triple this analysis assigns. Tokens not mentioned keep
    their primary-gold labels.
    """
    note:     str = ""
    per_token: Dict[int, Dict[str, str]] = field(default_factory=dict)


@dataclass
class SemanticAmbiguity:
    """One annotated ambiguity in a sentence."""
    sentence_id:   str
    kind:          AmbiguityKind
    span_tokens:   List[int]                              # affected token indices
    primary_analysis_note: str = ""
    alternatives:  List[AlternativeAnalysis] = field(default_factory=list)
    confidence:    float = 1.0                             # annotator confidence
    annotator_id:  str = ""

    def to_dict(self) -> Dict:
        return {
            "sentence_id":   self.sentence_id,
            "kind":          self.kind.value,
            "span_tokens":   list(self.span_tokens),
            "primary_analysis_note": self.primary_analysis_note,
            "alternatives":  [
                {"note": a.note, "per_token": dict(a.per_token)}
                for a in self.alternatives
            ],
            "confidence":    self.confidence,
            "annotator_id":  self.annotator_id,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SemanticAmbiguity":
        return cls(
            sentence_id=d["sentence_id"],
            kind=AmbiguityKind(d["kind"]),
            span_tokens=list(d.get("span_tokens", [])),
            primary_analysis_note=d.get("primary_analysis_note", ""),
            alternatives=[
                AlternativeAnalysis(
                    note=a.get("note", ""),
                    per_token={int(k): v for k, v in (a.get("per_token") or {}).items()},
                ) for a in d.get("alternatives", [])
            ],
            confidence=float(d.get("confidence", 1.0)),
            annotator_id=d.get("annotator_id", ""),
        )


# ===========================================================================
# Aggregating multiple ambiguities into one sentence-level record
# ===========================================================================

@dataclass
class SentenceSemanticAnnotation:
    """All semantic ambiguities annotated on one sentence."""
    sentence_id:    str
    ambiguities:    List[SemanticAmbiguity] = field(default_factory=list)
    annotation_pass: int = 1                            # which pass / version
    annotation_date: str = ""                           # ISO date string

    def is_token_ambiguous(self, token_index: int) -> bool:
        return any(token_index in a.span_tokens for a in self.ambiguities)

    def alternatives_for_token(
        self, token_index: int,
    ) -> List[Dict[str, str]]:
        """Return all alternative (case, role, marker) labels valid for
        this token across all annotated ambiguities. The evaluator
        uses this to construct a permissive gold set for the token.
        """
        out: List[Dict[str, str]] = []
        for amb in self.ambiguities:
            if token_index not in amb.span_tokens:
                continue
            for alt in amb.alternatives:
                if token_index in alt.per_token:
                    out.append(dict(alt.per_token[token_index]))
        return out


# ===========================================================================
# Evaluator integration helper
# ===========================================================================

def is_alternate_correct(
    annotation: Optional[SentenceSemanticAnnotation],
    token_index: int,
    pred: Dict[str, str],
) -> bool:
    """Return True iff ``pred`` matches *any* alternative analysis for
    this token. Used by the permissive evaluator: a primary-gold
    mismatch can still be 'correct' if it matches an alternative.
    """
    if annotation is None:
        return False
    alts = annotation.alternatives_for_token(token_index)
    if not alts:
        return False
    for alt in alts:
        if all(pred.get(k) == v for k, v in alt.items()):
            return True
    return False
