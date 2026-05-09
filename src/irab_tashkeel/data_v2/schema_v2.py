"""Schema v2 — canonical supervision format for the next-generation system.

This module defines the dataclasses that every source loader, training
pipeline, evaluator, and downstream component on the
``nextgen-grammatical-reasoning`` branch must use.

Design principles
-----------------

1. **Permanence over minimalism.** Empty placeholders are fine; missing
   fields are not. Every label carries provenance + confidence so future
   experiments can stratify by annotation quality without rewriting
   loaders.

2. **Multi-layer supervision.** Per-token (Layer A morph + B local syntax),
   per-span (Layer B phrasal), per-clause (B/C), per-construction (C),
   and per-sentence (D reasoning). Each layer is independently
   optional — a sentence can have rich morph but no constructions, or
   gold dep but no reasoning.

3. **Graph-ready from day one.** ``GrammarGraph`` slot exists even when
   no graph is computed; graph edges are first-class so retrieval v2
   and decoding v2 can index them directly without schema migration.

4. **Reasoning placeholders now.** Even when no reasoning data is
   loaded, ``ReasoningStep`` records exist as empty lists; the field
   structure (justification / derivation_chain / alternative_parses /
   ambiguity_notes / semantic_disambiguation / discourse_notes) is
   stable so when LLM-generated traces land later they slot in
   without breaking existing data.

5. **Construction-aware indexing.** ``Construction`` objects with full
   span / head / children / clause-membership / agreement / ambiguity
   support, replacing the flat ``signature.py`` pattern from the
   frozen baseline.

6. **Eval v2 contract.** Every annotation field carries a
   ``LabelTag`` with ``source``, ``confidence``, and ``alternatives``.
   The eval engine stratifies metrics by these.

7. **JSONL-friendly.** Every dataclass has ``to_dict()`` and
   ``from_dict()`` for stable, versioned, on-disk storage. Forward
   migration honours older schema versions.

Schema version: 2.0.0.

If you change anything in this file, bump SCHEMA_VERSION and
document the migration in docs/data_v2/SCHEMA_V2.md §Migration.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

SCHEMA_VERSION = "2.0.0"


# ===========================================================================
# Enumerations (stored as strings for forward compatibility)
# ===========================================================================

class AnnotationQuality(str, Enum):
    """Source-quality tiers used to stratify training and evaluation.

    Order matters: GOLD_HUMAN > GOLD_TREEBANK > SILVER_LLM_DISTILL >
    SILVER_PARSER_HIGH_CONF > BRONZE_PARSER_LOW_CONF > BRONZE_HEURISTIC >
    UNKNOWN.
    """
    GOLD_HUMAN              = "gold_human"
    GOLD_TREEBANK           = "gold_treebank"
    SILVER_LLM_DISTILL      = "silver_llm_distill"
    SILVER_PARSER_HIGH_CONF = "silver_parser_high_conf"
    BRONZE_PARSER_LOW_CONF  = "bronze_parser_low_conf"
    BRONZE_HEURISTIC        = "bronze_heuristic"
    UNKNOWN                 = "unknown"


class Domain(str, Enum):
    MSA_NEWS    = "msa_news"
    QURANIC     = "quranic"
    CLASSICAL   = "classical"
    EDUCATIONAL = "educational"
    PEDAGOGICAL = "pedagogical"
    DIALECT     = "dialect"
    UNKNOWN     = "unknown"


class ClauseType(str, Enum):
    MATRIX            = "matrix"
    NOMINAL_EMBEDDED  = "nominal_embedded"
    VERBAL_EMBEDDED   = "verbal_embedded"
    RELATIVE          = "relative"
    CONDITIONAL       = "conditional"
    PURPOSE           = "purpose"
    CIRCUMSTANTIAL    = "circumstantial"
    INTERROGATIVE     = "interrogative"
    VOCATIVE          = "vocative"
    UNKNOWN           = "unknown"


class EdgeType(str, Enum):
    DEP         = "dep"
    SEMANTIC    = "semantic"
    AGREEMENT   = "agreement"
    GOVERNOR    = "governor"
    COREF       = "coref"
    DISCOURSE   = "discourse"
    CONSTRUCTION_MEMBER = "construction_member"
    CLAUSE_MEMBER       = "clause_member"


# ===========================================================================
# Identifier helper
# ===========================================================================

def new_id(prefix: str = "") -> str:
    """Generate a stable UUID-based id with optional prefix."""
    base = uuid.uuid4().hex[:12]
    return f"{prefix}_{base}" if prefix else base


# ===========================================================================
# LabelTag — recurring sub-structure for every annotated field
# ===========================================================================

@dataclass
class LabelTag:
    """Wraps an annotated field's value with provenance + confidence + alternatives.

    Fields
    ------
    value
        Canonical label value, or ``None`` when the source did not
        produce one. ``None`` is meaningful — it distinguishes "this
        field was not annotated" from "this field was annotated with
        the empty string".
    source
        Origin string. Standard values: ``"gold_human"``,
        ``"gold_treebank"``, ``"silver_llm_distill"``,
        ``"silver_stanza"``, ``"silver_madamira"``, ``"bronze_heuristic"``.
        Loaders set this; do not invent new sources without updating
        the SCHEMA_V2.md provenance table.
    confidence
        Float in [0, 1]. Required even for gold (set to 1.0). For
        parser-derived sources, propagate the parser's reported
        confidence when available; otherwise use 0.5 as the default
        BRONZE_PARSER_LOW_CONF marker.
    alternatives
        Top-k alternatives observed by the source. Each entry is
        ``(value, confidence)``. Empty when the source emits a single
        deterministic answer.
    notes
        Free-form annotation notes (e.g., "non-canonical surface;
        kept for ambiguity audit"). Optional.
    """
    value: Optional[str] = None
    source: str = ""
    confidence: float = 1.0
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    notes: str = ""

    @property
    def is_present(self) -> bool:
        return self.value is not None and self.value != ""

    def to_dict(self) -> Dict[str, Any]:
        d = {"value": self.value, "source": self.source,
             "confidence": self.confidence}
        if self.alternatives:
            d["alternatives"] = [list(a) for a in self.alternatives]
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "LabelTag":
        if d is None:
            return cls()
        alts = d.get("alternatives") or []
        return cls(
            value=d.get("value"),
            source=d.get("source", ""),
            confidence=d.get("confidence", 1.0),
            alternatives=[tuple(a) for a in alts],
            notes=d.get("notes", ""),
        )


# ===========================================================================
# Layer A — Morphology
# ===========================================================================

@dataclass
class Morphology:
    """Per-token morphological labels (Layer A).

    Each axis is a :class:`LabelTag` so the same field can carry
    multiple sources at different confidence (e.g., gold UD-PADT for
    UD records, Stanza-derived for distill_v2 records).
    """
    gender:           LabelTag = field(default_factory=LabelTag)
    number:           LabelTag = field(default_factory=LabelTag)
    person:           LabelTag = field(default_factory=LabelTag)
    definite:         LabelTag = field(default_factory=LabelTag)
    pos:              LabelTag = field(default_factory=LabelTag)
    inflection_class: LabelTag = field(default_factory=LabelTag)
    mood:             LabelTag = field(default_factory=LabelTag)
    voice:            LabelTag = field(default_factory=LabelTag)
    aspect:           LabelTag = field(default_factory=LabelTag)
    derivation_form:  LabelTag = field(default_factory=LabelTag)   # Form I-X for verbs

    # Pairwise agreement: (other_token_idx, list_of_axes_in_agreement)
    agreement_with: List[Tuple[int, List[str]]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in ("gender", "number", "person", "definite", "pos",
                  "inflection_class", "mood", "voice", "aspect",
                  "derivation_form"):
            v = getattr(self, k)
            if v.is_present or v.alternatives:
                out[k] = v.to_dict()
        if self.agreement_with:
            out["agreement_with"] = [(i, list(axes))
                                       for i, axes in self.agreement_with]
        return out

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Morphology":
        if d is None:
            return cls()
        return cls(
            gender=LabelTag.from_dict(d.get("gender")),
            number=LabelTag.from_dict(d.get("number")),
            person=LabelTag.from_dict(d.get("person")),
            definite=LabelTag.from_dict(d.get("definite")),
            pos=LabelTag.from_dict(d.get("pos")),
            inflection_class=LabelTag.from_dict(d.get("inflection_class")),
            mood=LabelTag.from_dict(d.get("mood")),
            voice=LabelTag.from_dict(d.get("voice")),
            aspect=LabelTag.from_dict(d.get("aspect")),
            derivation_form=LabelTag.from_dict(d.get("derivation_form")),
            agreement_with=[(int(i), list(axes))
                              for i, axes in d.get("agreement_with", [])],
        )


# ===========================================================================
# Token — per-position record (Layer A + B + C)
# ===========================================================================

@dataclass
class Token:
    """A single tokenised position in the sentence.

    Holds Layer A (morphology, POS), Layer B (dep), Layer C (case /
    role / marker / semantic_role) labels, plus character offsets
    and free-form notes.

    Field design rules:
      - All annotated values use :class:`LabelTag`.
      - Position is 0-based within the sentence's tokens list.
      - ``dep_head_idx`` is 0-based with -1 = unset, -2 = root marker
        (this differs from UD's 1-based 0=root convention; loaders
        translate at ingestion time).
    """
    index: int                          = 0
    surface: str                        = ""
    normalized: str                     = ""
    char_start: int                     = -1
    char_end: int                       = -1

    # Layer A
    morph:           Morphology         = field(default_factory=Morphology)
    pos:             LabelTag           = field(default_factory=LabelTag)

    # Layer B
    dep_head_idx:    int                = -1                        # 0-based; -2 == root
    dep_label:       LabelTag           = field(default_factory=LabelTag)
    governor_pos:    Optional[str]      = None

    # Layer C — iʿrāb fields
    case:            LabelTag           = field(default_factory=LabelTag)
    role:            LabelTag           = field(default_factory=LabelTag)
    marker:          LabelTag           = field(default_factory=LabelTag)

    # Layer C — semantic role (PropBank-style; optional for now)
    semantic_role:   LabelTag           = field(default_factory=LabelTag)

    notes:           List[str]          = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "index": self.index,
            "surface": self.surface,
        }
        if self.normalized and self.normalized != self.surface:
            out["normalized"] = self.normalized
        if self.char_start >= 0: out["char_start"] = self.char_start
        if self.char_end   >= 0: out["char_end"]   = self.char_end
        m = self.morph.to_dict()
        if m: out["morph"] = m
        for k in ("pos", "dep_label", "case", "role", "marker", "semantic_role"):
            v = getattr(self, k)
            if v.is_present or v.alternatives:
                out[k] = v.to_dict()
        if self.dep_head_idx != -1: out["dep_head_idx"] = self.dep_head_idx
        if self.governor_pos:       out["governor_pos"] = self.governor_pos
        if self.notes:              out["notes"]        = list(self.notes)
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Token":
        return cls(
            index=int(d.get("index", 0)),
            surface=d.get("surface", ""),
            normalized=d.get("normalized", d.get("surface", "")),
            char_start=int(d.get("char_start", -1)),
            char_end=int(d.get("char_end", -1)),
            morph=Morphology.from_dict(d.get("morph")),
            pos=LabelTag.from_dict(d.get("pos")),
            dep_head_idx=int(d.get("dep_head_idx", -1)),
            dep_label=LabelTag.from_dict(d.get("dep_label")),
            governor_pos=d.get("governor_pos"),
            case=LabelTag.from_dict(d.get("case")),
            role=LabelTag.from_dict(d.get("role")),
            marker=LabelTag.from_dict(d.get("marker")),
            semantic_role=LabelTag.from_dict(d.get("semantic_role")),
            notes=list(d.get("notes", [])),
        )


# ===========================================================================
# Span — phrase-level (Layer B/C)
# ===========================================================================

@dataclass
class Span:
    span_id:       str             = field(default_factory=lambda: new_id("sp"))
    token_indices: List[int]       = field(default_factory=list)
    span_type:     str             = ""               # "NP", "VP", "PP", "SBAR", ...
    head_idx:      Optional[int]   = None             # 0-based index INTO the sentence (not into token_indices)
    source:        str             = ""
    confidence:    float           = 1.0
    notes:         str             = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not self.notes:    d.pop("notes", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Span":
        return cls(
            span_id=d.get("span_id", new_id("sp")),
            token_indices=list(d.get("token_indices", [])),
            span_type=d.get("span_type", ""),
            head_idx=d.get("head_idx"),
            source=d.get("source", ""),
            confidence=d.get("confidence", 1.0),
            notes=d.get("notes", ""),
        )


# ===========================================================================
# Clause — clause-level (Layer B/C)
# ===========================================================================

@dataclass
class Clause:
    clause_id:        str               = field(default_factory=lambda: new_id("cl"))
    token_indices:    List[int]         = field(default_factory=list)
    clause_type:      str               = ClauseType.UNKNOWN.value
    parent_clause_id: Optional[str]     = None
    head_idx:         Optional[int]     = None
    role_in_parent:   Optional[str]     = None        # what role this clause plays in its parent
    depth:            int               = 0           # 0 = matrix, 1 = once-embedded, etc.
    source:           str               = ""
    confidence:       float             = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Clause":
        return cls(
            clause_id=d.get("clause_id", new_id("cl")),
            token_indices=list(d.get("token_indices", [])),
            clause_type=d.get("clause_type", ClauseType.UNKNOWN.value),
            parent_clause_id=d.get("parent_clause_id"),
            head_idx=d.get("head_idx"),
            role_in_parent=d.get("role_in_parent"),
            depth=int(d.get("depth", 0)),
            source=d.get("source", ""),
            confidence=d.get("confidence", 1.0),
        )


# ===========================================================================
# Construction — replaces flat signature.py pattern (Layer C)
# ===========================================================================

@dataclass
class Construction:
    """A construction occurrence with full structural and ambiguity slots.

    Replaces the flat ``signature.py::ConstructionInstance`` of the
    frozen baseline with explicit clause-membership, children pointers,
    agreement relations, and alternative-analysis tracking.

    ``head_idx`` and ``children_indices`` are token indices (0-based)
    into the parent sentence. ``alternative_analyses`` is a list of
    competing parses for the same span; each is a dict with at least
    ``family``, ``head_idx``, ``children_indices``, ``confidence``.
    """
    construction_id:       str             = field(default_factory=lambda: new_id("cn"))
    family:                str             = ""           # kana_sisters, inna_sisters, ...
    subgroup:              str             = ""           # particle group within family
    token_indices:         List[int]       = field(default_factory=list)
    head_idx:              Optional[int]   = None
    children_indices:      List[int]       = field(default_factory=list)
    particle_surface:      str             = ""
    clause_id:             Optional[str]   = None         # which clause this construction lives in
    semantic_role:         Optional[str]   = None         # event-level role
    # (token_a, token_b, list_of_agreement_axes)
    agreement_relations:   List[Tuple[int, int, List[str]]] = field(default_factory=list)
    ambiguity_score:       float           = 0.0          # 0..1
    alternative_analyses:  List[Dict[str, Any]] = field(default_factory=list)
    source:                str             = ""
    confidence:            float           = 1.0
    notes:                 str             = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["agreement_relations"] = [
            [int(a), int(b), list(axes)]
            for a, b, axes in self.agreement_relations
        ]
        if not self.notes: d.pop("notes", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Construction":
        ar = [(int(a), int(b), list(axes))
              for a, b, axes in d.get("agreement_relations", [])]
        return cls(
            construction_id=d.get("construction_id", new_id("cn")),
            family=d.get("family", ""),
            subgroup=d.get("subgroup", ""),
            token_indices=list(d.get("token_indices", [])),
            head_idx=d.get("head_idx"),
            children_indices=list(d.get("children_indices", [])),
            particle_surface=d.get("particle_surface", ""),
            clause_id=d.get("clause_id"),
            semantic_role=d.get("semantic_role"),
            agreement_relations=ar,
            ambiguity_score=d.get("ambiguity_score", 0.0),
            alternative_analyses=list(d.get("alternative_analyses", [])),
            source=d.get("source", ""),
            confidence=d.get("confidence", 1.0),
            notes=d.get("notes", ""),
        )


# ===========================================================================
# Grammar graph (Step 4 placeholder)
# ===========================================================================

@dataclass
class GraphEdge:
    src_idx:    int                                # token index (source)
    dst_idx:    int                                # token index (destination)
    edge_type:  str            = EdgeType.DEP.value
    label:      str            = ""
    confidence: float          = 1.0
    notes:      str            = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not self.notes:  d.pop("notes", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphEdge":
        return cls(
            src_idx=int(d["src_idx"]), dst_idx=int(d["dst_idx"]),
            edge_type=d.get("edge_type", EdgeType.DEP.value),
            label=d.get("label", ""),
            confidence=d.get("confidence", 1.0),
            notes=d.get("notes", ""),
        )


@dataclass
class GrammarGraph:
    """First-class graph slot.

    Populated by Step 4 (grammar_graph engine). Edges are typed
    using :class:`EdgeType`; multiple edges may exist between the
    same pair of tokens (e.g., a dep edge + an agreement edge).
    """
    edges: List[GraphEdge] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"edges": [e.to_dict() for e in self.edges]}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["GrammarGraph"]:
        if d is None: return None
        return cls(edges=[GraphEdge.from_dict(e) for e in d.get("edges", [])])


# ===========================================================================
# Discourse links (Step 11 placeholder)
# ===========================================================================

@dataclass
class DiscourseLink:
    """Cross-sentence or intra-sentence discourse reference.

    Used for pronoun antecedent resolution, topic continuation, and
    rhetorical relations. Empty list is the norm for single-sentence
    sources; populated by Step 11 ingestion of discourse-annotated
    corpora.
    """
    source_token_idx:    int
    target_sentence_id:  Optional[str] = None
    target_token_idx:    Optional[int] = None
    link_type:           str           = ""    # "antecedent" / "topic_continuation" / "rhetorical"
    rhetorical_relation: str           = ""    # "cause" / "contrast" / "elaboration" / ...
    confidence:          float         = 1.0
    source:              str           = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscourseLink":
        return cls(
            source_token_idx=int(d["source_token_idx"]),
            target_sentence_id=d.get("target_sentence_id"),
            target_token_idx=d.get("target_token_idx"),
            link_type=d.get("link_type", ""),
            rhetorical_relation=d.get("rhetorical_relation", ""),
            confidence=d.get("confidence", 1.0),
            source=d.get("source", ""),
        )


# ===========================================================================
# Reasoning (Step 9 placeholder)
# ===========================================================================

@dataclass
class ReasoningStep:
    """One unit of structured reasoning supervision.

    Placeholder structure: when LLM-generated traces or textbook-
    extracted rationales land later, they slot into existing fields
    without breaking schema compatibility.
    """
    step_idx:                 int          = 0
    applies_to_type:          str          = ""    # "token" / "span" / "construction" / "clause"
    applies_to_id:            str          = ""    # token index (str) or UUID
    justification:            str          = ""
    derivation_chain:         List[str]    = field(default_factory=list)
    alternative_parses:       List[Dict]   = field(default_factory=list)
    ambiguity_notes:          str          = ""
    semantic_disambiguation:  str          = ""
    discourse_notes:          str          = ""
    transformation_logic:     str          = ""
    confidence:               float        = 1.0
    source:                   str          = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReasoningStep":
        return cls(
            step_idx=int(d.get("step_idx", 0)),
            applies_to_type=d.get("applies_to_type", ""),
            applies_to_id=d.get("applies_to_id", ""),
            justification=d.get("justification", ""),
            derivation_chain=list(d.get("derivation_chain", [])),
            alternative_parses=list(d.get("alternative_parses", [])),
            ambiguity_notes=d.get("ambiguity_notes", ""),
            semantic_disambiguation=d.get("semantic_disambiguation", ""),
            discourse_notes=d.get("discourse_notes", ""),
            transformation_logic=d.get("transformation_logic", ""),
            confidence=d.get("confidence", 1.0),
            source=d.get("source", ""),
        )


# ===========================================================================
# Sentence-level metadata
# ===========================================================================

@dataclass
class CurriculumMetadata:
    """Per-sentence metadata used by the Step 7 curriculum scheduler."""
    difficulty_level:           int   = 1   # 1..7 (curriculum stages)
    dependency_depth:           int   = 0
    clause_depth:               int   = 0
    construction_count:         int   = 0
    nested_construction_count:  int   = 0
    ambiguity_score:            float = 0.0
    semantic_pressure_score:    int   = 0   # 0..3
    discourse_complexity:       float = 0.0
    sentence_length_tokens:     int   = 0
    nested_clause_count:        int   = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "CurriculumMetadata":
        if d is None: return cls()
        return cls(**{k: d.get(k, getattr(cls(), k))
                       for k in cls().__dict__.keys()})


@dataclass
class AnnotationCompleteness:
    """Per-sentence flags + per-field completeness percentage.

    The eval engine uses these to stratify metrics by completeness.
    For example: report case_acc separately on sentences with
    has_dep=True vs has_dep=False to disentangle dep-feature
    contribution from raw model accuracy.
    """
    has_morph:                 bool  = False
    has_dep:                   bool  = False
    has_role:                  bool  = False
    has_marker:                bool  = False
    has_constructions:         bool  = False
    has_clauses:               bool  = False
    has_reasoning:             bool  = False
    has_graph:                 bool  = False
    has_discourse:             bool  = False
    has_alternative_parses:    bool  = False
    fields_complete_pct:       float = 0.0     # 0..1, fraction of (case, role, marker) populated

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "AnnotationCompleteness":
        if d is None: return cls()
        return cls(**{k: d.get(k, getattr(cls(), k))
                       for k in cls().__dict__.keys()})


@dataclass
class SentenceMetadata:
    """Provenance + annotation-quality metadata at the sentence level."""
    domain:               str = Domain.UNKNOWN.value
    source:               str = ""                                  # corpus identifier
    source_id:            str = ""                                  # within-corpus id
    annotation_quality:   str = AnnotationQuality.UNKNOWN.value
    parser_origin:        str = ""                                  # primary parser used
    morph_origin:         str = ""
    dep_origin:           str = ""
    role_origin:          str = ""
    marker_origin:        str = ""
    construction_origin:  str = ""
    reasoning_origin:     str = ""
    license:              str = ""
    ingestion_timestamp:  str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "SentenceMetadata":
        if d is None: return cls()
        return cls(**{k: d.get(k, getattr(cls(), k))
                       for k in cls().__dict__.keys()})


# ===========================================================================
# Sentence — top-level container
# ===========================================================================

@dataclass
class Sentence:
    """The complete schema_v2 sentence record.

    Serializes to a single JSONL line. Loaders build instances of
    this class; trainers, evaluators, and indexers consume them.

    Order of fields on disk:

    .. code-block:: text

        schema_version | sentence_id | raw_text | normalized_text |
        tokens | spans | clauses | constructions | graph |
        discourse_links | reasoning_steps | metadata |
        curriculum | completeness
    """
    schema_version:    str                          = SCHEMA_VERSION
    sentence_id:       str                          = field(default_factory=lambda: new_id("s"))
    raw_text:          str                          = ""
    normalized_text:   str                          = ""

    tokens:            List[Token]                  = field(default_factory=list)
    spans:             List[Span]                   = field(default_factory=list)
    clauses:           List[Clause]                 = field(default_factory=list)
    constructions:     List[Construction]           = field(default_factory=list)
    graph:             Optional[GrammarGraph]       = None
    discourse_links:   List[DiscourseLink]          = field(default_factory=list)
    reasoning_steps:   List[ReasoningStep]          = field(default_factory=list)

    metadata:          SentenceMetadata             = field(default_factory=SentenceMetadata)
    curriculum:        CurriculumMetadata           = field(default_factory=CurriculumMetadata)
    completeness:      AnnotationCompleteness       = field(default_factory=AnnotationCompleteness)

    # Convenience accessors --------------------------------------------------

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def token_at(self, idx: int) -> Token:
        return self.tokens[idx]

    def constructions_of_family(self, family: str) -> List[Construction]:
        return [c for c in self.constructions if c.family == family]

    def constructions_at(self, token_idx: int) -> List[Construction]:
        return [c for c in self.constructions if token_idx in c.token_indices]

    def has_construction_family(self, family: str) -> bool:
        return any(c.family == family for c in self.constructions)

    # (de)serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version":  self.schema_version,
            "sentence_id":     self.sentence_id,
            "raw_text":        self.raw_text,
            "normalized_text": self.normalized_text,
            "tokens":          [t.to_dict() for t in self.tokens],
            "metadata":        self.metadata.to_dict(),
            "curriculum":      self.curriculum.to_dict(),
            "completeness":    self.completeness.to_dict(),
        }
        if self.spans:
            out["spans"] = [s.to_dict() for s in self.spans]
        if self.clauses:
            out["clauses"] = [c.to_dict() for c in self.clauses]
        if self.constructions:
            out["constructions"] = [c.to_dict() for c in self.constructions]
        if self.graph is not None:
            out["graph"] = self.graph.to_dict()
        if self.discourse_links:
            out["discourse_links"] = [d.to_dict() for d in self.discourse_links]
        if self.reasoning_steps:
            out["reasoning_steps"] = [r.to_dict() for r in self.reasoning_steps]
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Sentence":
        version = d.get("schema_version", "1.0.0")
        if version != SCHEMA_VERSION:
            d = _migrate(d, version)
        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            sentence_id=d.get("sentence_id", new_id("s")),
            raw_text=d.get("raw_text", ""),
            normalized_text=d.get("normalized_text", ""),
            tokens=[Token.from_dict(t) for t in d.get("tokens", [])],
            spans=[Span.from_dict(s) for s in d.get("spans", [])],
            clauses=[Clause.from_dict(c) for c in d.get("clauses", [])],
            constructions=[Construction.from_dict(c)
                           for c in d.get("constructions", [])],
            graph=GrammarGraph.from_dict(d.get("graph")),
            discourse_links=[DiscourseLink.from_dict(x)
                             for x in d.get("discourse_links", [])],
            reasoning_steps=[ReasoningStep.from_dict(r)
                             for r in d.get("reasoning_steps", [])],
            metadata=SentenceMetadata.from_dict(d.get("metadata")),
            curriculum=CurriculumMetadata.from_dict(d.get("curriculum")),
            completeness=AnnotationCompleteness.from_dict(d.get("completeness")),
        )

    @classmethod
    def from_json(cls, s: str) -> "Sentence":
        return cls.from_dict(json.loads(s))


# ===========================================================================
# Migration hook
# ===========================================================================

def _migrate(d: Dict[str, Any], from_version: str) -> Dict[str, Any]:
    """Migrate older schema versions forward.

    Currently a no-op stub (no older versions exist on disk yet).
    When a new schema version is introduced, add migration logic here.
    """
    # placeholder; implement when 2.x → 3.x migrations are needed
    return d


# ===========================================================================
# JSONL helpers
# ===========================================================================

def write_jsonl(path: str, sentences: Iterable[Sentence]) -> None:
    """Write sentences to a JSONL file (one Sentence per line)."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for s in sentences:
            fh.write(s.to_json() + "\n")


def read_jsonl(path: str) -> Iterator[Sentence]:
    """Stream sentences from a JSONL file."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield Sentence.from_json(line)
