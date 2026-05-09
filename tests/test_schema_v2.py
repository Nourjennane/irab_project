"""Schema v2 unit tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from irab_tashkeel.data_v2.schema_v2 import (
    AnnotationCompleteness, AnnotationQuality, ClauseType, Clause,
    Construction, CurriculumMetadata, DiscourseLink, Domain, EdgeType,
    GraphEdge, GrammarGraph, LabelTag, Morphology, ReasoningStep,
    SCHEMA_VERSION, Sentence, SentenceMetadata, Span, Token, new_id,
    write_jsonl, read_jsonl,
)


# ---------------------------------------------------------------------------
# LabelTag basics
# ---------------------------------------------------------------------------

def test_labeltag_default_is_empty():
    lt = LabelTag()
    assert lt.value is None
    assert not lt.is_present


def test_labeltag_round_trip():
    lt = LabelTag(value="raf", source="gold_human", confidence=1.0,
                  alternatives=[("nasb", 0.1)], notes="test")
    d = lt.to_dict()
    lt2 = LabelTag.from_dict(d)
    assert lt2.value == "raf"
    assert lt2.source == "gold_human"
    assert lt2.alternatives == [("nasb", 0.1)]


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------

def test_morphology_empty_serialises_compactly():
    m = Morphology()
    assert m.to_dict() == {}


def test_morphology_round_trip():
    m = Morphology(
        gender=LabelTag(value="masc", source="ud_padt"),
        number=LabelTag(value="sg", source="ud_padt"),
        agreement_with=[(2, ["gender", "number"])],
    )
    d = m.to_dict()
    m2 = Morphology.from_dict(d)
    assert m2.gender.value == "masc"
    assert m2.agreement_with == [(2, ["gender", "number"])]


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

def test_token_round_trip():
    t = Token(
        index=1, surface="الطالب", normalized="الطالب",
        char_start=4, char_end=10,
        pos=LabelTag(value="noun", source="gold_human"),
        case=LabelTag(value="raf", source="gold_human"),
        role=LabelTag(value="ism_kana", source="gold_human"),
        marker=LabelTag(value="damma_visible", source="gold_human"),
        dep_head_idx=0,
        dep_label=LabelTag(value="nsubj", source="stanza_ud", confidence=0.9),
    )
    d = t.to_dict()
    t2 = Token.from_dict(d)
    assert t2.index == 1
    assert t2.role.value == "ism_kana"
    assert t2.dep_head_idx == 0
    assert t2.dep_label.confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_round_trip():
    c = Construction(
        family="kana_sisters", subgroup="kana_completion",
        token_indices=[0, 1, 2], head_idx=0, particle_surface="كان",
        agreement_relations=[(1, 2, ["case"])],
        ambiguity_score=0.15,
        alternative_analyses=[{"family": "mubtada_khabar", "confidence": 0.3}],
    )
    d = c.to_dict()
    c2 = Construction.from_dict(d)
    assert c2.family == "kana_sisters"
    assert c2.head_idx == 0
    assert c2.agreement_relations == [(1, 2, ["case"])]
    assert c2.alternative_analyses[0]["family"] == "mubtada_khabar"


# ---------------------------------------------------------------------------
# Sentence — full round trip
# ---------------------------------------------------------------------------

def _kana_example() -> Sentence:
    return Sentence(
        raw_text="كان الطالب مجتهداً",
        normalized_text="كان الطالب مجتهدا",
        tokens=[
            Token(index=0, surface="كان", normalized="كان",
                  case=LabelTag(value="mabni", source="gold_human"),
                  role=LabelTag(value="fil", source="gold_human"),
                  marker=LabelTag(value="fath_short", source="gold_human")),
            Token(index=1, surface="الطالب", normalized="الطالب",
                  case=LabelTag(value="raf", source="gold_human"),
                  role=LabelTag(value="ism_kana", source="gold_human"),
                  marker=LabelTag(value="damma_visible", source="gold_human")),
            Token(index=2, surface="مجتهداً", normalized="مجتهدا",
                  case=LabelTag(value="nasb", source="gold_human"),
                  role=LabelTag(value="khabar_kana", source="gold_human"),
                  marker=LabelTag(value="tanween_fath", source="gold_human")),
        ],
        constructions=[
            Construction(family="kana_sisters", subgroup="kana_completion",
                         token_indices=[0, 1, 2], head_idx=0,
                         particle_surface="كان", source="gold_human"),
        ],
        metadata=SentenceMetadata(
            domain=Domain.MSA_NEWS.value, source="manual_test",
            annotation_quality=AnnotationQuality.GOLD_HUMAN.value,
        ),
    )


def test_sentence_round_trip():
    s = _kana_example()
    d = s.to_dict()
    s2 = Sentence.from_dict(d)
    assert s2.n_tokens == 3
    assert s2.tokens[1].role.value == "ism_kana"
    assert s2.tokens[2].marker.value == "tanween_fath"
    assert s2.constructions_of_family("kana_sisters")[0].particle_surface == "كان"
    assert s2.has_construction_family("kana_sisters")
    assert not s2.has_construction_family("inna_sisters")


def test_sentence_jsonl_round_trip(tmp_path):
    s = _kana_example()
    p = tmp_path / "test.jsonl"
    write_jsonl(str(p), [s])
    sents = list(read_jsonl(str(p)))
    assert len(sents) == 1
    assert sents[0].tokens[1].role.value == "ism_kana"


def test_sentence_schema_version_pinned():
    s = Sentence()
    assert s.schema_version == SCHEMA_VERSION
    assert SCHEMA_VERSION == "2.0.0"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def test_graph_round_trip():
    g = GrammarGraph(edges=[
        GraphEdge(src_idx=0, dst_idx=1, edge_type=EdgeType.DEP.value, label="nsubj"),
        GraphEdge(src_idx=1, dst_idx=2, edge_type=EdgeType.AGREEMENT.value,
                  label="gender+number"),
    ])
    d = g.to_dict()
    g2 = GrammarGraph.from_dict(d)
    assert len(g2.edges) == 2
    assert g2.edges[0].label == "nsubj"


# ---------------------------------------------------------------------------
# Empty sentence still serializes
# ---------------------------------------------------------------------------

def test_empty_sentence_serializes():
    s = Sentence()
    d = s.to_dict()
    s2 = Sentence.from_dict(d)
    assert s2.n_tokens == 0
    assert s2.schema_version == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# IDs are unique
# ---------------------------------------------------------------------------

def test_ids_are_unique():
    ids = {new_id("s") for _ in range(1000)}
    assert len(ids) == 1000
