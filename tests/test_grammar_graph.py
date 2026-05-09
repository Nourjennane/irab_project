"""Tests for the grammar_graph engine (Step 4)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from irab_tashkeel.data_v2.constructions.detector import detect_constructions_pass
from irab_tashkeel.data_v2.loaders.gazelle import GazelleLoader
from irab_tashkeel.data_v2.metadata import difficulty
from irab_tashkeel.data_v2.schema_v2 import (
    Clause, ClauseType, Construction, DiscourseLink, LabelTag, Morphology,
    Sentence, Span, Token,
)
from irab_tashkeel.grammar_graph import (
    EDGE_TYPES, NODE_TYPES, SentenceGraph,
    batch_sparse, bridge_to_schema, build_sentence_graph,
    constructions_at_token, filter_by_curriculum, graph_depth,
    overlap_constructions, read_jsonl, shortest_path, to_sparse,
    walk_to_governor, write_jsonl,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_node_and_edge_type_constants():
    assert "token" in NODE_TYPES
    assert "construction" in NODE_TYPES
    assert "clause" in NODE_TYPES
    assert "phrase" in NODE_TYPES
    assert "discourse" in NODE_TYPES
    assert "dep" in EDGE_TYPES
    assert "agreement" in EDGE_TYPES
    assert "construction_member" in EDGE_TYPES
    assert "clause_member" in EDGE_TYPES
    assert "discourse_link" in EDGE_TYPES
    assert "coref" in EDGE_TYPES
    assert "semantic_link" in EDGE_TYPES


def test_empty_graph():
    g = SentenceGraph(sentence_id="s1")
    assert g.n_nodes() == 0
    assert g.n_edges() == 0
    assert graph_depth(g) == 0


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _make_kana_sentence() -> Sentence:
    return Sentence(
        raw_text="كان الطالب مجتهداً",
        tokens=[
            Token(index=0, surface="كان", dep_head_idx=-2,        # root
                  pos=LabelTag(value="verb", source="gold_human")),
            Token(index=1, surface="الطالب", dep_head_idx=0,       # head: kana
                  pos=LabelTag(value="noun", source="gold_human"),
                  dep_label=LabelTag(value="nsubj", source="gold_human", confidence=1.0)),
            Token(index=2, surface="مجتهداً", dep_head_idx=0,
                  pos=LabelTag(value="adjective", source="gold_human"),
                  dep_label=LabelTag(value="xcomp", source="gold_human", confidence=1.0)),
        ],
        constructions=[
            Construction(
                family="kana_sisters", subgroup="kana_completion",
                token_indices=[0, 1, 2], head_idx=0,
                particle_surface="كان",
                source="bronze_heuristic", confidence=0.9,
            ),
        ],
        clauses=[
            Clause(token_indices=[0, 1, 2], clause_type=ClauseType.MATRIX.value,
                   depth=0, source="gold_human"),
        ],
    )


def test_build_kana_sentence_graph():
    s = _make_kana_sentence()
    g = build_sentence_graph(s)
    assert g.sentence_id == s.sentence_id
    counts = g.node_type_counts()
    assert counts.get("token") == 3
    assert counts.get("construction") == 1
    assert counts.get("clause") == 1
    edge_counts = g.edge_type_counts()
    # 2 dep edges (head 1→0 and 2→0)
    assert edge_counts.get("dep") == 2
    # 3 construction_member edges (one per token in span)
    assert edge_counts.get("construction_member") == 3
    # 3 clause_member edges (one per token in clause)
    assert edge_counts.get("clause_member") == 3


def test_build_dep_root_skipped():
    """Root tokens (dep_head_idx == -2) should not produce a dep edge."""
    s = _make_kana_sentence()
    g = build_sentence_graph(s)
    # token 0 is the root; should have no outgoing dep edge
    out_dep = g.out_edges("t:" + s.sentence_id + ":0", edge_type="dep")
    assert len(out_dep) == 0


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def test_constructions_at_token():
    s = _make_kana_sentence()
    g = build_sentence_graph(s)
    # token 1 is in the kana construction
    cs = constructions_at_token(g, 1)
    assert len(cs) == 1


def test_overlap_constructions_no_overlap():
    s = _make_kana_sentence()
    g = build_sentence_graph(s)
    overlaps = overlap_constructions(g)
    # Single construction → no overlap
    assert overlaps == {}


def test_overlap_constructions_with_overlap():
    s = _make_kana_sentence()
    # add an iḍāfa construction overlapping with kana
    s.constructions.append(Construction(
        family="idafa", subgroup="any",
        token_indices=[1, 2], head_idx=1,
        source="bronze_heuristic", confidence=0.7,
    ))
    g = build_sentence_graph(s)
    overlaps = overlap_constructions(g)
    # tokens 1 and 2 should each be covered by both constructions
    assert 1 in overlaps and len(overlaps[1]) == 2
    assert 2 in overlaps and len(overlaps[2]) == 2


def test_walk_to_governor():
    s = _make_kana_sentence()
    g = build_sentence_graph(s)
    chain = walk_to_governor(g, 1, max_hops=3)
    # token 1 → token 0 (head); chain length 2
    assert len(chain) == 2


def test_shortest_path_dep_only():
    s = _make_kana_sentence()
    g = build_sentence_graph(s)
    sid = s.sentence_id
    p = shortest_path(g, f"t:{sid}:1", f"t:{sid}:2", edge_types=["dep"])
    # Both 1 and 2 head to 0, so path is 1 → 0 → 2
    assert p is not None
    assert len(p) == 3


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_filter_by_curriculum_token_only():
    s = _make_kana_sentence()
    g = build_sentence_graph(s)
    g_tokens = filter_by_curriculum(g, keep_node_types=["token"])
    assert g_tokens.node_type_counts() == {"token": 3}
    # Only dep edges remain (since other node types dropped)
    assert all(e.edge_type == "dep" for e in g_tokens.edges)


def test_filter_drop_ambiguous_constructions():
    s = _make_kana_sentence()
    s.constructions[0].ambiguity_score = 0.5
    g = build_sentence_graph(s)
    filtered = filter_by_curriculum(g, drop_ambiguous_constructions=True)
    assert filtered.node_type_counts().get("construction", 0) == 0


# ---------------------------------------------------------------------------
# Sparse export
# ---------------------------------------------------------------------------

def test_sparse_export_shape():
    s = _make_kana_sentence()
    g = build_sentence_graph(s)
    sp = to_sparse(g)
    assert sp.edge_index.shape[0] == 2
    assert sp.edge_index.shape[1] == g.n_edges()
    assert sp.node_type.shape[0] == g.n_nodes()
    assert sp.edge_type.shape[0] == g.n_edges()


def test_batch_sparse_offsets():
    s1 = _make_kana_sentence()
    s2 = _make_kana_sentence()
    s2.sentence_id = "s_other"
    g1 = build_sentence_graph(s1); sp1 = to_sparse(g1)
    g2 = build_sentence_graph(s2); sp2 = to_sparse(g2)
    batch = batch_sparse([sp1, sp2])
    # batch array length == sum of nodes
    assert batch["batch"].shape[0] == g1.n_nodes() + g2.n_nodes()
    # second-graph node ids in edge_index should be offset by g1.n_nodes()
    assert batch["edge_index"].shape[1] == g1.n_edges() + g2.n_edges()
    assert batch["n_graphs"] == 2


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_jsonl_round_trip(tmp_path):
    s = _make_kana_sentence()
    g = build_sentence_graph(s)
    p = tmp_path / "graphs.jsonl"
    write_jsonl(p, [g])
    gs = list(read_jsonl(p))
    assert len(gs) == 1
    assert gs[0].n_nodes() == g.n_nodes()
    assert gs[0].n_edges() == g.n_edges()


def test_bridge_to_schema_drops_non_token_edges():
    s = _make_kana_sentence()
    g = build_sentence_graph(s)
    sg = bridge_to_schema(g)
    # only dep + agreement between tokens; construction_member dropped
    for e in sg.edges:
        assert e.edge_type in ("dep", "agreement")


# ---------------------------------------------------------------------------
# End-to-end on Gazelle
# ---------------------------------------------------------------------------

def test_end_to_end_gazelle():
    loader = GazelleLoader(root=str(ROOT))
    sents = loader.load_all()
    detect_constructions_pass(sents)
    difficulty.populate_all(sents)

    graphs = [build_sentence_graph(s) for s in sents]
    assert len(graphs) == len(sents)
    # at least one graph should have a construction node
    assert any(g.nodes_of_type("construction") for g in graphs)
