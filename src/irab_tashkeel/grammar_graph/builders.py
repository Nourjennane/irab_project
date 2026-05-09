"""Graph builders — convert :class:`schema_v2.Sentence` → :class:`SentenceGraph`.

The builder is deterministic and stateless. Calling
``build_sentence_graph(s)`` produces the same graph regardless of
order. Empty schema_v2 fields produce empty graph regions; e.g., a
sentence with no clauses produces no clause nodes (and no
clause_member edges).

Produced edges
--------------

================== ===========================================
edge type           rule
================== ===========================================
dep                token → token where dep_head_idx is set
agreement          token ↔ token via Morphology.agreement_with
clause_member      token → clause (its container)
                   construction → clause (when c.clause_id is set)
construction_member token → construction (members of token_indices)
semantic_link      empty for now (Layer C semantic_role placeholder)
discourse_link     emitted from sentence.discourse_links
coref              empty placeholder
================== ===========================================

Nodes: one ``token`` per Token, one ``construction`` per Construction,
one ``clause`` per Clause, one ``phrase`` per Span, plus one
``discourse`` node per DiscourseLink endpoint that points OUT of
this sentence (in-sentence discourse refs use the existing token
nodes as endpoints).
"""
from __future__ import annotations

from typing import Optional

from ..data_v2.schema_v2 import (
    Construction, Sentence, Span, Clause, DiscourseLink, Token,
)
from .graph import EDGE_TYPES, GraphEdge, GraphNode, SentenceGraph


# ===========================================================================
# Node id helpers
# ===========================================================================

def _token_id(sentence_id: str, idx: int) -> str:
    return f"t:{sentence_id}:{idx}"


def _construction_id(sentence_id: str, c: Construction) -> str:
    return f"cn:{sentence_id}:{c.construction_id}"


def _clause_id(sentence_id: str, c: Clause) -> str:
    return f"cl:{sentence_id}:{c.clause_id}"


def _phrase_id(sentence_id: str, s: Span) -> str:
    return f"ph:{sentence_id}:{s.span_id}"


def _discourse_id(sentence_id: str, link_idx: int) -> str:
    return f"dc:{sentence_id}:{link_idx}"


# ===========================================================================
# Token / construction / clause / phrase node builders
# ===========================================================================

def _add_token_nodes(g: SentenceGraph, s: Sentence) -> None:
    sid = s.sentence_id
    for t in s.tokens:
        node = GraphNode(
            node_id=_token_id(sid, t.index),
            node_type="token",
            schema_ref=str(t.index),
            sentence_id=sid,
            metadata={
                "surface": t.surface,
                "normalized": t.normalized,
                "case": t.case.value, "role": t.role.value,
                "marker": t.marker.value, "pos": t.pos.value,
                "case_conf": t.case.confidence,
                "role_conf": t.role.confidence,
                "marker_conf": t.marker.confidence,
                "morph": {
                    "gender": t.morph.gender.value,
                    "number": t.morph.number.value,
                    "person": t.morph.person.value,
                    "definite": t.morph.definite.value,
                },
            },
        )
        g.add_node(node)


def _add_construction_nodes(g: SentenceGraph, s: Sentence) -> None:
    sid = s.sentence_id
    for c in s.constructions:
        node = GraphNode(
            node_id=_construction_id(sid, c),
            node_type="construction",
            schema_ref=c.construction_id,
            sentence_id=sid,
            metadata={
                "family": c.family,
                "subgroup": c.subgroup,
                "head_idx": c.head_idx,
                "particle_surface": c.particle_surface,
                "ambiguity_score": c.ambiguity_score,
                "n_alternatives": len(c.alternative_analyses),
                "confidence": c.confidence,
            },
        )
        g.add_node(node)


def _add_clause_nodes(g: SentenceGraph, s: Sentence) -> None:
    sid = s.sentence_id
    for cl in s.clauses:
        node = GraphNode(
            node_id=_clause_id(sid, cl),
            node_type="clause",
            schema_ref=cl.clause_id,
            sentence_id=sid,
            metadata={
                "clause_type": cl.clause_type,
                "depth": cl.depth,
                "head_idx": cl.head_idx,
                "role_in_parent": cl.role_in_parent,
                "parent_clause_id": cl.parent_clause_id,
            },
        )
        g.add_node(node)


def _add_phrase_nodes(g: SentenceGraph, s: Sentence) -> None:
    sid = s.sentence_id
    for sp in s.spans:
        node = GraphNode(
            node_id=_phrase_id(sid, sp),
            node_type="phrase",
            schema_ref=sp.span_id,
            sentence_id=sid,
            metadata={
                "span_type": sp.span_type,
                "head_idx": sp.head_idx,
                "confidence": sp.confidence,
            },
        )
        g.add_node(node)


def _add_discourse_nodes_and_edges(g: SentenceGraph, s: Sentence) -> None:
    sid = s.sentence_id
    for i, dl in enumerate(s.discourse_links):
        node_id = _discourse_id(sid, i)
        node = GraphNode(
            node_id=node_id, node_type="discourse",
            schema_ref=str(i), sentence_id=sid,
            metadata={
                "link_type": dl.link_type,
                "rhetorical_relation": dl.rhetorical_relation,
                "target_sentence_id": dl.target_sentence_id,
                "target_token_idx": dl.target_token_idx,
            },
        )
        g.add_node(node)
        # Edge from the source token to the discourse-target placeholder.
        src = _token_id(sid, dl.source_token_idx)
        if src in g.nodes:
            g.add_edge(GraphEdge(
                src=src, dst=node_id, edge_type="discourse_link",
                label=dl.link_type, confidence=dl.confidence,
                metadata={"rhetorical_relation": dl.rhetorical_relation},
            ))


# ===========================================================================
# Edge builders
# ===========================================================================

def _add_dep_edges(g: SentenceGraph, s: Sentence) -> None:
    sid = s.sentence_id
    for t in s.tokens:
        if t.dep_head_idx is None or t.dep_head_idx < 0:
            continue
        # head_idx == -2 means root; we don't add a root edge
        if t.dep_head_idx == -2:
            continue
        head_node = _token_id(sid, t.dep_head_idx)
        if head_node not in g.nodes:
            continue
        g.add_edge(GraphEdge(
            src=_token_id(sid, t.index),
            dst=head_node,
            edge_type="dep",
            label=t.dep_label.value or "",
            confidence=t.dep_label.confidence if t.dep_label.is_present else 0.5,
            metadata={"governor_pos": t.governor_pos},
        ))


def _add_agreement_edges(g: SentenceGraph, s: Sentence) -> None:
    sid = s.sentence_id
    seen = set()
    for t in s.tokens:
        for (other_idx, axes) in t.morph.agreement_with:
            pair = tuple(sorted([t.index, other_idx]))
            if pair in seen:
                continue
            seen.add(pair)
            src = _token_id(sid, t.index)
            dst = _token_id(sid, other_idx)
            if src not in g.nodes or dst not in g.nodes:
                continue
            g.add_edge(GraphEdge(
                src=src, dst=dst, edge_type="agreement",
                label="+".join(axes), confidence=1.0,
                metadata={"axes": list(axes)},
            ))


def _add_construction_edges(g: SentenceGraph, s: Sentence) -> None:
    sid = s.sentence_id
    for c in s.constructions:
        cn_id = _construction_id(sid, c)
        for tok_idx in c.token_indices:
            t_id = _token_id(sid, tok_idx)
            if t_id in g.nodes:
                g.add_edge(GraphEdge(
                    src=t_id, dst=cn_id,
                    edge_type="construction_member",
                    label=c.family,
                    confidence=c.confidence,
                    metadata={"is_head": tok_idx == c.head_idx},
                ))
        # construction → clause if assigned
        if c.clause_id:
            from .builders import _clause_id_lookup
            cl_id = _clause_id_lookup(sid, c.clause_id, g)
            if cl_id and cl_id in g.nodes:
                g.add_edge(GraphEdge(
                    src=cn_id, dst=cl_id,
                    edge_type="clause_member",
                    label=c.family, confidence=c.confidence,
                ))
        # Agreement relations declared at the construction level
        for (a_idx, b_idx, axes) in c.agreement_relations:
            a = _token_id(sid, a_idx); b = _token_id(sid, b_idx)
            if a in g.nodes and b in g.nodes:
                g.add_edge(GraphEdge(
                    src=a, dst=b, edge_type="agreement",
                    label="+".join(axes), confidence=c.confidence,
                    metadata={"axes": list(axes), "from_construction": c.construction_id},
                ))


def _clause_id_lookup(sid: str, schema_clause_id: str, g: SentenceGraph) -> Optional[str]:
    """Resolve a schema_v2 clause UUID to its in-graph node_id."""
    target = f"cl:{sid}:{schema_clause_id}"
    return target if target in g.nodes else None


def _add_clause_member_edges(g: SentenceGraph, s: Sentence) -> None:
    sid = s.sentence_id
    for cl in s.clauses:
        cl_node = _clause_id(sid, cl)
        if cl_node not in g.nodes:
            continue
        for tok_idx in cl.token_indices:
            t_id = _token_id(sid, tok_idx)
            if t_id in g.nodes:
                g.add_edge(GraphEdge(
                    src=t_id, dst=cl_node,
                    edge_type="clause_member",
                    label=cl.clause_type, confidence=cl.confidence,
                ))
        # clause → parent clause
        if cl.parent_clause_id:
            parent = f"cl:{sid}:{cl.parent_clause_id}"
            if parent in g.nodes:
                g.add_edge(GraphEdge(
                    src=cl_node, dst=parent,
                    edge_type="clause_member",
                    label="parent", confidence=1.0,
                    metadata={"role_in_parent": cl.role_in_parent},
                ))


def _add_phrase_member_edges(g: SentenceGraph, s: Sentence) -> None:
    sid = s.sentence_id
    for sp in s.spans:
        ph_id = _phrase_id(sid, sp)
        if ph_id not in g.nodes:
            continue
        for tok_idx in sp.token_indices:
            t_id = _token_id(sid, tok_idx)
            if t_id in g.nodes:
                g.add_edge(GraphEdge(
                    src=t_id, dst=ph_id,
                    edge_type="construction_member",
                    label=f"phrase:{sp.span_type}",
                    confidence=sp.confidence,
                    metadata={"phrase": True},
                ))


def _add_semantic_link_edges(g: SentenceGraph, s: Sentence) -> None:
    """Emit semantic_link edges from token.semantic_role values.

    Currently empty for all loaders; placeholder for when Arabic
    PropBank / SALMA-style semantic-role labels land.
    """
    sid = s.sentence_id
    for t in s.tokens:
        if not t.semantic_role.is_present:
            continue
        # We don't yet have predicate-argument structure on the
        # schema; emit a self-loop placeholder so the node carries
        # the semantic_role label as a graph fact.
        node_id = _token_id(sid, t.index)
        if node_id in g.nodes:
            g.add_edge(GraphEdge(
                src=node_id, dst=node_id,
                edge_type="semantic_link",
                label=t.semantic_role.value,
                confidence=t.semantic_role.confidence,
            ))


# ===========================================================================
# Public API
# ===========================================================================

def build_sentence_graph(s: Sentence) -> SentenceGraph:
    """Build a per-sentence grammar graph from a schema_v2 Sentence."""
    g = SentenceGraph(sentence_id=s.sentence_id)
    _add_token_nodes(g, s)
    _add_clause_nodes(g, s)
    _add_construction_nodes(g, s)
    _add_phrase_nodes(g, s)
    _add_discourse_nodes_and_edges(g, s)

    _add_dep_edges(g, s)
    _add_agreement_edges(g, s)
    _add_construction_edges(g, s)
    _add_clause_member_edges(g, s)
    _add_phrase_member_edges(g, s)
    _add_semantic_link_edges(g, s)
    return g


def build_corpus_graphs(sentences) -> "list[SentenceGraph]":
    return [build_sentence_graph(s) for s in sentences]
