"""Graph queries — depth, paths, overlaps, curriculum-aware filtering.

All queries operate on a built :class:`SentenceGraph`. They are
read-only; mutations go through :mod:`builders`.

Available queries
-----------------

- ``graph_depth(g, root_type="clause")`` — max BFS depth from
  matrix-clause roots
- ``shortest_path(g, src, dst, edge_types=None)`` — shortest path
  by edge count, optionally restricted to certain edge types
- ``constructions_at_token(g, token_idx)`` — list of construction
  node_ids covering a given token
- ``overlap_constructions(g)`` — token indices where ≥ 2
  constructions overlap, plus the set of constructions per index
- ``walk_to_governor(g, token_idx, max_hops=3)`` — climb dep edges
  to find a token's governor chain
- ``find_construction_clause_membership(g)`` — for each construction
  node, return its containing clause if any
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .graph import EDGE_TYPES, GraphEdge, GraphNode, SentenceGraph


# ===========================================================================
# Depth
# ===========================================================================

def graph_depth(g: SentenceGraph, root_type: str = "clause") -> int:
    """Max BFS depth from any node of ``root_type``.

    Falls back to depth from any source-like node if no nodes of
    ``root_type`` exist (e.g., when clauses are not yet populated).

    Returns 0 for empty graphs.
    """
    roots = [n.node_id for n in g.nodes.values() if n.node_type == root_type]
    if not roots:
        # fallback: nodes with no incoming edges
        roots = [nid for nid in g.nodes
                 if not g.in_edges(nid)]
    if not roots:
        return 0
    seen: Set[str] = set()
    max_d = 0
    for root in roots:
        q = deque([(root, 0)])
        while q:
            n, d = q.popleft()
            if n in seen: continue
            seen.add(n)
            max_d = max(max_d, d)
            for nb in g.neighbours(n, direction="out"):
                if nb not in seen:
                    q.append((nb, d + 1))
    return max_d


# ===========================================================================
# Shortest path
# ===========================================================================

def shortest_path(
    g: SentenceGraph, src: str, dst: str,
    edge_types: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """BFS shortest path from ``src`` to ``dst`` over selected edges.

    Returns a list of node_ids ``[src, ..., dst]`` or ``None`` if no
    path exists. ``edge_types=None`` means all types.
    """
    if src not in g.nodes or dst not in g.nodes:
        return None
    if src == dst:
        return [src]
    et: Optional[Set[str]] = None
    if edge_types is not None:
        et = set(edge_types)
        bad = et - set(EDGE_TYPES)
        if bad:
            raise ValueError(f"unknown edge types: {bad}")

    parents: Dict[str, Optional[str]] = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        for e in g._out.get(cur, []) + g._in.get(cur, []):
            if et is not None and e.edge_type not in et:
                continue
            other = e.dst if e.src == cur else e.src
            if other in parents: continue
            parents[other] = cur
            if other == dst:
                # reconstruct
                path = [dst]
                p = cur
                while p is not None:
                    path.append(p)
                    p = parents[p]
                return list(reversed(path))
            q.append(other)
    return None


# ===========================================================================
# Construction overlap queries
# ===========================================================================

def constructions_at_token(g: SentenceGraph, token_idx: int) -> List[str]:
    """Return construction node_ids that cover a given token index."""
    out: List[str] = []
    t_id = f"t:{g.sentence_id}:{token_idx}"
    if t_id not in g.nodes:
        return out
    for e in g.out_edges(t_id, edge_type="construction_member"):
        n = g.nodes.get(e.dst)
        if n and n.node_type == "construction":
            out.append(e.dst)
    return out


def overlap_constructions(g: SentenceGraph) -> Dict[int, List[str]]:
    """Token-index → list of construction node_ids that cover it.

    Only returns entries where ≥ 2 constructions cover the token
    (the cases worth surfacing for ambiguity resolution).
    """
    out: Dict[int, List[str]] = {}
    token_nodes = [(int(n.schema_ref), n.node_id)
                   for n in g.nodes_of_type("token")]
    for idx, t_id in token_nodes:
        cons = []
        for e in g.out_edges(t_id, edge_type="construction_member"):
            n = g.nodes.get(e.dst)
            if n and n.node_type == "construction":
                cons.append(e.dst)
        if len(cons) >= 2:
            out[idx] = cons
    return out


# ===========================================================================
# Governor traversal
# ===========================================================================

def walk_to_governor(g: SentenceGraph, token_idx: int,
                      max_hops: int = 3) -> List[str]:
    """Climb dep edges from a token to its governor chain.

    Returns a list of token node_ids in walked order. Stops at the
    sentence root or after ``max_hops`` steps.
    """
    out: List[str] = []
    cur = f"t:{g.sentence_id}:{token_idx}"
    if cur not in g.nodes:
        return out
    out.append(cur)
    for _ in range(max_hops):
        # The 'dep' edge points from dependent → head. Follow outgoing.
        outs = g.out_edges(cur, edge_type="dep")
        if not outs:
            break
        # Prefer first dep edge (most often there's only one)
        nxt = outs[0].dst
        if nxt in out:
            break
        out.append(nxt)
        cur = nxt
    return out


# ===========================================================================
# Construction-clause membership
# ===========================================================================

def find_construction_clause_membership(g: SentenceGraph) -> Dict[str, Optional[str]]:
    """For each construction node, return its containing clause node_id (or None)."""
    out: Dict[str, Optional[str]] = {}
    for c_node in g.nodes_of_type("construction"):
        target = None
        for e in g.out_edges(c_node.node_id, edge_type="clause_member"):
            n = g.nodes.get(e.dst)
            if n and n.node_type == "clause":
                target = e.dst
                break
        out[c_node.node_id] = target
    return out


# ===========================================================================
# Curriculum-aware filtering
# ===========================================================================

def filter_by_curriculum(
    g: SentenceGraph,
    *,
    keep_node_types: Optional[List[str]] = None,
    keep_edge_types: Optional[List[str]] = None,
    min_construction_confidence: float = 0.0,
    drop_ambiguous_constructions: bool = False,
) -> SentenceGraph:
    """Return a *new* SentenceGraph keeping only the requested
    nodes / edges. Curriculum-aware because:

    - Stage 1 morph-only training can request
      ``keep_node_types=["token"]`` to drop higher-level nodes.
    - Stage 4 nested-syntax training can request
      ``keep_edge_types=["dep","clause_member","construction_member"]``.
    - Setting ``drop_ambiguous_constructions=True`` removes
      constructions with ``ambiguity_score ≥ 0.3`` — useful for
      Stage-3 simple-construction training.
    """
    out = SentenceGraph(sentence_id=g.sentence_id)
    keep_n = set(keep_node_types) if keep_node_types is not None else None
    keep_e = set(keep_edge_types) if keep_edge_types is not None else None

    for n in g.nodes.values():
        if keep_n is not None and n.node_type not in keep_n:
            continue
        if n.node_type == "construction":
            if n.metadata.get("confidence", 1.0) < min_construction_confidence:
                continue
            if drop_ambiguous_constructions and n.metadata.get("ambiguity_score", 0) >= 0.3:
                continue
        out.add_node(n)
    for e in g.edges:
        if keep_e is not None and e.edge_type not in keep_e:
            continue
        if e.src in out.nodes and e.dst in out.nodes:
            out.add_edge(e)
    return out
