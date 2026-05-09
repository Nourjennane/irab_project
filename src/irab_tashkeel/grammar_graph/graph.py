"""Per-sentence grammar graph — multi-typed node + multi-typed edge.

Sits on top of :class:`schema_v2.Sentence` and produces a unified
graph that downstream Step 5 long-range reasoning, Step 8 decoding,
and future GNN-based architectures consume.

Node types
----------

- ``token``        — one per :class:`schema_v2.Token`
- ``construction`` — one per :class:`schema_v2.Construction`
- ``clause``       — one per :class:`schema_v2.Clause`
- ``phrase``       — one per :class:`schema_v2.Span`
- ``discourse``    — one per :class:`schema_v2.DiscourseLink` source/target

Edge types
----------

- ``dep``                  — dependency edge between two tokens (UD)
- ``agreement``            — gender/number/definite axis agreement
- ``clause_member``        — token → clause (and construction → clause)
- ``construction_member``  — token → construction
- ``semantic_link``        — predicate-argument / event role
- ``discourse_link``       — cross-sentence reference (placeholder)
- ``coref``                — co-reference placeholder

Multiple edges may exist between the same pair of nodes (e.g.,
a dep edge + an agreement edge between two tokens). Edges are
stored in a flat list with adjacency-index sidecar dicts for
O(1) neighbour lookup.

Serialisation
-------------

`to_dict()` / `from_dict()` round-trip with the same compact JSON
shape as schema_v2's other dataclasses. The Sentence.graph slot in
schema_v2 carries a *flat* list of token-to-token edges; the
SentenceGraph object here carries the *full* multi-level
representation. Use :func:`bridge_schema_v2_graph` to convert.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ===========================================================================
# Constants — edge / node type vocabularies
# ===========================================================================

NODE_TYPES = ("token", "construction", "clause", "phrase", "discourse")

EDGE_TYPES = (
    "dep",
    "agreement",
    "clause_member",
    "construction_member",
    "semantic_link",
    "discourse_link",
    "coref",
)


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class GraphNode:
    """A node in a SentenceGraph. ``node_id`` is unique per graph."""
    node_id:    str
    node_type:  str                                  # one of NODE_TYPES
    schema_ref: str                                  # token idx (str) or schema_v2 UUID
    sentence_id: str = ""
    metadata:   Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphNode":
        return cls(
            node_id=d["node_id"], node_type=d["node_type"],
            schema_ref=d["schema_ref"], sentence_id=d.get("sentence_id", ""),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class GraphEdge:
    """A directed edge between two nodes. Multiple edges can exist
    between the same pair if their ``edge_type`` differs.
    """
    src:        str
    dst:        str
    edge_type:  str                                  # one of EDGE_TYPES
    label:      str = ""
    confidence: float = 1.0
    metadata:   Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphEdge":
        return cls(
            src=d["src"], dst=d["dst"], edge_type=d["edge_type"],
            label=d.get("label", ""), confidence=d.get("confidence", 1.0),
            metadata=dict(d.get("metadata", {})),
        )


# ===========================================================================
# SentenceGraph
# ===========================================================================

class SentenceGraph:
    """Per-sentence grammar graph.

    Built by :mod:`grammar_graph.builders`; queried by
    :mod:`grammar_graph.queries`; exported to sparse tensors by
    :mod:`grammar_graph.sparse`.

    Internally maintains:
      - ``nodes`` dict: node_id → GraphNode
      - ``edges`` list of GraphEdge
      - adjacency sidecars ``_out`` / ``_in`` for O(deg(node)) lookup
        of neighbours
    """

    def __init__(self, sentence_id: str = ""):
        self.sentence_id: str = sentence_id
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._out: Dict[str, List[GraphEdge]] = defaultdict(list)
        self._in:  Dict[str, List[GraphEdge]] = defaultdict(list)

    # -- mutation -------------------------------------------------------------

    def add_node(self, node: GraphNode) -> str:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node_id {node.node_id}")
        if node.node_type not in NODE_TYPES:
            raise ValueError(f"unknown node_type {node.node_type!r}")
        self.nodes[node.node_id] = node
        return node.node_id

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.edge_type not in EDGE_TYPES:
            raise ValueError(f"unknown edge_type {edge.edge_type!r}")
        if edge.src not in self.nodes:
            raise ValueError(f"src node {edge.src!r} not in graph")
        if edge.dst not in self.nodes:
            raise ValueError(f"dst node {edge.dst!r} not in graph")
        self.edges.append(edge)
        self._out[edge.src].append(edge)
        self._in[edge.dst].append(edge)

    # -- queries --------------------------------------------------------------

    def n_nodes(self) -> int:
        return len(self.nodes)

    def n_edges(self) -> int:
        return len(self.edges)

    def out_edges(self, node_id: str, edge_type: Optional[str] = None) -> List[GraphEdge]:
        edges = self._out.get(node_id, [])
        if edge_type is None:
            return list(edges)
        return [e for e in edges if e.edge_type == edge_type]

    def in_edges(self, node_id: str, edge_type: Optional[str] = None) -> List[GraphEdge]:
        edges = self._in.get(node_id, [])
        if edge_type is None:
            return list(edges)
        return [e for e in edges if e.edge_type == edge_type]

    def neighbours(self, node_id: str, edge_type: Optional[str] = None,
                    direction: str = "both") -> Set[str]:
        """Return set of neighbour node_ids.

        direction: "out" (only outgoing), "in" (only incoming),
        "both" (default).
        """
        out: Set[str] = set()
        if direction in ("out", "both"):
            for e in self.out_edges(node_id, edge_type):
                out.add(e.dst)
        if direction in ("in", "both"):
            for e in self.in_edges(node_id, edge_type):
                out.add(e.src)
        return out

    def nodes_of_type(self, node_type: str) -> List[GraphNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def edges_of_type(self, edge_type: str) -> List[GraphEdge]:
        return [e for e in self.edges if e.edge_type == edge_type]

    # -- structural metrics ---------------------------------------------------

    def degree_histogram(self) -> Dict[int, int]:
        from collections import Counter
        ctr: Counter = Counter()
        for n in self.nodes:
            deg = len(self._out.get(n, [])) + len(self._in.get(n, []))
            ctr[deg] += 1
        return dict(ctr)

    def node_type_counts(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(n.node_type for n in self.nodes.values()))

    def edge_type_counts(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(e.edge_type for e in self.edges))

    # -- (de)serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SentenceGraph":
        g = cls(sentence_id=d.get("sentence_id", ""))
        for nd in d.get("nodes", []):
            g.add_node(GraphNode.from_dict(nd))
        for ed in d.get("edges", []):
            g.add_edge(GraphEdge.from_dict(ed))
        return g

    def __repr__(self) -> str:
        return (f"SentenceGraph(id={self.sentence_id!r}, "
                f"n_nodes={self.n_nodes()}, n_edges={self.n_edges()}, "
                f"types={self.node_type_counts()})")
