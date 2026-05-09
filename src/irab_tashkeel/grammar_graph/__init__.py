"""grammar_graph — Step 4 grammar graph engine.

Public API:

    graph.SentenceGraph
    graph.GraphNode
    graph.GraphEdge
    graph.NODE_TYPES, EDGE_TYPES

    builders.build_sentence_graph(sentence) → SentenceGraph
    builders.build_corpus_graphs(sentences) → List[SentenceGraph]

    queries.graph_depth(g, root_type)
    queries.shortest_path(g, src, dst, edge_types=None)
    queries.constructions_at_token(g, token_idx)
    queries.overlap_constructions(g)
    queries.walk_to_governor(g, token_idx, max_hops=3)
    queries.find_construction_clause_membership(g)
    queries.filter_by_curriculum(g, ...)

    sparse.to_sparse(g, token_features=None) → SparseGraph
    sparse.batch_sparse(graphs) → dict (PyG-compatible)

    serialization.write_jsonl(path, graphs)
    serialization.read_jsonl(path)
    serialization.bridge_to_schema(g) → schema_v2.GrammarGraph
"""
from .graph import (
    EDGE_TYPES, GraphEdge, GraphNode, NODE_TYPES, SentenceGraph,
)
from .builders import build_corpus_graphs, build_sentence_graph
from .queries import (
    constructions_at_token, filter_by_curriculum,
    find_construction_clause_membership, graph_depth,
    overlap_constructions, shortest_path, walk_to_governor,
)
from .sparse import (
    EDGE_TYPE_TO_ID, NODE_TYPE_TO_ID, SparseGraph,
    batch_sparse, to_sparse,
)
from .serialization import bridge_to_schema, read_jsonl, write_jsonl

__all__ = [
    "EDGE_TYPES", "EDGE_TYPE_TO_ID", "GraphEdge", "GraphNode",
    "NODE_TYPES", "NODE_TYPE_TO_ID", "SentenceGraph", "SparseGraph",
    "batch_sparse", "bridge_to_schema", "build_corpus_graphs",
    "build_sentence_graph", "constructions_at_token",
    "filter_by_curriculum", "find_construction_clause_membership",
    "graph_depth", "overlap_constructions", "read_jsonl",
    "shortest_path", "to_sparse", "walk_to_governor", "write_jsonl",
]
