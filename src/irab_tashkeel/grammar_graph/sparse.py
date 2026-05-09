"""Sparse-tensor export for downstream GNN training.

Produces the flat representations PyTorch Geometric / DGL / sparse
COO consumers expect:

    - ``edge_index``       : ``LongTensor[2, n_edges]``
    - ``edge_type``        : ``LongTensor[n_edges]``  (id into EDGE_TYPES)
    - ``edge_attr``        : ``FloatTensor[n_edges, k]`` (confidence, …)
    - ``node_type``        : ``LongTensor[n_nodes]``  (id into NODE_TYPES)
    - ``node_attr``        : ``FloatTensor[n_nodes, k]``  (case/role/marker
                                                            one-hots, …)
    - ``node_id_to_index`` : ``Dict[str, int]``
    - ``node_index_to_id`` : ``Dict[int, str]``

Optional ``torch`` import — works without torch (returns numpy
arrays). Useful for offline indexing + provenance verification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .graph import EDGE_TYPES, NODE_TYPES, SentenceGraph

NODE_TYPE_TO_ID = {t: i for i, t in enumerate(NODE_TYPES)}
EDGE_TYPE_TO_ID = {t: i for i, t in enumerate(EDGE_TYPES)}


@dataclass
class SparseGraph:
    edge_index: np.ndarray                     # shape (2, n_edges) — int64
    edge_type:  np.ndarray                     # shape (n_edges,)   — int64
    edge_attr:  np.ndarray                     # shape (n_edges, 1) — float32 (confidence)
    node_type:  np.ndarray                     # shape (n_nodes,)   — int64
    node_attr:  np.ndarray                     # shape (n_nodes, ?) — float32 (raw features)
    node_id_to_index: Dict[str, int]           # mapping
    node_index_to_id: Dict[int, str]
    sentence_id: str = ""

    def to_torch(self):
        """Convert all arrays to torch tensors (lazy import)."""
        import torch
        return {
            "edge_index": torch.from_numpy(self.edge_index).long(),
            "edge_type":  torch.from_numpy(self.edge_type).long(),
            "edge_attr":  torch.from_numpy(self.edge_attr).float(),
            "node_type":  torch.from_numpy(self.node_type).long(),
            "node_attr":  torch.from_numpy(self.node_attr).float(),
            "node_id_to_index": dict(self.node_id_to_index),
            "sentence_id": self.sentence_id,
        }


def to_sparse(g: SentenceGraph, *,
              token_features: Optional[List[str]] = None) -> SparseGraph:
    """Convert a :class:`SentenceGraph` to flat sparse arrays.

    ``token_features`` lists which Token attributes to encode as
    raw float features. Default: ``["case_conf", "role_conf",
    "marker_conf"]`` — the per-field confidences. Non-token nodes
    pad to all-zero feature vectors.
    """
    if token_features is None:
        token_features = ["case_conf", "role_conf", "marker_conf"]

    nodes = list(g.nodes.values())
    n_to_i = {n.node_id: i for i, n in enumerate(nodes)}
    i_to_n = {i: n.node_id for n, i in zip(nodes, range(len(nodes)))}

    n_nodes = len(nodes)
    n_edges = len(g.edges)
    n_feats = len(token_features)

    node_type = np.zeros(n_nodes, dtype=np.int64)
    node_attr = np.zeros((n_nodes, n_feats), dtype=np.float32)
    for i, n in enumerate(nodes):
        node_type[i] = NODE_TYPE_TO_ID.get(n.node_type, 0)
        for k, feat in enumerate(token_features):
            v = n.metadata.get(feat)
            if isinstance(v, (int, float)):
                node_attr[i, k] = float(v)

    if n_edges == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type  = np.zeros((0,),     dtype=np.int64)
        edge_attr  = np.zeros((0, 1),   dtype=np.float32)
    else:
        edge_index = np.zeros((2, n_edges), dtype=np.int64)
        edge_type  = np.zeros(n_edges,       dtype=np.int64)
        edge_attr  = np.zeros((n_edges, 1),  dtype=np.float32)
        for i, e in enumerate(g.edges):
            edge_index[0, i] = n_to_i[e.src]
            edge_index[1, i] = n_to_i[e.dst]
            edge_type[i]     = EDGE_TYPE_TO_ID.get(e.edge_type, 0)
            edge_attr[i, 0]  = float(e.confidence)

    return SparseGraph(
        edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr,
        node_type=node_type, node_attr=node_attr,
        node_id_to_index=n_to_i, node_index_to_id=i_to_n,
        sentence_id=g.sentence_id,
    )


# ===========================================================================
# Batching (for GNN training later)
# ===========================================================================

def batch_sparse(graphs: List[SparseGraph]) -> Dict[str, np.ndarray]:
    """Concatenate multiple :class:`SparseGraph` into a batched representation.

    Adds a ``batch`` array of shape ``(n_total_nodes,)`` indicating
    which graph each node belongs to (PyG convention).
    """
    if not graphs:
        return {
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_type":  np.zeros((0,), dtype=np.int64),
            "edge_attr":  np.zeros((0, 1), dtype=np.float32),
            "node_type":  np.zeros((0,), dtype=np.int64),
            "node_attr":  np.zeros((0, 0), dtype=np.float32),
            "batch":      np.zeros((0,), dtype=np.int64),
            "n_graphs":   0,
        }

    n_node_per_graph: List[int] = [len(g.node_type) for g in graphs]
    cumul = np.cumsum([0] + n_node_per_graph[:-1])

    edge_indices = []
    for i, g in enumerate(graphs):
        ei = g.edge_index + cumul[i]      # offset node ids
        edge_indices.append(ei)

    out = {
        "edge_index": np.concatenate(edge_indices, axis=1) if edge_indices else
                       np.zeros((2, 0), dtype=np.int64),
        "edge_type":  np.concatenate([g.edge_type for g in graphs]) if graphs else
                       np.zeros((0,), dtype=np.int64),
        "edge_attr":  np.concatenate([g.edge_attr for g in graphs], axis=0) if graphs else
                       np.zeros((0, 1), dtype=np.float32),
        "node_type":  np.concatenate([g.node_type for g in graphs]) if graphs else
                       np.zeros((0,), dtype=np.int64),
        "node_attr":  np.concatenate([g.node_attr for g in graphs], axis=0) if graphs else
                       np.zeros((0, 0), dtype=np.float32),
        "batch":      np.concatenate([np.full(n, i, dtype=np.int64)
                                       for i, n in enumerate(n_node_per_graph)])
                       if graphs else np.zeros((0,), dtype=np.int64),
        "n_graphs":   len(graphs),
    }
    return out
