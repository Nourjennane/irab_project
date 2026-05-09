"""SentenceGraph (de)serialisation.

Disk format: one JSON object per line, with the same compact shape
as schema_v2 records. Use these helpers to write a corpus of
graphs alongside the schema_v2 sentence corpus.

Bridge with schema_v2.GrammarGraph
----------------------------------

``schema_v2.Sentence.graph`` is a flat list of token-to-token
edges (created when the schema was specified — before this
graph-engine module existed). The new :class:`SentenceGraph` is
multi-typed and richer. Use:

    bridge_to_schema(g) → schema_v2.GrammarGraph

to project the multi-level graph down to schema_v2's flat slot,
when persisting the graph alongside the sentence record.

The bridge is one-way (richer → flatter); for full fidelity, write
the SentenceGraph as a separate JSONL via :func:`write_jsonl`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List

from ..data_v2.schema_v2 import GrammarGraph as SchemaGraph
from ..data_v2.schema_v2 import GraphEdge as SchemaGraphEdge
from .graph import GraphEdge, GraphNode, SentenceGraph


# ===========================================================================
# JSONL IO
# ===========================================================================

def write_jsonl(path: str | Path, graphs: Iterable[SentenceGraph]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as fh:
        for g in graphs:
            fh.write(json.dumps(g.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[SentenceGraph]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            yield SentenceGraph.from_dict(json.loads(line))


# ===========================================================================
# Bridge to schema_v2
# ===========================================================================

def bridge_to_schema(g: SentenceGraph) -> SchemaGraph:
    """Project the multi-level graph down to schema_v2's flat
    token-to-token edges (Sentence.graph slot).

    Only ``dep`` and ``agreement`` edges between tokens are kept;
    construction-member / clause-member edges go to non-token nodes
    and are dropped from the projection. For full fidelity write
    the SentenceGraph as a sidecar JSONL via :func:`write_jsonl`.
    """
    sg = SchemaGraph()
    for e in g.edges:
        if e.edge_type not in ("dep", "agreement"):
            continue
        src = g.nodes.get(e.src); dst = g.nodes.get(e.dst)
        if not src or not dst: continue
        if src.node_type != "token" or dst.node_type != "token":
            continue
        try:
            src_idx = int(src.schema_ref)
            dst_idx = int(dst.schema_ref)
        except (ValueError, TypeError):
            continue
        sg.edges.append(SchemaGraphEdge(
            src_idx=src_idx, dst_idx=dst_idx,
            edge_type=e.edge_type, label=e.label,
            confidence=e.confidence,
        ))
    return sg
