"""Phase R — grammar memory (per-family FAISS index + per-instance metadata).

The memory is structured as one JSONL + one FAISS file per construction
family. At inference time, retrieval is filtered by family (binary)
then by particle group (binary), then ranked by cosine similarity on
the encoder span embedding.

Storage layout::

    data/grammar_memory/
        kana_sisters/
            instances.jsonl              # List[ConstructionInstance.to_dict()]
            embeddings.faiss             # FAISS IndexFlatIP, normalised vectors
        inna_sisters/
            ...
        ...
        _build_summary.json              # build provenance + counts

The class supports lazy loading (only families needed at inference
time are loaded).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .signature import ConstructionInstance, ALL_FAMILIES, symbolic_overlap


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize rows to unit L2 norm (so dot product = cosine)."""
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / max(n, 1e-12)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


@dataclass
class RetrievalHit:
    """One retrieved construction analogue."""
    instance: ConstructionInstance
    cosine: float
    sym_overlap: float
    score: float                 # final aggregated score
    rank: int


class GrammarMemoryBuilder:
    """Single-family in-memory builder. Used during memory construction."""

    def __init__(self, family: str):
        self.family = family
        self.instances: List[ConstructionInstance] = []
        self._embeddings: List[np.ndarray] = []

    def add(self, instance: ConstructionInstance, embedding: np.ndarray) -> None:
        emb = _l2_normalize(embedding.astype(np.float32))
        instance.embedding_idx = len(self.instances)
        self.instances.append(instance)
        self._embeddings.append(emb)

    def save(self, out_dir: Path) -> Dict:
        """Save instances.jsonl + embeddings.faiss for this family."""
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write JSONL
        jsonl_path = out_dir / "instances.jsonl"
        with jsonl_path.open("w") as fh:
            for inst in self.instances:
                fh.write(json.dumps(inst.to_dict(), ensure_ascii=False) + "\n")
        # Write FAISS index
        faiss_path = out_dir / "embeddings.faiss"
        if self._embeddings:
            import faiss
            d = self._embeddings[0].shape[0]
            mat = np.stack(self._embeddings, axis=0).astype(np.float32)
            index = faiss.IndexFlatIP(d)
            index.add(mat)
            faiss.write_index(index, str(faiss_path))
        else:
            # No instances for this family — write empty marker
            faiss_path.write_bytes(b"")
        return {
            "family": self.family,
            "n_instances": len(self.instances),
            "embedding_dim": self._embeddings[0].shape[0] if self._embeddings else 0,
        }


class GrammarMemory:
    """Loaded grammar memory — read-only, supports retrieve()."""

    def __init__(self, root: Path, families: Optional[List[str]] = None):
        self.root = Path(root)
        self.families: List[str] = families or list(ALL_FAMILIES)
        self._instances: Dict[str, List[ConstructionInstance]] = {}
        self._indices: Dict[str, "faiss.Index"] = {}
        self._loaded: Dict[str, bool] = {f: False for f in self.families}

    def _load(self, family: str) -> None:
        if self._loaded.get(family):
            return
        fam_dir = self.root / family
        jsonl_path = fam_dir / "instances.jsonl"
        faiss_path = fam_dir / "embeddings.faiss"
        if not jsonl_path.exists():
            self._instances[family] = []
            self._loaded[family] = True
            return
        instances: List[ConstructionInstance] = []
        with jsonl_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                instances.append(ConstructionInstance.from_dict(json.loads(line)))
        self._instances[family] = instances
        if faiss_path.exists() and faiss_path.stat().st_size > 0:
            import faiss
            self._indices[family] = faiss.read_index(str(faiss_path))
        else:
            self._indices[family] = None
        self._loaded[family] = True

    def retrieve(
        self,
        query: ConstructionInstance,
        query_embedding: np.ndarray,
        k: int = 5,
        alpha: float = 0.3,
        require_particle_group: bool = True,
    ) -> List[RetrievalHit]:
        """Retrieve top-k analogues for ``query``.

        Symbolic filter: candidate must match query.construction (binary).
        If ``require_particle_group``, candidate must also match
        query.particle_group (binary).

        Vector ranking: cosine similarity on FAISS index (top-(k * 4))
        candidates from the symbolic-matched set, then re-ranked by
        ``alpha * symbolic_overlap + (1 - alpha) * cosine``.
        """
        family = query.construction
        if family not in self.families:
            return []
        self._load(family)
        instances = self._instances.get(family, [])
        index = self._indices.get(family)
        if not instances or index is None:
            return []

        # Step 1: symbolic filter
        candidates_idx: List[int] = []
        for i, inst in enumerate(instances):
            if require_particle_group and inst.particle_group != query.particle_group:
                continue
            # Don't retrieve the query itself if its instance_id matches
            if inst.instance_id == query.instance_id:
                continue
            candidates_idx.append(i)
        if not candidates_idx:
            return []

        # Step 2: vector top-(k * 4) within symbolic-matched set
        # Use FAISS to find global top-k_search, then filter to candidates_idx
        k_search = min(len(instances), max(k * 8, 32))
        q_emb = _l2_normalize(query_embedding.astype(np.float32)).reshape(1, -1)
        scores, idxs = index.search(q_emb, k_search)
        scores = scores[0]
        idxs = idxs[0]

        # Build lookup of allowed candidate indices
        allowed = set(candidates_idx)
        hits: List[Tuple[int, float]] = []  # (instance_idx, cosine)
        for cosine_score, instance_idx in zip(scores, idxs):
            if instance_idx in allowed:
                hits.append((int(instance_idx), float(cosine_score)))
            if len(hits) >= k * 2:
                break

        # If not enough hits in top-k_search, fall back to full scan over candidates
        if len(hits) < k:
            # Embed all candidates (already in the FAISS index — recompute via
            # internal vectors)
            cand_embs = np.stack(
                [index.reconstruct(int(i)) for i in candidates_idx], axis=0
            )
            cosines = (q_emb @ cand_embs.T)[0]
            order = np.argsort(-cosines)
            hits = [(candidates_idx[int(o)], float(cosines[int(o)])) for o in order[: k * 2]]

        # Step 3: aggregate score = alpha * symbolic + (1-alpha) * cosine
        scored: List[RetrievalHit] = []
        for instance_idx, cosine_score in hits:
            inst = instances[instance_idx]
            sym = symbolic_overlap(query, inst)
            agg = alpha * sym + (1.0 - alpha) * cosine_score
            scored.append(RetrievalHit(
                instance=inst,
                cosine=cosine_score,
                sym_overlap=sym,
                score=agg,
                rank=0,
            ))
        scored.sort(key=lambda h: -h.score)
        for r, h in enumerate(scored[:k]):
            h.rank = r
        return scored[:k]

    def n_instances(self, family: str) -> int:
        self._load(family)
        return len(self._instances.get(family, []))


def save_build_summary(root: Path, family_summaries: List[Dict]) -> None:
    out_path = root / "_build_summary.json"
    summary = {
        "n_families": len(family_summaries),
        "families": family_summaries,
        "total_instances": sum(s["n_instances"] for s in family_summaries),
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
