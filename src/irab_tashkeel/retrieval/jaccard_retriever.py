"""Jaccard sentence-similarity retriever over a structured i'rāb corpus.

Phase A retrieval reference implementation — used by the structured predictor
to surface ``k`` nearest training-corpus sentences for the qualitative trace.

Index storage: in-memory list of (sentence_str, normalized_token_set,
items_dict). Build cost: O(N × W) over the corpus; query cost per call:
O(N × min(W_query, W_doc)) for the Jaccard scan.

For 5K training sentences this runs in milliseconds, no FAISS needed. The
journal version will swap in a dense embedding retriever; the call signature
``get_top_k(query, k=5) -> list[RetrievedExample]`` stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ..structured.schema import arabic_normalize
from ..structured.word_irab import SentenceIrab, WordIrab


@dataclass
class RetrievedExample:
    sentence: str
    score: float
    items: List[WordIrab]
    source_idx: int


def _tokenize(s: str) -> Sequence[str]:
    """Whitespace tokenize after Arabic normalization."""
    return arabic_normalize(s).split()


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class JaccardRetriever:
    """In-memory Jaccard top-K retriever over a sentence-level structured corpus."""

    def __init__(self, corpus: Iterable[SentenceIrab]):
        self._items: List[Tuple[str, frozenset, List[WordIrab]]] = []
        for s in corpus:
            tokens = frozenset(_tokenize(s.sentence))
            self._items.append((s.sentence, tokens, list(s.items)))
        self._n = len(self._items)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "JaccardRetriever":
        path = Path(path)
        sents: List[SentenceIrab] = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    sents.append(SentenceIrab.from_json_line(line))
        return cls(sents)

    def __len__(self) -> int:
        return self._n

    def get_top_k(
        self,
        query: str,
        k: int = 5,
        *,
        min_score: float = 0.0,
        exclude_exact: bool = True,
    ) -> List[RetrievedExample]:
        """Return the K most similar corpus sentences by Jaccard token overlap.

        Args:
            query: query sentence.
            k: number of results to return.
            min_score: filter out matches with Jaccard < min_score.
            exclude_exact: drop the corpus row whose sentence equals the query
                exactly (useful when the query came from the same corpus).
        """
        q_tokens = frozenset(_tokenize(query))
        if not q_tokens:
            return []
        scored: List[Tuple[float, int]] = []
        for idx, (sent, tokens, _items) in enumerate(self._items):
            if exclude_exact and sent.strip() == query.strip():
                continue
            score = _jaccard(q_tokens, tokens)
            if score >= min_score:
                scored.append((score, idx))
        scored.sort(key=lambda t: -t[0])
        out: List[RetrievedExample] = []
        for score, idx in scored[:k]:
            sent, _tok, items = self._items[idx]
            out.append(RetrievedExample(sentence=sent, score=score, items=items, source_idx=idx))
        return out
