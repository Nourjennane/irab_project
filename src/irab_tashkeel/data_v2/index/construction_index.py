"""Construction-aware index over schema_v2 sentences.

In-memory index (Phase 1) supporting fast filtering by:

- construction family
- particle / subgroup
- domain
- annotation quality
- difficulty level (for curriculum sampling)
- semantic-pressure score
- nested-construction depth
- sentence length range

The index is built from a list of :class:`Sentence` objects (or
streamed JSONL) and provides O(1) bucket lookup. It's deliberately
simple so it works without external dependencies; FAISS-backed
similarity search comes later in ``retrieval_v2``.

Usage::

    idx = ConstructionIndex.from_jsonl("data_v2/annotated/train.jsonl")
    kana_examples = idx.filter(family="kana_sisters", min_difficulty=3)
    quranic_with_overlap = idx.filter(domain="quranic", min_overlap=1)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Tuple

from ..schema_v2 import Sentence, read_jsonl


@dataclass
class ConstructionIndex:
    """In-memory index of schema_v2 sentences."""
    sentences: List[Sentence] = field(default_factory=list)
    # Inverted indices
    by_family:    Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))
    by_subgroup:  Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))
    by_domain:    Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))
    by_quality:   Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))
    by_difficulty: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    @classmethod
    def from_sentences(cls, sentences: List[Sentence]) -> "ConstructionIndex":
        idx = cls()
        for s in sentences:
            idx.add(s)
        return idx

    @classmethod
    def from_jsonl(cls, path: str) -> "ConstructionIndex":
        return cls.from_sentences(list(read_jsonl(path)))

    def add(self, sentence: Sentence) -> int:
        """Add a sentence, return its position in the sentences list."""
        i = len(self.sentences)
        self.sentences.append(sentence)
        for c in sentence.constructions:
            self.by_family[c.family].add(i)
            if c.subgroup:
                self.by_subgroup[c.subgroup].add(i)
        self.by_domain[sentence.metadata.domain].add(i)
        self.by_quality[sentence.metadata.annotation_quality].add(i)
        self.by_difficulty[sentence.curriculum.difficulty_level].add(i)
        return i

    # ----------------------------------------------------------------
    # Filters
    # ----------------------------------------------------------------

    def filter(
        self,
        *,
        family: Optional[str] = None,
        subgroup: Optional[str] = None,
        domain: Optional[str] = None,
        quality: Optional[str] = None,
        min_difficulty: Optional[int] = None,
        max_difficulty: Optional[int] = None,
        min_overlap: Optional[int] = None,
        min_semantic_pressure: Optional[int] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> List[Sentence]:
        """Return sentences matching all provided constraints."""
        candidates: Optional[Set[int]] = None

        def _intersect(s: Set[int]) -> Set[int]:
            nonlocal candidates
            return s if candidates is None else candidates & s

        if family is not None:
            candidates = _intersect(self.by_family.get(family, set()))
        if subgroup is not None:
            candidates = _intersect(self.by_subgroup.get(subgroup, set()))
        if domain is not None:
            candidates = _intersect(self.by_domain.get(domain, set()))
        if quality is not None:
            candidates = _intersect(self.by_quality.get(quality, set()))

        if candidates is None:
            candidates = set(range(len(self.sentences)))

        out: List[Sentence] = []
        for i in candidates:
            s = self.sentences[i]
            if min_difficulty is not None and s.curriculum.difficulty_level < min_difficulty:
                continue
            if max_difficulty is not None and s.curriculum.difficulty_level > max_difficulty:
                continue
            if min_overlap is not None and s.curriculum.nested_construction_count < min_overlap:
                continue
            if min_semantic_pressure is not None and s.curriculum.semantic_pressure_score < min_semantic_pressure:
                continue
            if min_length is not None and s.curriculum.sentence_length_tokens < min_length:
                continue
            if max_length is not None and s.curriculum.sentence_length_tokens > max_length:
                continue
            out.append(s)
        return out

    # ----------------------------------------------------------------
    # Aggregates
    # ----------------------------------------------------------------

    def family_histogram(self) -> Dict[str, int]:
        return {f: len(s) for f, s in self.by_family.items()}

    def domain_histogram(self) -> Dict[str, int]:
        return {f: len(s) for f, s in self.by_domain.items()}

    def difficulty_histogram(self) -> Dict[int, int]:
        return {k: len(v) for k, v in self.by_difficulty.items()}

    def __len__(self) -> int:
        return len(self.sentences)
