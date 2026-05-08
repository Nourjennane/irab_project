"""Quranic grammar-memory retrieval.

A lightweight, modular extension to :class:`JaccardRetriever` that surfaces
classical Arabic *grammatical* exemplars (Quranic, by default) for a query
sentence. It works in two layers:

1. **Construction detector** — a regex-based tagger that labels each sentence
   with one or more construction types (INNA, KANA, PREP, IDAFA, EXCEPTION,
   COORDINATION). Operates on the surface sentence + (optionally) the gold
   i'rāb prose for higher precision when available.

2. **Construction-aware Jaccard retriever** — at query time, detects the
   query's constructions and returns top-K Quranic exemplars that share at
   least one construction tag, ordered by Jaccard surface similarity.

This is the minimal viable grammar memory: no embeddings, no FAISS, no
retraining. The interface (``retrieve(sentence, k=...)`` returning a list of
:class:`GrammarExample`) is FAISS-compatible so the journal version can swap
in dense retrieval without changing the predictor / demo.

Used by:
* qualitative trace renderer (display similar Quranic constructions)
* Gradio demo (optional ``Similar Quranic constructions`` panel)
* future-work hook for retrieval-aware logit reranking
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from ..structured.schema import arabic_normalize


# ---------------------------------------------------------------------------
# Construction tags
# ---------------------------------------------------------------------------
CONSTRUCTION_TAGS = (
    "INNA",          # ـ ان/أن/لكن/ليت/لعل/كأن sister
    "KANA",          # كان / أصبح / ظل / ليس / صار / بات / أمسى / ما زال etc.
    "PREP",          # حرف جر present
    "IDAFA",         # iḍāfa chain (مضاف إليه)
    "EXCEPTION",     # سوى / إلا / عدا / خلا / حاشا / غير
    "COORDINATION",  # حرف عطف (و / ف / ثم …)
    "RELATIVE",      # الذي / التي / الذين / اللاتي / مَن / ما (relative pronouns)
    "VOCATIVE",      # يا / ايا / هيا (نداء)
)


# Trigger surface forms (kept in fold-normalised form: alif & ya folded).
_INNA = {"ان", "لكن", "ليت", "لعل"}
_KANA = {
    "كان", "كانت", "كانوا", "يكون", "تكون",
    "ليس", "ليست",
    "اصبح", "اصبحت", "اصبحوا",
    "صار", "صارت",
    "ظل", "ظلت",
    "بات", "باتت",
    "امسي", "امست",
    "مازال", "ماتزال",
}
_PREP = {"في", "من", "الي", "علي", "عن", "ب", "ل", "ك", "حتي", "منذ", "مذ"}
_EXCEPTION = {"الا", "سوي", "عدا", "خلا", "حاشا", "غير"}
_COORD = {"و", "ف", "ثم", "او", "ام"}
_RELATIVE = {"الذي", "التي", "الذين", "اللاتي", "اللواتي", "من", "ما"}
_VOCATIVE = {"يا", "ايا", "هيا", "اي"}


def _tokens(s: str) -> List[str]:
    return arabic_normalize(s).split()


def detect_constructions(sentence: str, irab_concat: Optional[str] = None) -> Set[str]:
    """Detect which constructions are present in a sentence.

    `irab_concat` is the concatenated gold i'rāb prose (whitespace joined).
    If supplied, it also signals iḍāfa (via the literal ``مضاف إليه``) and
    coordination (via ``حرف عطف``).
    """
    tags: Set[str] = set()
    toks = _tokens(sentence)
    tokset = set(toks)

    if tokset & _INNA:
        tags.add("INNA")
    if tokset & _KANA:
        tags.add("KANA")
    if tokset & _PREP:
        tags.add("PREP")
    if tokset & _EXCEPTION:
        tags.add("EXCEPTION")
    if tokset & _COORD:
        tags.add("COORDINATION")
    if tokset & _RELATIVE:
        tags.add("RELATIVE")
    if tokset & _VOCATIVE:
        tags.add("VOCATIVE")

    if irab_concat:
        irab_norm = arabic_normalize(irab_concat)
        if "مضاف اليه" in irab_norm:
            tags.add("IDAFA")
        if "حرف عطف" in irab_norm and "COORDINATION" not in tags:
            tags.add("COORDINATION")
        if "منادي" in irab_norm or "نداء" in irab_norm:
            tags.add("VOCATIVE")
        if "استثناء" in irab_norm:
            tags.add("EXCEPTION")
    return tags


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
@dataclass
class GrammarExample:
    """One indexed grammatical exemplar."""

    sentence: str
    irab_per_word: List[Tuple[str, str]]   # (word, irab_prose)
    constructions: Set[str] = field(default_factory=set)
    source: str = ""                       # e.g. "MASAQ", or "Yarob"
    sura_verse: Optional[str] = None       # only populated for Quranic rows
    score: float = 0.0                     # filled by the retriever


# ---------------------------------------------------------------------------
# Grammar memory
# ---------------------------------------------------------------------------
class GrammarMemory:
    """Construction-aware Jaccard retriever over a curated grammar corpus.

    Build:
        gm = GrammarMemory.from_masaq("data/masaq_eval.jsonl")
    Query:
        hits = gm.retrieve("ذهب الطالب إلى المدرسة", k=5)
    """

    def __init__(self, examples: Iterable[GrammarExample]):
        self._items: List[Tuple[GrammarExample, frozenset]] = []
        for ex in examples:
            tokens = frozenset(_tokens(ex.sentence))
            self._items.append((ex, tokens))
        self._by_tag: dict[str, List[int]] = {t: [] for t in CONSTRUCTION_TAGS}
        for idx, (ex, _) in enumerate(self._items):
            for t in ex.constructions:
                self._by_tag.setdefault(t, []).append(idx)

    @classmethod
    def from_masaq(cls, path: str | Path) -> "GrammarMemory":
        path = Path(path)
        examples: List[GrammarExample] = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sent = rec.get("sentence", "")
                items = rec.get("items") or []
                irab_pairs = [(it.get("word", ""), it.get("irab", "")) for it in items
                              if isinstance(it, dict)]
                irab_concat = " \n".join(p[1] for p in irab_pairs)
                tags = detect_constructions(sent, irab_concat=irab_concat)
                examples.append(GrammarExample(
                    sentence=sent,
                    irab_per_word=irab_pairs,
                    constructions=tags,
                    source="MASAQ",
                    sura_verse=rec.get("sura_verse"),
                ))
        return cls(examples)

    def __len__(self) -> int:
        return len(self._items)

    def _jaccard(self, a: frozenset, b: frozenset) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def retrieve(
        self,
        sentence: str,
        *,
        k: int = 5,
        require_tag: Optional[str] = None,
        prefer_shared_constructions: bool = True,
        min_score: float = 0.0,
    ) -> List[GrammarExample]:
        """Return up to ``k`` exemplars most similar to ``sentence``.

        Args:
            sentence: query Arabic sentence.
            k: max number of results.
            require_tag: if set, only consider exemplars with this construction.
            prefer_shared_constructions: when True, exemplars that share at
                least one construction tag with the query are scored with a
                +0.05 bonus; this makes the retriever construction-aware
                without dropping all overlap-only matches.
            min_score: drop exemplars below this score.
        """
        q_tokens = frozenset(_tokens(sentence))
        if not q_tokens:
            return []
        q_tags = detect_constructions(sentence)

        scored: List[Tuple[float, int]] = []
        candidates: Iterable[int]
        if require_tag is not None:
            candidates = self._by_tag.get(require_tag, [])
        else:
            candidates = range(len(self._items))

        for idx in candidates:
            ex, tokens = self._items[idx]
            score = self._jaccard(q_tokens, tokens)
            if prefer_shared_constructions and (q_tags & ex.constructions):
                score += 0.05
            if score >= min_score:
                scored.append((score, idx))

        scored.sort(key=lambda t: -t[0])
        out: List[GrammarExample] = []
        for score, idx in scored[:k]:
            ex = self._items[idx][0]
            ex.score = float(score)
            out.append(ex)
        return out

    def stats(self) -> dict:
        return {
            "n": len(self._items),
            "tag_counts": {t: len(v) for t, v in self._by_tag.items()},
        }
