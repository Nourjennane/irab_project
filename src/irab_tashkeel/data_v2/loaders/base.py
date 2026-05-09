"""Base loader interface for all data_v2 sources.

Every source corpus (distill_v2, UD-PADT, MASAQ, Gazelle, CamelTB,
Sonnet-distilled, Quranic Arabic Corpus, Extended Quranic Treebank,
educational/pedagogical scrapes, …) ingests through a subclass of
:class:`BaseLoader`. The loader is responsible for:

1. Reading source-native records.
2. Normalising surface text via ``data_v2.normalization``.
3. Building one :class:`Sentence` per source record with
   appropriate :class:`LabelTag` provenance + confidence.
4. Setting :class:`SentenceMetadata` (domain, source, source_id,
   annotation_quality, parser_origin).
5. (Optionally) populating Construction / Clause / Graph fields if
   the source provides them.

Loaders MUST NOT compute curriculum metadata (difficulty, semantic
pressure, etc.) — that's a separate post-processing pass in
``data_v2.metadata``. The split keeps loaders deterministic and the
metadata computation reusable across loaders.
"""
from __future__ import annotations

import abc
import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..schema_v2 import (
    AnnotationQuality, Domain, Sentence, SentenceMetadata, write_jsonl,
)


class BaseLoader(abc.ABC):
    """Abstract base for all source loaders.

    Subclass attributes
    -------------------
    source_id            unique identifier for the source corpus
                         (e.g. "distill_v2", "ud_padt", "masaq_eval",
                         "gazelle_test")
    domain               :class:`Domain` value (msa_news / quranic / ...)
    annotation_quality   :class:`AnnotationQuality` tier
    parser_origin        primary parser used (e.g. "stanza_ud",
                         "ud_padt_gold", "haiku_distill", "human")
    license              license string (e.g. "MIT", "research-only")
    """

    source_id: str = ""
    domain: str = Domain.UNKNOWN.value
    annotation_quality: str = AnnotationQuality.UNKNOWN.value
    parser_origin: str = ""
    license: str = ""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # -- abstract --------------------------------------------------------------

    @abc.abstractmethod
    def iter_raw(self) -> Iterator[Dict[str, Any]]:
        """Yield raw records from the source.

        The shape of each dict is source-specific; ``normalize_row``
        knows how to interpret it.
        """

    @abc.abstractmethod
    def normalize_row(self, raw: Dict[str, Any], idx: int) -> Optional[Sentence]:
        """Convert one raw record to a schema_v2 :class:`Sentence`.

        Return ``None`` to skip the record (e.g., empty sentences,
        records that fail validation).
        """

    # -- common --------------------------------------------------------------

    def _make_metadata(self, source_id_within: str = "") -> SentenceMetadata:
        return SentenceMetadata(
            domain=self.domain,
            source=self.source_id,
            source_id=source_id_within,
            annotation_quality=self.annotation_quality,
            parser_origin=self.parser_origin,
            morph_origin=self.parser_origin,
            dep_origin=self.parser_origin,
            role_origin=self.parser_origin,
            marker_origin=self.parser_origin,
            construction_origin="",
            reasoning_origin="",
            license=self.license,
            ingestion_timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )

    def load_all(self) -> List[Sentence]:
        """Load and normalise every record. Skips records where
        ``normalize_row`` returns ``None``.
        """
        out: List[Sentence] = []
        for i, raw in enumerate(self.iter_raw()):
            sent = self.normalize_row(raw, i)
            if sent is not None:
                out.append(sent)
        return out

    def write_jsonl(self, out_path: str | Path) -> int:
        """Convenience: load all, write to JSONL, return count."""
        sents = self.load_all()
        write_jsonl(str(out_path), sents)
        return len(sents)


# ---------------------------------------------------------------------------
# Loader registry — auto-populated by submodules
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {}


def register_loader(cls: type) -> type:
    """Decorator to register a loader by its ``source_id``."""
    if not getattr(cls, "source_id", ""):
        raise ValueError(f"{cls.__name__} must define source_id")
    _REGISTRY[cls.source_id] = cls
    return cls


def get_loader(source_id: str) -> type:
    """Look up a loader class by source id; raises if unknown."""
    if source_id not in _REGISTRY:
        raise KeyError(f"unknown source_id={source_id!r}; "
                       f"registered: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[source_id]


def all_registered() -> List[str]:
    return sorted(_REGISTRY.keys())
