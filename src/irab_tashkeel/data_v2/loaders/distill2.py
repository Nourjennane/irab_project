"""Distill-v2 → schema_v2 loader.

Ingests the frozen-baseline production corpus
(``data/morph_v1_dep/train.jsonl``) into the schema_v2 sentence
format. The distill_v2 records carry per-token (word, case, role,
marker, pos, deprel, head_idx, governor_upos) plus optional irab
prose; we map them to schema_v2 with provenance
``annotation_quality=SILVER_LLM_DISTILL`` (Haiku-distilled) and
``parser_origin="haiku_distill+stanza_ud"``.

The 70% Stanza-aligned subset of distill_v2 records carries dep
features; the remaining 30% (where Stanza couldn't align) becomes
``has_dep=False`` and the loader sets ``dep_head_idx=-1`` for those
tokens.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional

from ..normalization import arabic_normalize, normalize_text, tokenize_whitespace
from ..schema_v2 import (
    AnnotationCompleteness, AnnotationQuality, Construction, Domain,
    LabelTag, Morphology, Sentence, Token,
)
from .base import BaseLoader, register_loader


@register_loader
class Distill2Loader(BaseLoader):
    """Loader for ``data/morph_v1_dep/train.jsonl`` — frozen-baseline production training corpus."""

    source_id          = "distill_v2"
    domain             = Domain.MSA_NEWS.value
    annotation_quality = AnnotationQuality.SILVER_LLM_DISTILL.value
    parser_origin      = "haiku_distill+stanza_ud"
    license            = "research-only-haiku-distillation"

    def iter_raw(self) -> Iterator[Dict[str, Any]]:
        path = self.root / "data" / "morph_v1_dep" / "train.jsonl"
        if not path.exists():
            return
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                yield json.loads(line)

    def normalize_row(self, raw: Dict[str, Any], idx: int) -> Optional[Sentence]:
        sentence = raw.get("sentence", "")
        items = raw.get("items", [])
        if not sentence or not items:
            return None

        normalized = normalize_text(sentence)

        # Build tokens. distill_v2 items carry per-word labels.
        tokens = []
        any_dep = False
        for i, it in enumerate(items):
            surf = it.get("word", "")
            tok = Token(
                index=i,
                surface=surf,
                normalized=arabic_normalize(surf),
                pos=LabelTag(value=it.get("pos") or None,
                              source=self.parser_origin,
                              confidence=0.85),
                case=LabelTag(value=it.get("case") or None,
                                source="haiku_distill",
                                confidence=0.85),
                role=LabelTag(value=it.get("role") or None,
                                source="haiku_distill",
                                confidence=0.85),
                marker=LabelTag(value=it.get("marker") or None,
                                source="haiku_distill",
                                confidence=0.85),
                dep_label=LabelTag(value=it.get("deprel") or None,
                                    source="stanza_ud",
                                    confidence=0.84),     # Stanza UAS
                governor_pos=it.get("governor_upos") or None,
            )
            head_idx = it.get("head_idx")
            if head_idx is not None and head_idx >= 0:
                tok.dep_head_idx = int(head_idx)
                any_dep = True
            tokens.append(tok)

        # Construction detection happens in a separate pass; populate empty
        # for now so the ``has_constructions`` completeness flag is False.
        constructions = []

        # Compute completeness
        comp = AnnotationCompleteness(
            has_morph=False,
            has_dep=any_dep,
            has_role=any(t.role.is_present for t in tokens),
            has_marker=any(t.marker.is_present for t in tokens),
            has_constructions=bool(constructions),
            has_clauses=False,
            has_reasoning=False,
            has_graph=False,
            has_discourse=False,
            has_alternative_parses=False,
        )
        n_present = sum(1 for t in tokens for f in (t.case, t.role, t.marker)
                        if f.is_present)
        comp.fields_complete_pct = n_present / max(3 * len(tokens), 1)

        meta = self._make_metadata(source_id_within=str(idx))

        return Sentence(
            raw_text=sentence,
            normalized_text=normalized,
            tokens=tokens,
            constructions=constructions,
            metadata=meta,
            completeness=comp,
        )
