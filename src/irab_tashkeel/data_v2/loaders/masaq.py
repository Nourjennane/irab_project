"""MASAQ → schema_v2 loader (Quranic register).

Ingests ``data/masaq_eval.jsonl`` into schema_v2 sentences with
``domain=QURANIC`` and ``annotation_quality=GOLD_HUMAN`` (MASAQ
gold is human-annotated). Iʿrāb prose extraction reuses the
frozen-baseline structural extractor (with the Step 16 kana fix).

Quranic-specific behaviour
--------------------------

- **Normalisation:** uses ``quranic_normalize`` for ``surface`` →
  ``normalized`` so alif wasla (ٱ) and alif maqsura (ى) distinctions
  are preserved. Diacritics in the source are stripped only when
  they are full tashkīl; Quranic small marks are left in place.
- **Register metadata:** sets ``metadata.source = "masaq_quranic"``
  and tracks per-sentence ``sura_verse`` in ``metadata.source_id``.
- **Construction-density:** populated by a follow-up
  ``constructions.detector`` pass; the loader itself does not run it.
- **Semantic-pressure metadata:** populated by
  ``metadata.difficulty.populate_metadata``; loaders are deterministic
  and don't compute it.

The MASAQ JSONL row format::

    {
      "sentence": "<arabic surface>",
      "items": [{"word": "...", "irab": "<irab prose>"}, ...],
      "source": "masaq",
      "sura_verse": "15:67",
    }
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional

from ..normalization import quranic_normalize, normalize_text
from ..schema_v2 import (
    AnnotationCompleteness, AnnotationQuality, Domain, LabelTag, Sentence, Token,
)
from .base import BaseLoader, register_loader


@register_loader
class MasaqLoader(BaseLoader):
    """MASAQ — Quranic Arabic with hand-curated iʿrāb prose annotations."""

    source_id          = "masaq_quranic"
    domain             = Domain.QURANIC.value
    annotation_quality = AnnotationQuality.GOLD_HUMAN.value
    parser_origin      = "human+structural_extractor_v2"
    license            = "research-only-quranic-corpus"

    def iter_raw(self) -> Iterator[Dict[str, Any]]:
        path = self.root / "data" / "masaq_eval.jsonl"
        if not path.exists():
            return
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                yield json.loads(line)

    def normalize_row(self, raw: Dict[str, Any], idx: int) -> Optional[Sentence]:
        from irab_tashkeel.evaluation.structural import extract
        from irab_tashkeel.structured.schema import canonicalize_role

        sentence = raw.get("sentence", "")
        items = raw.get("items", [])
        sura_verse = raw.get("sura_verse", "")
        if not sentence or not items:
            return None

        # Same case + marker normalisation as Gazelle / eval_per_construction
        CASE_NORM = {
            "marfu": "raf", "mansub": "nasb", "majrur": "jarr",
            "majzum": "jazm", "mabni": "mabni", "raf": "raf",
            "nasb": "nasb", "jarr": "jarr", "jazm": "jazm",
        }
        MARKER_NORM = {
            "الضمة الظاهرة": "damma_visible", "الضمة المقدرة": "damma_hidden",
            "الفتحة الظاهرة": "fatha_visible", "الفتحة المقدرة": "fatha_hidden",
            "الكسرة الظاهرة": "kasra_visible", "الكسرة المقدرة": "kasra_hidden",
            "تنوين الضم": "tanween_damm", "تنوين الفتح": "tanween_fath",
            "تنوين الكسر": "tanween_kasr",
            "السكون": "sukun", "السكون المقدر": "sukun_hidden",
            "الياء": "ya", "الواو": "waw", "الألف": "alif",
            "النون": "nun", "الفتح": "fath_short",
        }

        def _nc(c):
            return CASE_NORM.get((c or "").strip(), c)

        def _nm(m):
            if not m: return None
            m = m.strip()
            if m in MARKER_NORM: return MARKER_NORM[m]
            for k, v in MARKER_NORM.items():
                if k in m: return v
            return m

        tokens = []
        for i, it in enumerate(items):
            w = it.get("word", "")
            irab_prose = it.get("irab", "")
            if not w:
                continue
            ext = extract(irab_prose)
            grole = canonicalize_role(ext.role) if (ext and ext.role) else None
            gcase = _nc(ext.case) if ext else None
            gmarker = _nm(ext.marker) if ext else None

            tok = Token(
                index=i,
                surface=w,
                normalized=quranic_normalize(w),
                case=LabelTag(value=gcase, source="gold_human",
                                confidence=1.0 if gcase else 0.0) if gcase else LabelTag(),
                role=LabelTag(value=grole,
                                source="gold_human+structural_extractor_v2",
                                confidence=1.0 if grole else 0.0) if grole else LabelTag(),
                marker=LabelTag(value=gmarker, source="gold_human",
                                  confidence=1.0 if gmarker else 0.0) if gmarker else LabelTag(),
                notes=[f"raw_irab: {irab_prose[:120]}"],
            )
            tokens.append(tok)

        comp = AnnotationCompleteness(
            has_morph=False, has_dep=False,
            has_role=any(t.role.is_present for t in tokens),
            has_marker=any(t.marker.is_present for t in tokens),
            has_constructions=False, has_clauses=False,
            has_reasoning=False, has_graph=False, has_discourse=False,
        )
        n_present = sum(1 for t in tokens for f in (t.case, t.role, t.marker)
                        if f.is_present)
        comp.fields_complete_pct = n_present / max(3 * len(tokens), 1)

        meta = self._make_metadata(source_id_within=sura_verse or str(idx))
        # Override per-layer origin
        meta.morph_origin = ""
        meta.dep_origin = ""
        meta.role_origin = "gold_human+structural_extractor_v2"
        meta.marker_origin = "gold_human"

        return Sentence(
            raw_text=sentence,
            normalized_text=normalize_text(sentence),
            tokens=tokens,
            metadata=meta,
            completeness=comp,
        )
