"""Gazelle → schema_v2 loader (held-out MSA news evaluation surface).

Wraps the frozen-baseline ``irab_tashkeel.data.gazelle`` loader
+ ``irab_tashkeel.evaluation.structural.extract`` + the Step 16
kana-aware role extractor, producing schema_v2 sentences with
``annotation_quality=GOLD_HUMAN`` and the corrected role labels
(``ism_kana`` / ``khabar_kana`` properly extracted).

This loader is the canonical Gazelle source going forward — every
next-gen experiment that evaluates on Gazelle should consume from
the JSONL it produces, not from the legacy ``gazelle.py`` directly.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

from ..normalization import arabic_normalize, normalize_text
from ..schema_v2 import (
    AnnotationCompleteness, AnnotationQuality, Domain, LabelTag, Sentence, Token,
)
from .base import BaseLoader, register_loader


@register_loader
class GazelleLoader(BaseLoader):
    """Held-out MSA news evaluation set with the corrected kana extractor."""

    source_id          = "gazelle_test"
    domain             = Domain.MSA_NEWS.value
    annotation_quality = AnnotationQuality.GOLD_HUMAN.value
    parser_origin      = "human+structural_extractor_v2"
    license            = "research-only"

    def iter_raw(self) -> Iterator[Dict[str, Any]]:
        from irab_tashkeel.data.gazelle import load_gazelle_iraab
        from irab_tashkeel.evaluation.structural import split_sentence_iraab

        for it in load_gazelle_iraab():
            pairs = split_sentence_iraab(it.answer)
            if not pairs: continue
            yield {
                "sentence": it.sentence,
                "pairs": pairs,
                "raw_answer": it.answer,
            }

    def normalize_row(self, raw: Dict[str, Any], idx: int) -> Optional[Sentence]:
        from irab_tashkeel.evaluation.structural import extract
        from irab_tashkeel.structured.schema import canonicalize_role

        sentence = raw["sentence"]
        pairs = raw["pairs"]
        if not sentence or not pairs:
            return None

        # The same case + marker normalisation tables used in
        # eval_per_construction.py — keep the canonicalisation
        # consistent with the frozen-baseline evaluator.
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
        for i, (w, irab_prose) in enumerate(pairs):
            ext = extract(irab_prose)
            grole = canonicalize_role(ext.role) if (ext and ext.role) else None
            gcase = _nc(ext.case) if ext else None
            gmarker = _nm(ext.marker) if ext else None

            tok = Token(
                index=i,
                surface=w,
                normalized=arabic_normalize(w),
                case=LabelTag(value=gcase, source="gold_human",
                                confidence=1.0 if gcase else 0.0) if gcase else LabelTag(),
                role=LabelTag(value=grole, source="gold_human+structural_extractor_v2",
                                confidence=1.0 if grole else 0.0) if grole else LabelTag(),
                marker=LabelTag(value=gmarker, source="gold_human",
                                  confidence=1.0 if gmarker else 0.0) if gmarker else LabelTag(),
                notes=[f"raw_irab: {irab_prose[:120]}"],
            )
            tokens.append(tok)

        comp = AnnotationCompleteness(
            has_morph=False,
            has_dep=False,
            has_role=any(t.role.is_present for t in tokens),
            has_marker=any(t.marker.is_present for t in tokens),
            has_constructions=False,
            has_clauses=False,
            has_reasoning=False,
            has_graph=False,
            has_discourse=False,
        )
        n_present = sum(1 for t in tokens for f in (t.case, t.role, t.marker)
                        if f.is_present)
        comp.fields_complete_pct = n_present / max(3 * len(tokens), 1)

        meta = self._make_metadata(source_id_within=str(idx))

        return Sentence(
            raw_text=sentence,
            normalized_text=normalize_text(sentence),
            tokens=tokens,
            metadata=meta,
            completeness=comp,
        )
