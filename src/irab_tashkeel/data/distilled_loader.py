"""Load LLM-distilled MSA i'rāb pairs into MTLExample format.

Reads the JSONL written by `irab_tashkeel.data.distill` and aligns each
teacher-generated `[{word, irab, ...}, ...]` array with its source sentence.
Examples whose word-count doesn't match the bare sentence are dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ..models.labels import IRAB_TO_ID
from ..models.tokenizer import compute_word_offsets, normalize, strip_diacritics
from .schema import MTLExample


def _normalize_word(w: str) -> str:
    return strip_diacritics(normalize(w or "")).strip(" .,،؛:؟!\"'«»{}[]()")


def load_distilled_examples(path: Path | str = "data/distilled_irab.jsonl") -> List[MTLExample]:
    path = Path(path)
    if not path.exists():
        return []
    examples: List[MTLExample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sentence = (row.get("sentence") or "").strip()
            items = row.get("items") or []
            if not sentence or not items:
                continue

            bare_text = strip_diacritics(normalize(sentence)).strip()
            bare_words = bare_text.split()
            if len(bare_words) < 2 or len(bare_words) > 60:
                continue

            # Positional alignment: only accept full coverage.
            if len(items) != len(bare_words):
                continue
            # Quality check: ≥70% of teacher words match (after normalization).
            ok = sum(
                1 for it, bw in zip(items, bare_words)
                if _normalize_word(it.get("word", "")) == _normalize_word(bw)
            )
            if ok < max(1, int(0.7 * len(bare_words))):
                continue

            irab_targets = [str(it.get("irab", "")).strip() for it in items]
            irab_ids = [IRAB_TO_ID["other"]] * len(bare_words)
            word_offsets = compute_word_offsets(bare_text)

            examples.append(MTLExample(
                bare_text=bare_text,
                diac_labels=[0] * len(bare_text),
                mask_diac=False,
                word_offsets=word_offsets,
                irab_labels=irab_ids,
                mask_irab=True,
                err_labels=[0] * len(bare_text),
                mask_err=False,
                source="distilled",
                sent_id=None,
                irab_targets=irab_targets,
            ))
    return examples
