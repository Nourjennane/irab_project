"""External reasoning-trace ingestor (Step 9 ingestion side).

Designed for future use: imports human-authored or LLM-generated
reasoning traces from external corpora (Arabic textbook scrapes,
exam-solution datasets, mawdoo3.com / arabic-tools.net pages with
full grammatical analyses) and merges them into schema_v2
sentences.

Currently reserved infrastructure — no external sources are
implemented yet. The interface is here so when a source lands
(e.g. a structured Arabic-textbook dataset), the ingestor can be
implemented as a single subclass without touching schema_v2.

Trace JSONL row format
----------------------

```json
{
  "sentence_text": "كان الطالب مجتهداً",
  "match_strategy": "surface_exact",   // surface_exact | normalized | fuzzy
  "reasoning_steps": [
    {
      "applies_to_type": "construction",
      "applies_to_pattern": "kana_completion",
      "justification": "كان: فعل ماضٍ ناقص مبني...",
      "derivation_chain": ["...", "..."],
      "transformation_logic": "kana_completion: ism→raf, khabar→nasb",
      "confidence": 1.0
    }
  ],
  "source": "<corpus_name>",
  "license": "<license>"
}
```

The ingestor matches each external sentence to a schema_v2
:class:`Sentence` (by surface text under appropriate
normalisation), then merges the reasoning_steps in.
"""
from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from ..data_v2.normalization import arabic_normalize, normalize_text
from ..data_v2.schema_v2 import ReasoningStep, Sentence


class BaseReasoningIngestor(abc.ABC):
    """Subclass to add a new external reasoning-trace source."""

    source_id: str = ""
    license: str = ""

    @abc.abstractmethod
    def iter_raw(self) -> Iterator[Dict[str, Any]]:
        """Yield raw rows from the external corpus."""

    @abc.abstractmethod
    def parse_row(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert one raw row to the canonical trace JSONL format
        (see module docstring). Return ``None`` to skip."""

    def ingest_into_sentences(
        self, sentences: List[Sentence],
    ) -> int:
        """Match each external trace to a schema_v2 sentence and
        append its reasoning_steps. Returns the count of sentences
        that received at least one reasoning step from this source.
        """
        # Build a lookup index by normalised text
        norm_idx: Dict[str, Sentence] = {}
        for s in sentences:
            key = normalize_text(s.raw_text)
            norm_idx.setdefault(key, s)

        n_matched = 0
        for raw in self.iter_raw():
            parsed = self.parse_row(raw)
            if parsed is None:
                continue
            key = normalize_text(parsed.get("sentence_text", ""))
            target = norm_idx.get(key)
            if target is None:
                # Try surface-folded fallback
                key2 = arabic_normalize(parsed.get("sentence_text", ""))
                norm2_idx = {arabic_normalize(s.raw_text): s for s in sentences}
                target = norm2_idx.get(key2)
            if target is None:
                continue
            # Append reasoning steps
            base_idx = max((rs.step_idx for rs in target.reasoning_steps),
                           default=-1) + 1
            for j, step in enumerate(parsed.get("reasoning_steps", [])):
                target.reasoning_steps.append(ReasoningStep(
                    step_idx=base_idx + j,
                    applies_to_type=step.get("applies_to_type", ""),
                    applies_to_id=step.get("applies_to_id", ""),
                    justification=step.get("justification", ""),
                    derivation_chain=list(step.get("derivation_chain", [])),
                    transformation_logic=step.get("transformation_logic", ""),
                    ambiguity_notes=step.get("ambiguity_notes", ""),
                    semantic_disambiguation=step.get("semantic_disambiguation", ""),
                    discourse_notes=step.get("discourse_notes", ""),
                    confidence=step.get("confidence", 0.9),
                    source=self.source_id,
                ))
            target.completeness.has_reasoning = True
            n_matched += 1
        return n_matched


# ===========================================================================
# Reference implementation: trace JSONL ingestor
# ===========================================================================

class JsonlReasoningIngestor(BaseReasoningIngestor):
    """Ingest from a flat JSONL file in the canonical trace format.

    Use this when an external source has been pre-processed into
    the canonical row shape; for raw web scrapes / textbook OCR,
    write a source-specific subclass that overrides ``parse_row``.
    """

    def __init__(self, path: str | Path, source_id: str = "external",
                 license: str = "research-only"):
        self.path = Path(path)
        self.source_id = source_id
        self.license = license

    def iter_raw(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                yield json.loads(line)

    def parse_row(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not raw.get("sentence_text"):
            return None
        if not raw.get("reasoning_steps"):
            return None
        return raw
