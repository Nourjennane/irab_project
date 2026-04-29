"""Gazelle (UBC-NLP, EMNLP 2024) i'rāb subset loader.

Source: https://huggingface.co/datasets/UBC-NLP/gazelle_benchmark
File:   Iraab.jsonl  — manually curated MSA sentences with full traditional
        i'rāb prose answers. 40 high-quality items in the released split.

Sentence-level (one prose i'rāb per whole sentence), so this is **not**
suitable for the per-word seq2seq decoder. Use it as:

  1. **Few-shot prompts** for LLM-as-baseline (Stack A) and zero-shot eval.
  2. **Held-out eval set** for any model that produces full-sentence i'rāb.

We also expose the related Grammatical_Rules_Explanation file as auxiliary
instruction-tuning material for the LoRA route.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


GAZELLE_REPO = "UBC-NLP/gazelle_benchmark"


@dataclass
class GazelleItem:
    """One sentence-level i'rāb pair."""
    sentence: str            # the diacritized sentence (extracted from the prompt)
    answer: str              # full Arabic i'rāb prose
    raw_input: str           # the original instruction text
    task: str                # "Iraab" | "Grammatical rules and Definitions" | etc.


_SENT_RE = re.compile(r"الجملة\s*:\s*(.+?)$", flags=re.MULTILINE)


def _extract_sentence(prompt: str) -> Optional[str]:
    """Pull the sentence after `الجملة :` from a Gazelle prompt."""
    m = _SENT_RE.search(prompt)
    if m:
        return m.group(1).strip().rstrip(".،؛").strip()
    return None


def _hf_download(file: str, cache_dir: Path | str = "data/hf_cache") -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        GAZELLE_REPO, file, repo_type="dataset", cache_dir=str(cache_dir),
    ))


def load_gazelle_iraab(cache_dir: Path | str = "data/hf_cache") -> List[GazelleItem]:
    """Load the Iraab.jsonl items as (sentence, answer) pairs."""
    path = _hf_download("Iraab.jsonl", cache_dir=cache_dir)
    items: List[GazelleItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sent = _extract_sentence(row.get("Input", ""))
            ans = (row.get("Answer", "") or "").strip()
            if not sent or not ans:
                continue
            items.append(GazelleItem(
                sentence=sent, answer=ans,
                raw_input=row["Input"], task=row.get("Task", "Iraab"),
            ))
    return items


def load_gazelle_grammar_rules(cache_dir: Path | str = "data/hf_cache") -> List[GazelleItem]:
    """Auxiliary: grammar-rule explanations. Use for instruction tuning."""
    path = _hf_download("Grammatical_Rules_Explanation.jsonl", cache_dir=cache_dir)
    items: List[GazelleItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            items.append(GazelleItem(
                sentence=row.get("Input", ""),
                answer=(row.get("Answer", "") or "").strip(),
                raw_input=row["Input"],
                task=row.get("Task", "Grammar rules"),
            ))
    return items
