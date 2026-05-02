"""AraT5v2-base full-irab inference wrapper.

Loads a fine-tuned `irab_arat5_sft.py` checkpoint and runs per-word generation
in the same `WordIrab`-shaped output as the LLM baselines, so it slots into
`run_baselines.py` and the structural-extraction metrics with no special-case
plumbing.

Usage:
    from .arat5_irab import ArAT5IrabPredictor
    pred = ArAT5IrabPredictor("runs/irab_arat5v2_distill_v2_487235/final")
    items = pred.predict("ذهب الطالب إلى المدرسة")
    # → [WordIrab(word=..., irab=..., ...), ...]
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .llm_baselines import WordIrab


def _format_input(word: str, sentence: str, max_sent_len: int = 220) -> str:
    sent = (sentence or "").strip()
    if len(sent) > max_sent_len:
        sent = sent[:max_sent_len]
    return f"أعرب: {word.strip()} | في: {sent}"


# ---------------------------------------------------------------------------
# Lightweight diacritic-aware splitter (predict per surface word)
# ---------------------------------------------------------------------------
_DIAC_RE = re.compile(r"[ً-ْٰ]")
_NON_AR_RE = re.compile(r"[^ء-ي]+")


def _bare_word(s: str) -> str:
    return _NON_AR_RE.sub("", _DIAC_RE.sub("", s or ""))


def _split_into_words(sentence: str) -> List[str]:
    """Whitespace tokenisation; keep numbers / punct as separate tokens too."""
    return [t for t in re.split(r"\s+", (sentence or "").strip()) if t]


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
@dataclass
class ArAT5IrabPredictor:
    model_path: str
    max_input_length: int = 320
    max_target_length: int = 192
    device: Optional[str] = None       # "cuda" / "cpu" / None=auto

    def __post_init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        device_env = os.environ.get("ARAT5_DEVICE")
        if device_env:
            self.device = device_env
        elif self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [arat5_irab] loading {self.model_path} on {self.device}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def _generate_one(self, word: str, sentence: str) -> str:
        import torch
        text = _format_input(word, sentence)
        inputs = self.tok(text, return_tensors="pt",
                          truncation=True, max_length=self.max_input_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_target_length,
                do_sample=False,
                num_beams=1,
            )
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def predict(self, sentence: str) -> List[WordIrab]:
        out: List[WordIrab] = []
        for w in _split_into_words(sentence):
            irab = self._generate_one(w, sentence)
            out.append(WordIrab(word=w, irab=irab))
        return out
