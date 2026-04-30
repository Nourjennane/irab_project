"""Mix A hybrid inference: Claude RAG (case + role) + AraT5v2 (marker).

Per the project's research bet, traditional Arabic i'rāb decomposes into
two asymmetric sub-tasks:

  - **Knowledge-bound** (case, role): a frontier LLM with retrieval
    captures these via its pretrained Arabic-grammar competence.
  - **Style-bound** (marker phrasing): a small specialist fine-tuned on
    Yarob + distilled marker pairs fits the corpus-specific surface form.

This module wires the two together. For each input sentence:

    1. Run Claude few-shot RAG → list of WordIrab (with case, role, marker)
    2. For each word: call AraT5v2 specialist with [case={c}] [role={r}] {word} | {sentence}
       → predicted marker_phrase (or "<NO_MARKER>" for indeclinable cases)
    3. If specialist's marker is non-NO_MARKER, replace RAG's marker.
    4. Rebuild the prose `irab` field from the canonical template.

Inference cost: 1 API call per sentence (RAG) + N local AraT5v2 forward
passes per sentence (~3 ms each on RTX 4060). Negligible vs the API call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .llm_baselines import (
    FewShotExample, WordIrab, claude_fewshot_rag, load_combined_fewshots,
)


NO_MARKER = "<NO_MARKER>"


# ---------------------------------------------------------------------------
# Prose-form rebuild from structured fields
# ---------------------------------------------------------------------------
_CASE_WORD = {
    "rafʿ":  ("مرفوع",  "رفعه"),
    "naṣb":  ("منصوب",  "نصبه"),
    "jarr":  ("مجرور",  "جره"),
    "jazm":  ("مجزوم",  "جزمه"),
}


def rebuild_irab(case: Optional[str], role: Optional[str], marker: Optional[str]) -> str:
    """Reconstruct the prose i'rāb from the structured fields + marker.

    Examples:
        ("rafʿ", "فاعل", "الضمة الظاهرة") → "فاعل مرفوع وعلامة رفعه الضمة الظاهرة"
        ("jarr", "اسم مجرور", "الكسرة الظاهرة")
            → "اسم مجرور وعلامة جره الكسرة الظاهرة"
        ("mabni", "حرف عطف", None) → "حرف عطف مبني لا محل له من الإعراب"

    For mabni / unknown cases we don't synthesize anything novel — we leave
    the caller's existing irab in place (caller checks).
    """
    role = (role or "").strip()
    if not role:
        return ""

    if case == "mabni":
        # Don't try to synthesize mabni prose; preserve whatever the LLM produced.
        return ""

    if case in _CASE_WORD:
        case_adj, case_gen = _CASE_WORD[case]
        if marker and marker != NO_MARKER:
            return f"{role} {case_adj} وعلامة {case_gen} {marker}"
        else:
            return f"{role} {case_adj}"

    return ""


# ---------------------------------------------------------------------------
# AraT5v2 marker specialist wrapper
# ---------------------------------------------------------------------------
class MarkerSpecialist:
    """Lazy-loaded wrapper around the AraT5v2 marker fine-tune."""

    def __init__(self, model_path: Path | str):
        self.model_path = Path(model_path)
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device).eval()

    @staticmethod
    def _format_input(word: str, sentence: str, case: Optional[str], role: Optional[str]) -> str:
        case = (case or "-").strip() or "-"
        role = (role or "-").strip() or "-"
        # Match the exact format used in marker_arat5_sft.py
        sent = sentence.strip()
        if len(sent) > 200:
            sent = sent[:200]
        return f"أعرب علامة: {word} | في: {sent} | الحالة: {case} | المحل: {role}"

    def predict(
        self,
        word: str,
        sentence: str,
        case: Optional[str],
        role: Optional[str],
    ) -> str:
        """Return the marker phrase for one word, or `<NO_MARKER>`."""
        self._ensure_loaded()
        import torch
        prompt = self._format_input(word, sentence, case, role)
        enc = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self._device)
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=32, do_sample=False, num_beams=1)
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip() or NO_MARKER

    def predict_batch(
        self,
        words: Sequence[str],
        sentences: Sequence[str],
        cases: Sequence[Optional[str]],
        roles: Sequence[Optional[str]],
    ) -> List[str]:
        """Batched marker prediction for one whole sentence (or several)."""
        self._ensure_loaded()
        import torch
        prompts = [
            self._format_input(w, s, c, r)
            for w, s, c, r in zip(words, sentences, cases, roles)
        ]
        enc = self._tokenizer(prompts, return_tensors="pt", padding=True,
                              truncation=True, max_length=256).to(self._device)
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=32, do_sample=False, num_beams=1)
        return [self._tokenizer.decode(o, skip_special_tokens=True).strip() or NO_MARKER for o in out]


# ---------------------------------------------------------------------------
# Hybrid predictor
# ---------------------------------------------------------------------------
class HybridPredictor:
    """Mix A: Claude RAG (case + role) + AraT5v2 (marker overlay)."""

    def __init__(
        self,
        marker_model_path: Path | str,
        rag_pool: Optional[Sequence[FewShotExample]] = None,
        rag_k: int = 5,
        rag_model: str = "claude-haiku-4-5",
    ):
        self.marker = MarkerSpecialist(marker_model_path)
        self.rag_pool = rag_pool if rag_pool is not None else load_combined_fewshots()
        self.rag_k = rag_k
        self.rag_model = rag_model

    def predict(self, sentence: str) -> List[WordIrab]:
        """Run RAG + marker overlay, return per-word i'rāb."""
        rag_items = claude_fewshot_rag(
            sentence, self.rag_pool, k=self.rag_k, model=self.rag_model,
        )
        if not rag_items:
            return []

        words = [it.word for it in rag_items]
        cases = [it.case for it in rag_items]
        roles = [it.role for it in rag_items]
        sents = [sentence] * len(rag_items)
        markers = self.marker.predict_batch(words, sents, cases, roles)

        out: List[WordIrab] = []
        for it, new_marker in zip(rag_items, markers):
            # Overlay only when the specialist returns a real marker AND
            # the case is declinable (mabni cases keep the LLM's prose intact).
            use_specialist = (
                new_marker and new_marker != NO_MARKER
                and it.case in {"rafʿ", "naṣb", "jarr", "jazm"}
            )
            if use_specialist:
                rebuilt = rebuild_irab(it.case, it.role, new_marker)
                if rebuilt:
                    it.marker = new_marker
                    it.irab = rebuilt
            out.append(it)
        return out
