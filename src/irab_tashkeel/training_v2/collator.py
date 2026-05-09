"""Batch collator for schema_v2 → torch tensors.

Tokenises with the encoder's tokeniser, aligns word-level labels
to subword positions via first-subtoken pooling (matching
``structured/model._word_first_pool``), and produces padded
tensors for every multi-task head:

  - case / role / marker / pos (one int per word; IGNORE for unset)
  - morph axes: gender / number / definite / person / aspect / mood / voice
  - dep_head_idx (one int per word; IGNORE = -100)

The collator is encoder-agnostic — the tokeniser is passed in.
That lets the same collator support AraT5v2 / AraBART / CAMeLBERT
without a per-encoder copy.

Label IDs use the existing :mod:`irab_tashkeel.structured.schema`
canonical taxonomy. New label values that don't appear in the
schema's set are mapped to IGNORE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..structured.schema import (
    CASE_LABELS, MARKER_LABELS, POS_LABELS, ROLE_LABELS,
    CASE_TO_ID, MARKER_TO_ID, ROLE_TO_ID, POS_TO_ID,
)

IGNORE = -100      # PyTorch CrossEntropyLoss default ignore index


# ===========================================================================
# Morph axis vocabularies (parallel structure to existing morphology heads)
# ===========================================================================

GENDER_VOCAB   = ["masc", "fem", "common", "und"]
NUMBER_VOCAB   = ["sg", "dual", "plural", "und"]
DEFINITE_VOCAB = ["definite", "indefinite", "construct", "und"]
PERSON_VOCAB   = ["1", "2", "3", "und"]
ASPECT_VOCAB   = ["imperfective", "perfective", "und"]
MOOD_VOCAB     = ["indicative", "subjunctive", "jussive", "imperative", "und"]
VOICE_VOCAB    = ["active", "passive", "und"]

MORPH_VOCABS = {
    "gender":   GENDER_VOCAB,
    "number":   NUMBER_VOCAB,
    "definite": DEFINITE_VOCAB,
    "person":   PERSON_VOCAB,
    "aspect":   ASPECT_VOCAB,
    "mood":     MOOD_VOCAB,
    "voice":    VOICE_VOCAB,
}

MORPH_TO_ID = {
    axis: {v: i for i, v in enumerate(vocab)}
    for axis, vocab in MORPH_VOCABS.items()
}


def _label_id(value: Optional[str], vocab_to_id: Dict[str, int]) -> int:
    """Map a label string to its int id; IGNORE if unset / unknown."""
    if value is None or value == "":
        return IGNORE
    return vocab_to_id.get(value, IGNORE)


# ===========================================================================
# Collator
# ===========================================================================

@dataclass
class CollatorConfig:
    max_subtokens:    int = 320
    pad_token_id:     int = 0       # set by tokeniser-specific instantiation
    ignore_index:     int = IGNORE


class SchemaV2Collator:
    """Encoder-agnostic collator.

    Args
    ----
    tokenizer : a HuggingFace tokenizer (already loaded)
    config    : :class:`CollatorConfig`
    """

    def __init__(self, tokenizer, config: Optional[CollatorConfig] = None):
        self.tokenizer = tokenizer
        self.config = config or CollatorConfig(
            pad_token_id=tokenizer.pad_token_id or 0,
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            import torch
        except ImportError:
            raise ImportError("collator requires torch")

        max_sub = self.config.max_subtokens
        ignore  = self.config.ignore_index
        pad_id  = self.config.pad_token_id

        # Per-item subword tokenisation + word-start tracking
        all_input_ids: List[List[int]] = []
        all_word_starts: List[List[int]] = []
        all_word_ends:   List[List[int]] = []
        kept_words_per_item: List[List[str]] = []

        for item in batch:
            words = item["words"]
            ids: List[int] = []
            starts: List[int] = []
            ends: List[int] = []
            kept: List[str] = []
            for w in words:
                sub = self.tokenizer.encode(w, add_special_tokens=False)
                if not sub:
                    continue
                start = len(ids)
                if start + len(sub) >= max_sub - 1:
                    break
                ids.extend(sub)
                starts.append(start)
                ends.append(len(ids))
                kept.append(w)
            eos_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
            if eos_id is not None:
                ids.append(int(eos_id))
            all_input_ids.append(ids)
            all_word_starts.append(starts)
            all_word_ends.append(ends)
            kept_words_per_item.append(kept)

        # Pad subword sequences
        max_len = max((len(x) for x in all_input_ids), default=1)
        max_words = max((len(w) for w in kept_words_per_item), default=1)

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        word_starts = torch.zeros((len(batch), max_words), dtype=torch.long)
        word_ends   = torch.zeros((len(batch), max_words), dtype=torch.long)
        word_mask   = torch.zeros((len(batch), max_words), dtype=torch.long)

        for i, ids in enumerate(all_input_ids):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        for i, starts in enumerate(all_word_starts):
            word_starts[i, : len(starts)] = torch.tensor(starts, dtype=torch.long)
            word_mask[i, : len(starts)] = 1
        for i, ends in enumerate(all_word_ends):
            word_ends[i, : len(ends)] = torch.tensor(ends, dtype=torch.long)

        # Per-word labels
        case_labels   = torch.full((len(batch), max_words), ignore, dtype=torch.long)
        role_labels   = torch.full((len(batch), max_words), ignore, dtype=torch.long)
        marker_labels = torch.full((len(batch), max_words), ignore, dtype=torch.long)
        pos_labels    = torch.full((len(batch), max_words), ignore, dtype=torch.long)
        morph_labels  = {
            axis: torch.full((len(batch), max_words), ignore, dtype=torch.long)
            for axis in MORPH_VOCABS
        }
        dep_head_labels = torch.full((len(batch), max_words), ignore, dtype=torch.long)

        for i, item in enumerate(batch):
            kept = kept_words_per_item[i]
            n_kept = len(kept)
            # The truncation may drop words past the subtoken limit; honour
            # n_kept rather than the original word list length.
            for j in range(n_kept):
                case_labels[i, j]   = _label_id(item["case"][j], CASE_TO_ID)
                role_labels[i, j]   = _label_id(item["role"][j], ROLE_TO_ID)
                marker_labels[i, j] = _label_id(item["marker"][j], MARKER_TO_ID)
                pos_labels[i, j]    = _label_id(item["pos"][j], POS_TO_ID)
                for axis in MORPH_VOCABS:
                    morph_labels[axis][i, j] = _label_id(
                        item["morph"][axis][j], MORPH_TO_ID[axis]
                    )
                # dep_head — UD-PADT 0-based; IGNORE for -1, else the index
                head = item["dep_heads"][j]
                if head is not None and head >= 0 and head < n_kept:
                    dep_head_labels[i, j] = head
                # head == -2 (root marker) → IGNORE; per-word root prediction
                # is not a head we train against in this iteration

        out = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "word_starts":    word_starts,
            "word_ends":      word_ends,
            "word_mask":      word_mask,
            "case_labels":    case_labels,
            "role_labels":    role_labels,
            "marker_labels":  marker_labels,
            "pos_labels":     pos_labels,
            "dep_head_labels": dep_head_labels,
            "sentence_ids":   [item["sentence_id"] for item in batch],
        }
        for axis, t in morph_labels.items():
            out[f"morph_{axis}_labels"] = t
        return out
