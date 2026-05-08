"""PyTorch Dataset for the structured i'rāb corpus.

Reads the sentence-level JSONL produced by
``scripts/structured/build_structured_corpus.py``, tokenizes each sentence with
the AraT5v2 SentencePiece tokenizer, and emits per-word label tensors aligned
to subword positions.

Word-level alignment strategy: tokenize each whitespace-separated word
independently, concatenate the subword IDs (with EOS at the end), and record
each word's (start, end) subword index. At forward-time the model gathers
hidden states for each word's span and pools (mean) them.

This avoids the offset-mapping ambiguity that SentencePiece sometimes
introduces around mixed-script tokens (Arabic + punctuation/digits).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .schema import (
    CASE_TO_ID, ROLE_TO_ID, MARKER_TO_ID, POS_TO_ID,
    N_CASE, N_ROLE, N_MARKER, N_POS,
)
from .word_irab import SentenceIrab

# CrossEntropyLoss ignore_index for unlabeled / padded word slots.
IGNORE = -100


@dataclass
class StructuredExample:
    """One tokenized + aligned sentence ready for the model."""

    input_ids: torch.LongTensor               # (T,)
    attention_mask: torch.LongTensor          # (T,)
    word_starts: torch.LongTensor             # (W,) inclusive subword index of each word's first subword
    word_ends: torch.LongTensor               # (W,) exclusive end (i.e. word i covers [word_starts[i], word_ends[i]))
    case_labels: torch.LongTensor             # (W,)
    role_labels: torch.LongTensor             # (W,)
    marker_labels: torch.LongTensor           # (W,)
    pos_labels: torch.LongTensor              # (W,)
    sentence: str
    words: List[str]


class StructuredIrabDataset(Dataset):
    """Reads a structured-corpus JSONL produced by ``build_structured_corpus.py``."""

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        *,
        max_subwords: int = 320,
        max_words: int = 64,
        skip_long: bool = True,
        role_to_id: Optional[Dict[str, int]] = None,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_subwords = max_subwords
        self.max_words = max_words
        self.skip_long = skip_long
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
        # Phase 4a: role_to_id is parametrised so v3 (rev 2 / Phase 1) and v4
        # (Phase 4a) can share this dataset class. Default = v3 ROLE_TO_ID,
        # which preserves the rev 2 + Phase 1 path byte-identical.
        self._role_to_id = ROLE_TO_ID if role_to_id is None else role_to_id

        self._records: List[Dict] = []
        n_skipped_long = 0
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = SentenceIrab.from_json_line(line)
                if not rec.items:
                    continue
                if skip_long and len(rec.items) > max_words:
                    n_skipped_long += 1
                    continue
                self._records.append(self._encode(rec))
        if n_skipped_long:
            print(f"[StructuredIrabDataset] skipped {n_skipped_long} sentences longer than {max_words} words")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict:
        return self._records[idx]

    # --- encoding ------------------------------------------------------
    def _encode(self, rec: SentenceIrab) -> Dict:
        ids: List[int] = []
        word_starts: List[int] = []
        word_ends: List[int] = []
        words: List[str] = []
        case_lbl: List[int] = []
        role_lbl: List[int] = []
        marker_lbl: List[int] = []
        pos_lbl: List[int] = []

        for w in rec.items:
            sub = self.tokenizer.encode(w.word, add_special_tokens=False)
            if not sub:
                continue
            start = len(ids)
            if start + len(sub) >= self.max_subwords - 1:  # -1 reserves EOS slot
                break
            ids.extend(sub)
            word_starts.append(start)
            word_ends.append(len(ids))
            words.append(w.word)
            case_lbl.append(CASE_TO_ID[w.case] if w.case in CASE_TO_ID else IGNORE)
            role_lbl.append(self._role_to_id[w.role] if w.role in self._role_to_id else IGNORE)
            marker_lbl.append(MARKER_TO_ID[w.marker] if w.marker in MARKER_TO_ID else IGNORE)
            pos_lbl.append(POS_TO_ID[w.pos] if w.pos in POS_TO_ID else IGNORE)

        # Append EOS so attention has a clean terminator.
        if self.eos_id is not None:
            ids.append(self.eos_id)
        attention_mask = [1] * len(ids)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "word_starts": torch.tensor(word_starts, dtype=torch.long),
            "word_ends": torch.tensor(word_ends, dtype=torch.long),
            "case_labels": torch.tensor(case_lbl, dtype=torch.long),
            "role_labels": torch.tensor(role_lbl, dtype=torch.long),
            "marker_labels": torch.tensor(marker_lbl, dtype=torch.long),
            "pos_labels": torch.tensor(pos_lbl, dtype=torch.long),
            "sentence": rec.sentence,
            "words": words,
        }


@dataclass
class StructuredCollator:
    """Pads input_ids and word-level tensors to the batch max."""

    pad_token_id: int

    def __call__(self, batch: Sequence[Dict]) -> Dict:
        max_t = max(int(b["input_ids"].size(0)) for b in batch)
        max_w = max(int(b["word_starts"].size(0)) for b in batch)
        bsz = len(batch)

        input_ids = torch.full((bsz, max_t), self.pad_token_id, dtype=torch.long)
        attn = torch.zeros((bsz, max_t), dtype=torch.long)
        word_starts = torch.zeros((bsz, max_w), dtype=torch.long)
        word_ends = torch.zeros((bsz, max_w), dtype=torch.long)
        word_mask = torch.zeros((bsz, max_w), dtype=torch.long)
        case_l = torch.full((bsz, max_w), IGNORE, dtype=torch.long)
        role_l = torch.full((bsz, max_w), IGNORE, dtype=torch.long)
        marker_l = torch.full((bsz, max_w), IGNORE, dtype=torch.long)
        pos_l = torch.full((bsz, max_w), IGNORE, dtype=torch.long)

        sentences: List[str] = []
        words_list: List[List[str]] = []
        for i, b in enumerate(batch):
            t = int(b["input_ids"].size(0))
            w = int(b["word_starts"].size(0))
            input_ids[i, :t] = b["input_ids"]
            attn[i, :t] = b["attention_mask"]
            word_starts[i, :w] = b["word_starts"]
            word_ends[i, :w] = b["word_ends"]
            word_mask[i, :w] = 1
            case_l[i, :w] = b["case_labels"]
            role_l[i, :w] = b["role_labels"]
            marker_l[i, :w] = b["marker_labels"]
            pos_l[i, :w] = b["pos_labels"]
            sentences.append(b["sentence"])
            words_list.append(b["words"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "word_starts": word_starts,
            "word_ends": word_ends,
            "word_mask": word_mask,
            "case_labels": case_l,
            "role_labels": role_l,
            "marker_labels": marker_l,
            "pos_labels": pos_l,
            "sentences": sentences,
            "words": words_list,
        }
