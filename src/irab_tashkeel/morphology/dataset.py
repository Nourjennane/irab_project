"""Masked multi-task dataset for Phase 1.

Reads the merged corpus produced by ``merge_corpora.py``: each line is a JSON
record carrying a sentence + per-word labels for either i'rāb (when
``has_irab=True``) OR morphology (when ``has_morph=True``) — never both, in
Phase 1.

Masking strategy (frozen for Phase 1):
* When ``has_irab=False``: all four i'rāb head labels (case/role/marker/pos)
  are set to ``IGNORE_INDEX = -100``. CrossEntropyLoss with ``ignore_index``
  handles the per-token mask.
* When ``has_morph=False``: all seven morph head labels are set to ``-100``.
* If a batch happens to contain only one source, the heads of the other
  source produce a ``NaN``/empty-mean loss; the trainer guards this by
  zeroing heads whose entire batch is ignored.

Tokenization is identical to rev 2's :class:`StructuredIrabDataset`: per-word
SentencePiece encode + first-subtoken pooling. We deliberately reuse the
existing collator and add no new tokenization paths so any rev-2 reproduction
still works byte-for-byte.

The class is **separate** from ``StructuredIrabDataset``; rev 2's training
path imports the original. Phase 1's training path imports this one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ..structured.dataset import IGNORE  # = -100
from ..structured.schema import (
    CASE_TO_ID, ROLE_TO_ID, MARKER_TO_ID, POS_TO_ID,
)
from .schema import (
    GENDER_TO_ID, NUMBER_TO_ID, DEFINITE_TO_ID, PERSON_TO_ID,
    ASPECT_TO_ID, MOOD_TO_ID, VOICE_TO_ID,
    MORPH_FEATURES, UPOS_TO_CANONICAL_POS,
)


_MORPH_TO_ID = {
    "gender": GENDER_TO_ID,
    "number": NUMBER_TO_ID,
    "definite": DEFINITE_TO_ID,
    "person": PERSON_TO_ID,
    "aspect": ASPECT_TO_ID,
    "mood": MOOD_TO_ID,
    "voice": VOICE_TO_ID,
}


class MorphAwareStructuredIrabDataset(Dataset):
    """Joint i'rāb + morphology dataset reading the Phase 1 merged corpus.

    Output keys per example are the union of rev 2's keys + per-morph-feature
    label tensors. Examples without i'rāb labels have their i'rāb labels set
    to IGNORE; examples without morph labels have their morph labels set to
    IGNORE.
    """

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
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
        # Phase 4a: parametrised role_to_id (v3 default = rev 2 / Phase 1 path).
        self._role_to_id = ROLE_TO_ID if role_to_id is None else role_to_id

        self._records: List[Dict] = []
        n_skipped_long = 0
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                items = rec.get("items") or []
                if not items:
                    continue
                if skip_long and len(items) > max_words:
                    n_skipped_long += 1
                    continue
                self._records.append(self._encode(rec))
        if n_skipped_long:
            print(f"[MorphAwareDataset] skipped {n_skipped_long} sentences longer than {max_words} words")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict:
        return self._records[idx]

    def _encode(self, rec: Dict) -> Dict:
        has_irab = bool(rec.get("has_irab"))
        has_morph = bool(rec.get("has_morph"))
        items = rec["items"]

        ids: List[int] = []
        word_starts: List[int] = []
        word_ends: List[int] = []
        kept_words: List[str] = []

        # i'rāb labels
        case_lbl: List[int] = []
        role_lbl: List[int] = []
        marker_lbl: List[int] = []
        pos_lbl: List[int] = []

        # morph labels
        gender_lbl: List[int] = []
        number_lbl: List[int] = []
        definite_lbl: List[int] = []
        person_lbl: List[int] = []
        aspect_lbl: List[int] = []
        mood_lbl: List[int] = []
        voice_lbl: List[int] = []

        for w in items:
            sub = self.tokenizer.encode(w["word"], add_special_tokens=False)
            if not sub:
                continue
            start = len(ids)
            if start + len(sub) >= self.max_subwords - 1:
                break
            ids.extend(sub)
            word_starts.append(start)
            word_ends.append(len(ids))
            kept_words.append(w["word"])

            # i'rāb labels: use values when has_irab else IGNORE
            if has_irab:
                case_lbl.append(CASE_TO_ID.get(w.get("case"), IGNORE))
                role_lbl.append(self._role_to_id.get(w.get("role"), IGNORE))
                marker_lbl.append(MARKER_TO_ID.get(w.get("marker"), IGNORE))
                # POS source for has_irab examples is the iʿrāb "pos" field.
                pos_lbl.append(POS_TO_ID.get(w.get("pos"), IGNORE))
            else:
                case_lbl.append(IGNORE)
                role_lbl.append(IGNORE)
                marker_lbl.append(IGNORE)
                # On UD-PADT examples we DO want to train POS — UPOS gives us
                # a reliable POS label. So even when has_irab=False, fill POS
                # from the UD-derived "pos_ud" field if present.
                pos_ud = w.get("pos_ud")
                if pos_ud and pos_ud in POS_TO_ID:
                    pos_lbl.append(POS_TO_ID[pos_ud])
                else:
                    pos_lbl.append(IGNORE)

            # morph labels: use values when has_morph else IGNORE
            if has_morph:
                gender_lbl.append(GENDER_TO_ID.get(w.get("gender", "und"), IGNORE))
                number_lbl.append(NUMBER_TO_ID.get(w.get("number", "und"), IGNORE))
                definite_lbl.append(DEFINITE_TO_ID.get(w.get("definite", "und"), IGNORE))
                person_lbl.append(PERSON_TO_ID.get(w.get("person", "und"), IGNORE))
                aspect_lbl.append(ASPECT_TO_ID.get(w.get("aspect", "und"), IGNORE))
                mood_lbl.append(MOOD_TO_ID.get(w.get("mood", "und"), IGNORE))
                voice_lbl.append(VOICE_TO_ID.get(w.get("voice", "und"), IGNORE))
            else:
                gender_lbl.append(IGNORE); number_lbl.append(IGNORE)
                definite_lbl.append(IGNORE); person_lbl.append(IGNORE)
                aspect_lbl.append(IGNORE); mood_lbl.append(IGNORE)
                voice_lbl.append(IGNORE)

        if self.eos_id is not None:
            ids.append(int(self.eos_id))
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
            "gender_labels": torch.tensor(gender_lbl, dtype=torch.long),
            "number_labels": torch.tensor(number_lbl, dtype=torch.long),
            "definite_labels": torch.tensor(definite_lbl, dtype=torch.long),
            "person_labels": torch.tensor(person_lbl, dtype=torch.long),
            "aspect_labels": torch.tensor(aspect_lbl, dtype=torch.long),
            "mood_labels": torch.tensor(mood_lbl, dtype=torch.long),
            "voice_labels": torch.tensor(voice_lbl, dtype=torch.long),
            "has_irab": int(has_irab),
            "has_morph": int(has_morph),
            "sentence": rec.get("sentence", ""),
            "words": kept_words,
        }


@dataclass
class MorphAwareCollator:
    """Pads input_ids, word indices, and ALL label tensors to batch max."""

    pad_token_id: int

    def __call__(self, batch: Sequence[Dict]) -> Dict:
        max_t = max(int(b["input_ids"].size(0)) for b in batch)
        max_w = max(int(b["word_starts"].size(0)) for b in batch)
        bsz = len(batch)

        def _pad_t(key, fill, dtype=torch.long):
            out = torch.full((bsz, max_t), fill, dtype=dtype)
            for i, b in enumerate(batch):
                t = int(b[key].size(0))
                out[i, :t] = b[key]
            return out

        def _pad_w(key, fill, dtype=torch.long):
            out = torch.full((bsz, max_w), fill, dtype=dtype)
            for i, b in enumerate(batch):
                w = int(b[key].size(0))
                out[i, :w] = b[key]
            return out

        word_mask = torch.zeros((bsz, max_w), dtype=torch.long)
        for i, b in enumerate(batch):
            word_mask[i, : int(b["word_starts"].size(0))] = 1

        out = {
            "input_ids": _pad_t("input_ids", self.pad_token_id),
            "attention_mask": _pad_t("attention_mask", 0),
            "word_starts": _pad_w("word_starts", 0),
            "word_ends": _pad_w("word_ends", 0),
            "word_mask": word_mask,
            "case_labels":   _pad_w("case_labels",   IGNORE),
            "role_labels":   _pad_w("role_labels",   IGNORE),
            "marker_labels": _pad_w("marker_labels", IGNORE),
            "pos_labels":    _pad_w("pos_labels",    IGNORE),
            "gender_labels":   _pad_w("gender_labels",   IGNORE),
            "number_labels":   _pad_w("number_labels",   IGNORE),
            "definite_labels": _pad_w("definite_labels", IGNORE),
            "person_labels":   _pad_w("person_labels",   IGNORE),
            "aspect_labels":   _pad_w("aspect_labels",   IGNORE),
            "mood_labels":     _pad_w("mood_labels",     IGNORE),
            "voice_labels":    _pad_w("voice_labels",    IGNORE),
            "has_irab": torch.tensor([b["has_irab"] for b in batch], dtype=torch.long),
            "has_morph": torch.tensor([b["has_morph"] for b in batch], dtype=torch.long),
        }
        return out
