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
    # Graph emission (off by default; trainer flips it on)
    emit_graph:       bool = False
    keep_edge_types:  Optional[set] = None   # None = all; else subset of {1,2,3,4,5,6,7,8}


# Edge type ids — must stay aligned with models.graph_refiner.EDGE_TYPES
# and grammar_graph.sparse.EDGE_TYPE_TO_ID.
EDGE_DEP                 = 1
EDGE_AGREEMENT           = 2
EDGE_CONSTRUCTION_MEMBER = 3
EDGE_CLAUSE_MEMBER       = 4
EDGE_GOVERNOR            = 5
EDGE_OVERLAP             = 6
EDGE_DISCOURSE           = 7
EDGE_COREF               = 8


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
                # Reject self-loops (head == j): some distill_v2 data carries
                # spurious self-loops (e.g. tokens 0/1/8 of certain sentences
                # report their own index as the head — likely a 1-vs-0-index
                # bug in the upstream parser). Self-loops would also clash
                # with the diagonal mask in the governor head, producing
                # +inf CE.
                head = item["dep_heads"][j]
                if (head is not None and head >= 0 and head < n_kept
                        and head != j):
                    dep_head_labels[i, j] = head

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

        # ----- Graph emission (Step 1 of structural reasoning upgrade) -----
        # Build a dense (B, max_words, max_words) edge-type matrix where
        # cell [b, i, j] = edge type id (0 = no edge). The graph_refiner
        # consumes this directly. We do NOT recompute graph structure in
        # the model forward.
        if self.config.emit_graph:
            B = len(batch)
            edge_index = torch.zeros((B, max_words, max_words), dtype=torch.long)
            for i, item in enumerate(batch):
                n_kept = len(kept_words_per_item[i])
                self._populate_word_edges(
                    edge_index[i], item, n_kept,
                    keep_edge_types=self.config.keep_edge_types,
                )
            out["word_edge_index"] = edge_index
        return out

    def _populate_word_edges(
        self, mat: "torch.Tensor", item: Dict[str, Any], n_kept: int,
        keep_edge_types: Optional[set] = None,
    ) -> None:
        """Fill a (W, W) edge-type matrix in place from the dataset item.

        Edge sources:
          - ``dep_heads[j]`` → bidirectional dep edge (j ↔ head)
          - constructions   → clique across token_indices (type 3)
          - construction overlap (≥ 2 share a token) → type 6
        """
        if n_kept == 0:
            return
        keep = keep_edge_types

        def _set(i: int, j: int, etype: int) -> None:
            if i == j or i >= n_kept or j >= n_kept or i < 0 or j < 0:
                return
            if keep is not None and etype not in keep:
                return
            # Don't overwrite a stronger edge with a weaker one. Priority
            # order: dep > construction_member > overlap > everything else.
            cur = int(mat[i, j].item())
            if cur != 0 and cur < etype:
                # prefer the lower-numbered (more structural) edge type
                return
            mat[i, j] = etype

        # Dep edges (bidirectional so message passes both ways)
        dep_heads = item.get("dep_heads", [])
        for j in range(min(n_kept, len(dep_heads))):
            head = dep_heads[j]
            if isinstance(head, int) and 0 <= head < n_kept:
                _set(j, head, EDGE_DEP)
                _set(head, j, EDGE_DEP)

        # Construction edges + overlap detection
        constructions = item.get("constructions", [])
        membership: Dict[int, int] = {}     # token_idx → count of constructions
        for c in constructions:
            idxs = [i for i in c.get("token_indices", []) if 0 <= i < n_kept]
            for i in idxs:
                membership[i] = membership.get(i, 0) + 1
            # Clique within construction
            for a in idxs:
                for b in idxs:
                    _set(a, b, EDGE_CONSTRUCTION_MEMBER)
        # Overlap: tokens belonging to ≥ 2 constructions get a stronger
        # signal — mark all pairs among them with EDGE_OVERLAP.
        overlap_tokens = [t for t, c in membership.items() if c >= 2]
        for a in overlap_tokens:
            for b in overlap_tokens:
                _set(a, b, EDGE_OVERLAP)
