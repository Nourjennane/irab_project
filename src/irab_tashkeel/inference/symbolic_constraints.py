"""Symbolic grammar constraints applied as logit-bias reranking.

Phase 3 / v1 rebuild — four lightweight constraint families, each implemented
as an additive logit bias on the model's per-word case (and optionally role)
logits. The bias is **soft**, never hard: a low-confidence model may still
override a constraint when the data disagrees.

Constraint families:

1. ``prep_to_jarr``   — after one of the canonical prepositions, the next noun
   gets a ``+λ`` bias on case=jarr and role=ism_majrur.
2. ``inna_sisters``   — after إن / أن / لكن / ليت / كأن / لعل, the first noun in
   the clause gets ``+λ`` on case=nasb + role=ism_inna; the next noun in the
   clause gets ``+λ`` on case=raf + role=khabar_inna.
3. ``kana_sisters``   — after كان / ليس / أصبح / ظل / صار / بات / أمسى / ما زال,
   the first noun gets ``+λ`` on case=raf + role=ism_kana; the next noun gets
   ``+λ`` on case=nasb + role=khabar_kana.
4. ``idafa_stub``     — for two consecutive bare-noun words (no determiner, no
   preposition between), bias the second toward case=jarr + role=mudaaf_ilayh.

The "first noun" heuristic walks left-to-right within the window after the
trigger particle, picking the first POS-predicted noun.

This module is pure tensor / torch — no Arabic NLP dependencies beyond the
hard-coded particle vocabularies.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from ..structured.schema import (
    CASE_TO_ID, ROLE_TO_ID, POS_TO_ID,
)


# ---------------------------------------------------------------------------
# Trigger particle vocabularies (fold-alif / fold-ya forms used at compare time)
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    # Strip diacritics
    s = "".join(c for c in s if not (0x064B <= ord(c) <= 0x0652) and ord(c) != 0x0670)
    # Drop tatweel
    s = s.replace("ـ", "")
    # Fold alif & ya variants for matching the trigger lookup
    return (
        s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
         .replace("ى", "ي")
    )


# Note: PREPS are listed in already-normalized form (alif+ya folded).
# E.g. "إلى" -> "الي" (NOT "الى"), "على" -> "علي".
PREPS: Set[str] = {
    "في", "من", "الي", "علي", "عن", "ب", "ل", "ك", "حتي", "منذ", "مذ",
    "خلا", "عدا", "حاشا",
}
# Single-letter prefixes ب and ل often appear glued to the next word;
# distill_v2 uses them as standalone "ب" / "ل" rows (per the role harf_jarr
# distribution), so a literal-match is what we need here.

# Inna sisters (إن، أن، لكن، ليت، لعل) — after alif fold these collapse to
# {ان, لكن, ليت, لعل}.  كأن also folds to كان which would collide with the
# kana trigger; we exclude it to avoid the ambiguity.
INNA_SISTERS: Set[str] = {"ان", "لكن", "ليت", "لعل"}

KANA_SISTERS: Set[str] = {
    "كان", "كانت", "كانوا", "كن", "يكون", "تكون", "اكون", "نكون",
    "ليس", "ليست", "ليسوا",
    "اصبح", "اصبحت",
    "ظل", "ظلت",
    "صار", "صارت",
    "بات", "باتت",
    "امسي", "امست",
    "مازال", "ماتزال", "مايزال",
}

# Definite article prefix ("ال").  After alif fold it stays "ال".
def _has_al_prefix(word_norm: str) -> bool:
    return word_norm.startswith("ال") and len(word_norm) > 2


# ---------------------------------------------------------------------------
# Constraint result bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class ConstraintTrace:
    """Per-word log of which constraint families fired.

    Indexed [batch_idx][word_idx] -> list of constraint names that fired.
    """
    fired: List[List[List[str]]] = field(default_factory=list)

    def init(self, batch_size: int, n_words: int):
        self.fired = [[[] for _ in range(n_words)] for _ in range(batch_size)]

    def add(self, b: int, w: int, name: str):
        if 0 <= b < len(self.fired) and 0 <= w < len(self.fired[b]):
            self.fired[b][w].append(name)


# ---------------------------------------------------------------------------
# Per-sentence constraint application (numpy/torch friendly)
# ---------------------------------------------------------------------------
def _argmax_pos(pos_logits_w: torch.Tensor) -> int:
    """Index of the model's predicted POS for a word."""
    return int(pos_logits_w.argmax().item())


def _is_predicted_noun_or_adj(pos_pred: int) -> bool:
    return pos_pred in (POS_TO_ID["noun"], POS_TO_ID["adjective"], POS_TO_ID["pronoun"])


def apply_constraints(
    case_logits: torch.Tensor,         # (B, W, N_CASE)
    role_logits: torch.Tensor,         # (B, W, N_ROLE)
    pos_logits: torch.Tensor,          # (B, W, N_POS)
    words: Sequence[Sequence[str]],    # B lists of W word strings
    word_mask: torch.Tensor,           # (B, W)
    lambda_case: float = 1.5,
    lambda_role: float = 0.8,
    enabled: Optional[Set[str]] = None,
    trace: Optional[ConstraintTrace] = None,
) -> Tuple[torch.Tensor, torch.Tensor, ConstraintTrace]:
    """Apply enabled symbolic constraints to case + role logits.

    Returns (new_case_logits, new_role_logits, trace).  Original tensors are
    not mutated; the caller can ablate constraints by passing
    ``enabled=set()`` (the no-op identity).
    """
    if enabled is None:
        enabled = {"prep_to_jarr", "inna_sisters", "kana_sisters", "idafa_stub"}

    case_logits = case_logits.clone()
    role_logits = role_logits.clone()
    B, W, _ = case_logits.shape
    if trace is None:
        trace = ConstraintTrace()
    trace.init(B, W)

    JARR = CASE_TO_ID["jarr"]
    NASB = CASE_TO_ID["nasb"]
    RAF = CASE_TO_ID["raf"]

    R_ISM_MAJRUR = ROLE_TO_ID["ism_majrur"]
    R_MUDAAF = ROLE_TO_ID["mudaaf_ilayh"]
    R_ISM_INNA = ROLE_TO_ID["ism_inna"]
    R_KHABAR_INNA = ROLE_TO_ID["khabar_inna"]
    R_ISM_KANA = ROLE_TO_ID["ism_kana"]
    R_KHABAR_KANA = ROLE_TO_ID["khabar_kana"]

    for b in range(B):
        ws = list(words[b])
        n = int(word_mask[b].sum().item())
        norm_words = [_norm(w) for w in ws[:n]]

        # ---------------- 1) prep -> jarr -----------------
        if "prep_to_jarr" in enabled:
            for i in range(1, n):
                if norm_words[i - 1] in PREPS:
                    pos_i = _argmax_pos(pos_logits[b, i])
                    if _is_predicted_noun_or_adj(pos_i):
                        case_logits[b, i, JARR] += lambda_case
                        role_logits[b, i, R_ISM_MAJRUR] += lambda_role
                        trace.add(b, i, "prep_to_jarr")

        # ---------------- 2) inna sisters -----------------
        if "inna_sisters" in enabled:
            inna_pos = next((j for j, w in enumerate(norm_words) if w in INNA_SISTERS), None)
            if inna_pos is not None:
                # find first POS=noun after inna -> ism inna (nasb)
                ism_idx = None
                for j in range(inna_pos + 1, n):
                    if _is_predicted_noun_or_adj(_argmax_pos(pos_logits[b, j])):
                        ism_idx = j
                        break
                if ism_idx is not None:
                    case_logits[b, ism_idx, NASB] += lambda_case
                    role_logits[b, ism_idx, R_ISM_INNA] += lambda_role
                    trace.add(b, ism_idx, "inna_ism_to_nasb")
                    # find next POS=noun -> khabar inna (raf)
                    for k in range(ism_idx + 1, n):
                        if _is_predicted_noun_or_adj(_argmax_pos(pos_logits[b, k])):
                            case_logits[b, k, RAF] += lambda_case
                            role_logits[b, k, R_KHABAR_INNA] += lambda_role
                            trace.add(b, k, "inna_khabar_to_raf")
                            break

        # ---------------- 3) kana sisters -----------------
        if "kana_sisters" in enabled:
            kana_pos = next((j for j, w in enumerate(norm_words) if w in KANA_SISTERS), None)
            if kana_pos is not None:
                ism_idx = None
                for j in range(kana_pos + 1, n):
                    if _is_predicted_noun_or_adj(_argmax_pos(pos_logits[b, j])):
                        ism_idx = j
                        break
                if ism_idx is not None:
                    case_logits[b, ism_idx, RAF] += lambda_case
                    role_logits[b, ism_idx, R_ISM_KANA] += lambda_role
                    trace.add(b, ism_idx, "kana_ism_to_raf")
                    for k in range(ism_idx + 1, n):
                        if _is_predicted_noun_or_adj(_argmax_pos(pos_logits[b, k])):
                            case_logits[b, k, NASB] += lambda_case
                            role_logits[b, k, R_KHABAR_KANA] += lambda_role
                            trace.add(b, k, "kana_khabar_to_nasb")
                            break

        # ---------------- 4) idafa stub -----------------
        if "idafa_stub" in enabled:
            for i in range(1, n):
                wi = norm_words[i]
                wim1 = norm_words[i - 1]
                if not wim1 or not wi:
                    continue
                # heuristic: previous word is a bare noun (no ال, not a particle)
                # and current word is bare (no ال), neither is a preposition.
                if wim1 in PREPS or wi in PREPS:
                    continue
                pi_prev = _argmax_pos(pos_logits[b, i - 1])
                pi_cur = _argmax_pos(pos_logits[b, i])
                if pi_prev != POS_TO_ID["noun"] or pi_cur != POS_TO_ID["noun"]:
                    continue
                if _has_al_prefix(wi):
                    continue
                # weaker bias than the explicit prep rule
                case_logits[b, i, JARR] += 0.5 * lambda_case
                role_logits[b, i, R_MUDAAF] += 0.5 * lambda_role
                trace.add(b, i, "idafa_stub")

    return case_logits, role_logits, trace
