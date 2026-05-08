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


ALL_CONSTRAINTS = {
    # Phase 3 (rev 2)
    "prep_to_jarr",
    "inna_sisters",
    "kana_sisters",
    "idafa_stub",
    # Phase 4 (added 2026-05-08)
    "adjective_agreement",        # noun raf -> following adj raf, etc.
    "coordination_share_case",    # X و Y share case
    "idafa_chain",                # mudaaf_ilayh -> jarr stronger
    "naat_propagation",           # naat inherits case from head noun
    "munadi_to_nasb",             # vocative noun -> nasb
}


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
        enabled = set(ALL_CONSTRAINTS)

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

        # ============== Phase 4 — stronger constraints ==============
        # ---------------- 5) adjective agreement -----------------
        # If word is predicted as adjective (POS=adjective) and the head noun
        # immediately precedes, copy the head noun's argmax case onto this
        # word's logits as a soft bias. Adjectives in Arabic agree with their
        # head noun's case.
        if "adjective_agreement" in enabled:
            ADJ = POS_TO_ID["adjective"]
            NOUN = POS_TO_ID["noun"]
            R_NAAT = ROLE_TO_ID["naat"]
            for i in range(1, n):
                pos_i = _argmax_pos(pos_logits[b, i])
                if pos_i != ADJ:
                    continue
                pos_prev = _argmax_pos(pos_logits[b, i - 1])
                if pos_prev != NOUN:
                    continue
                head_case = int(case_logits[b, i - 1].argmax().item())
                case_logits[b, i, head_case] += lambda_case
                role_logits[b, i, R_NAAT] += 0.5 * lambda_role
                trace.add(b, i, "adjective_agreement")

        # ---------------- 6) coordination shares case -----------------
        # Pattern: X COORD Y where COORD is و / ف / ثم / أو / أم. Y inherits
        # X's argmax case as a soft bias (Arabic conjoined noun phrases
        # typically share case).
        if "coordination_share_case" in enabled:
            COORD = {"و", "ف", "ثم", "او", "ام"}
            R_MATUF = ROLE_TO_ID["matuf"]
            for i in range(2, n):
                if norm_words[i - 1] in COORD:
                    head_case = int(case_logits[b, i - 2].argmax().item())
                    case_logits[b, i, head_case] += lambda_case
                    role_logits[b, i, R_MATUF] += 0.5 * lambda_role
                    trace.add(b, i, "coordination_share_case")

        # ---------------- 7) idafa chain (stronger) -----------------
        # Two consecutive nouns where the second has no determiner ال and
        # neither is preceded by a preposition: bias the second toward
        # mudaaf_ilayh + jarr more strongly than the stub.
        # (Distinguished from idafa_stub by a stricter condition: previous
        # word's role is already predicted as a head noun — mubtada / fail /
        # khabar / mafoul_bih / ism_majrur / mudaaf_ilayh.)
        if "idafa_chain" in enabled:
            HEAD_ROLES = {
                ROLE_TO_ID["mubtada"], ROLE_TO_ID["fail"], ROLE_TO_ID["khabar"],
                ROLE_TO_ID["mafoul_bih"], ROLE_TO_ID["ism_majrur"],
                ROLE_TO_ID["mudaaf_ilayh"], ROLE_TO_ID["ism_inna"],
                ROLE_TO_ID["khabar_inna"], ROLE_TO_ID["ism_kana"],
                ROLE_TO_ID["khabar_kana"],
            }
            for i in range(1, n):
                wi = norm_words[i]
                if not wi or _has_al_prefix(wi):
                    continue
                if norm_words[i - 1] in PREPS:
                    continue
                pi_prev = _argmax_pos(pos_logits[b, i - 1])
                pi_cur = _argmax_pos(pos_logits[b, i])
                if pi_prev != POS_TO_ID["noun"] or pi_cur != POS_TO_ID["noun"]:
                    continue
                role_prev = int(role_logits[b, i - 1].argmax().item())
                if role_prev not in HEAD_ROLES:
                    continue
                # full strength (no half-bias unlike the stub)
                case_logits[b, i, JARR] += lambda_case
                role_logits[b, i, R_MUDAAF] += lambda_role
                trace.add(b, i, "idafa_chain")

        # ---------------- 8) naat (adjective) propagation -----------------
        # If the model already predicts role=naat at word i, ensure case
        # agreement with word i-1 (the head noun). This is a redundant signal
        # to (5) but kicks in when the model has identified the role first.
        if "naat_propagation" in enabled:
            R_NAAT_ID = ROLE_TO_ID["naat"]
            for i in range(1, n):
                role_i = int(role_logits[b, i].argmax().item())
                if role_i != R_NAAT_ID:
                    continue
                head_case = int(case_logits[b, i - 1].argmax().item())
                case_logits[b, i, head_case] += lambda_case
                trace.add(b, i, "naat_propagation")

        # ---------------- 9) munadā (vocative) -> nasb -----------------
        # After يا (vocative particle), the immediately following noun is in
        # nasb when it's a definite munadā (e.g. يا أيها الناس -> أيها is
        # mabni على الضم but for indefinite the noun goes to nasb).
        # We pick nasb as the dominant pattern; a conservative bias applies.
        if "munadi_to_nasb" in enabled:
            VOC = {"يا", "ايا", "هيا", "اي"}
            R_MUNADA = ROLE_TO_ID["munada"]
            for i in range(1, n):
                if norm_words[i - 1] in VOC:
                    pos_i = _argmax_pos(pos_logits[b, i])
                    if pos_i in (POS_TO_ID["noun"], POS_TO_ID["adjective"]):
                        case_logits[b, i, NASB] += 0.5 * lambda_case
                        role_logits[b, i, R_MUNADA] += lambda_role
                        trace.add(b, i, "munadi_to_nasb")

    return case_logits, role_logits, trace


# ---------------------------------------------------------------------------
# Hierarchical inference: role -> case biasing
# ---------------------------------------------------------------------------
# Strong role-implies-case priors derived from the canonical schema. After
# the role head's argmax (or CRF-Viterbi) is fixed, these biases nudge case
# toward the syntactically-implied value. Soft (logit add), not hard.
ROLE_TO_CASE_PRIOR = {
    "fail":         "raf",
    "naib_fail":    "raf",
    "mubtada":      "raf",
    "khabar":       "raf",
    "ism_kana":     "raf",
    "khabar_inna":  "raf",
    "mafoul_bih":   "nasb",
    "khabar_kana":  "nasb",
    "ism_inna":     "nasb",
    "mafoul_other": "nasb",
    "hal":          "nasb",
    "tamyeez":      "nasb",
    "munada":       "nasb",
    "mudaaf_ilayh": "jarr",
    "ism_majrur":   "jarr",
    "harf_jarr":    "mabni",
    "harf_atf":     "mabni",
    "harf_other":   "mabni",
    "fil":          "mabni",
}


def apply_hierarchical(
    case_logits: torch.Tensor,        # (B, W, N_CASE)
    role_pred: torch.Tensor,          # (B, W) long — Viterbi or argmax of role
    word_mask: torch.Tensor,          # (B, W)
    *,
    lambda_hier: float = 1.0,
    trace: Optional[ConstraintTrace] = None,
) -> Tuple[torch.Tensor, ConstraintTrace]:
    """Bias case logits by the role-implied case prior.

    For each word with a confident role prediction, add ``+lambda_hier`` to
    the case index implied by ROLE_TO_CASE_PRIOR. Soft, ablation-friendly.
    """
    from ..structured.schema import ID_TO_ROLE, CASE_TO_ID

    case_logits = case_logits.clone()
    B, W, _ = case_logits.shape
    if trace is None:
        trace = ConstraintTrace()
        trace.init(B, W)

    for b in range(B):
        n = int(word_mask[b].sum().item())
        for i in range(n):
            r_id = int(role_pred[b, i].item())
            r_label = ID_TO_ROLE.get(r_id)
            if r_label is None:
                continue
            implied = ROLE_TO_CASE_PRIOR.get(r_label)
            if implied is None:
                continue
            c_idx = CASE_TO_ID.get(implied)
            if c_idx is None:
                continue
            case_logits[b, i, c_idx] += lambda_hier
            trace.add(b, i, f"hierarchical_role_to_case[{r_label}->{implied}]")
    return case_logits, trace
