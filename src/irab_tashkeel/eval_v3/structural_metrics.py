"""Attachment + governor + overlap accuracy.

These metrics evaluate the *structural* correctness of predictions
beyond plain label accuracy:

  - attachment_accuracy     — for each token, did the model link it to
                               the correct head construction (idafa /
                               kana-sister-frame / etc.)?
  - governor_accuracy       — when the gold provides a governor token
                               (via dep_head_idx), did the model's
                               predicted role+case combination imply
                               the same governor relation?
  - overlap_accuracy        — on tokens belonging to ≥ 2 constructions,
                               did the model assign the role consistent
                               with the *correct* family (the one whose
                               case+marker prevailed in gold)?

The "implied governor" mapping comes from the canonical role→source
table:

  mudaaf_ilayh   → governed by the iḍāfa head
  ism_kana       → governed by kāna
  khabar_kana    → governed by kāna
  ism_inna       → governed by inna
  khabar_inna    → governed by inna
  ism_majrur     → governed by the preceding harf_jarr
  fail / naib_fail / mafoul_*   → governed by the verb

"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..eval_v2 import SentencePrediction, TokenOutcome, extract_outcomes


# Role → "source family" mapping for governor inference
ROLE_TO_GOVERNOR_FAMILY = {
    "mudaaf_ilayh":  "idafa_head",
    "ism_kana":      "kana_first",
    "khabar_kana":   "kana_first",
    "ism_inna":      "inna_first",
    "khabar_inna":   "inna_first",
    "ism_majrur":    "preceding_harf_jarr",
    "fail":          "verb",
    "naib_fail":     "passive_verb",
    "mafoul_bih":    "verb",
    "mafoul_other":  "verb",
    "mafoul_mutlaq": "verb",
}


def _gold_governor_family(gold_role: Optional[str]) -> Optional[str]:
    if gold_role is None:
        return None
    return ROLE_TO_GOVERNOR_FAMILY.get(gold_role)


def attachment_accuracy(
    sentences: List, predictions: List[SentencePrediction],
) -> Dict[str, float]:
    """Did the model assign the same governor-family as gold?"""
    outcomes = extract_outcomes(sentences, predictions)
    n_total = 0
    n_correct = 0
    per_family: Dict[str, Dict[str, int]] = {}
    for o in outcomes:
        if o.gold_role is None or o.pred_role is None:
            continue
        gf = _gold_governor_family(o.gold_role)
        pf = _gold_governor_family(o.pred_role)
        if gf is None and pf is None:
            continue
        n_total += 1
        if gf == pf:
            n_correct += 1
        per_family.setdefault(gf or "none", {"n": 0, "correct": 0})
        per_family[gf or "none"]["n"] += 1
        if gf == pf:
            per_family[gf or "none"]["correct"] += 1
    return {
        "attachment_accuracy": round(n_correct / max(n_total, 1), 4),
        "n":                   n_total,
        "per_family":          {k: {**v,
                                     "acc": round(v["correct"] / max(v["n"], 1), 4)}
                                 for k, v in per_family.items()},
    }


def governor_accuracy(
    sentences: List, predictions: List[SentencePrediction],
) -> Dict[str, float]:
    """When gold has a dep_head_idx, did the predicted role's
    governor-family match the head's POS family? Approximate; the
    ground truth governor token is not always reachable from role
    alone, but governor *family* is."""
    by_sid = {s.sentence_id: s for s in sentences}
    outcomes = extract_outcomes(sentences, predictions)
    n = 0
    n_match = 0
    for o in outcomes:
        s = by_sid.get(o.sentence_id)
        if s is None or o.token_index >= len(s.tokens):
            continue
        t = s.tokens[o.token_index]
        head = t.dep_head_idx
        if head is None or head < 0 or head >= len(s.tokens):
            continue
        head_pos = s.tokens[head].pos.value or ""
        # Map predicted role → expected head POS
        gf = _gold_governor_family(o.pred_role)
        if gf is None:
            continue
        expected_pos_set = {
            "verb":                 {"VERB", "AUX"},
            "passive_verb":         {"VERB", "AUX"},
            "preceding_harf_jarr":  {"ADP"},
            "idafa_head":           {"NOUN", "PROPN", "PRON"},
            "kana_first":           {"AUX", "VERB"},
            "inna_first":           {"PART", "SCONJ"},
        }.get(gf, set())
        if not expected_pos_set:
            continue
        n += 1
        if head_pos in expected_pos_set:
            n_match += 1
    return {
        "governor_accuracy": round(n_match / max(n, 1), 4),
        "n":                 n,
    }


def overlap_accuracy(
    sentences: List, predictions: List[SentencePrediction],
) -> Dict[str, float]:
    """For tokens in ≥ 2 constructions, did the predicted role match
    the gold role at all?"""
    by_sid = {s.sentence_id: s for s in sentences}
    outcomes = extract_outcomes(sentences, predictions)
    n = 0
    n_correct = 0
    for o in outcomes:
        s = by_sid.get(o.sentence_id)
        if s is None:
            continue
        n_in = sum(1 for c in s.constructions
                    if o.token_index in c.token_indices)
        if n_in < 2:
            continue
        if o.gold_role is None:
            continue
        n += 1
        if o.role_correct is True:
            n_correct += 1
    return {
        "overlap_accuracy": round(n_correct / max(n, 1), 4),
        "n":                n,
    }
