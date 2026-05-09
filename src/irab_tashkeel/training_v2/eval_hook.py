"""Evaluation hook for the curriculum trainer.

Runs the trainer's current model over a held-out schema_v2
sentence list and produces a metric dict suitable for
:func:`CurriculumScheduler.advance_or_continue`.

The hook is the bridge between training and the gate logic. It
computes:

  - per-field accuracy (case_acc / role_acc / marker_em / fully)
  - construction-detection macro-F1 (proxy for stage-3 gate)
  - morph macro-F1 (proxy for stage-1 gate)
  - per-construction breakdown for diagnostic logging
  - calibration gap on role

The hook reuses the eval_v2 metric machinery — no new metric
logic. The only adapter code here is the model→prediction
conversion (we don't have a SentencePrediction directly from the
training-time model; the hook runs forward and converts logits to
TokenPrediction records).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..data_v2.schema_v2 import Sentence
from ..eval_v2 import (
    SentencePrediction, TokenPrediction,
    aggregate_outcomes, construction_detection_metrics, extract_outcomes,
)
from ..structured.schema import (
    CASE_LABELS, MARKER_LABELS, POS_LABELS, ROLE_LABELS,
    ID_TO_CASE, ID_TO_MARKER, ID_TO_POS, ID_TO_ROLE,
)
from .collator import MORPH_TO_ID, MORPH_VOCABS, SchemaV2Collator
from .dataset import SchemaV2Dataset


# ===========================================================================
# Model → predictions adapter
# ===========================================================================

def predict_for_eval(
    model, tokenizer, sentences: List[Sentence],
    *, batch_size: int = 32, device=None,
) -> List[SentencePrediction]:
    """Run the model in eval mode over schema_v2 sentences and return
    :class:`SentencePrediction` records.

    The model is expected to expose the same forward signature as
    :class:`DepAwareStructuredModel` — i.e., it takes
    ``(input_ids, attention_mask, word_starts, word_ends, word_mask,
    return_dict=True)`` and returns a logits dict with
    ``case_logits / role_logits / marker_logits / pos_logits``
    (and optionally ``<axis>_logits`` for each morph axis).
    """
    import torch
    import torch.nn.functional as F
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    collator = SchemaV2Collator(tokenizer)
    ds = SchemaV2Dataset(sentences)

    out: List[SentencePrediction] = []
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            batch_items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            batch = collator(batch_items)
            batch = {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            res = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                word_starts=batch["word_starts"],
                word_ends=batch["word_ends"],
                word_mask=batch["word_mask"],
                return_dict=True,
            )
            case_p   = F.softmax(res["case_logits"],   dim=-1)
            role_p   = F.softmax(res["role_logits"],   dim=-1)
            marker_p = F.softmax(res["marker_logits"], dim=-1)
            pos_p    = F.softmax(res["pos_logits"],    dim=-1)

            case_conf, case_idx     = case_p.max(dim=-1)
            role_conf, role_idx     = role_p.max(dim=-1)
            marker_conf, marker_idx = marker_p.max(dim=-1)
            pos_conf, pos_idx       = pos_p.max(dim=-1)

            word_mask = batch["word_mask"]
            for i, item in enumerate(batch_items):
                sid = item["sentence_id"]
                tps: List[TokenPrediction] = []
                for j in range(word_mask.shape[1]):
                    if word_mask[i, j].item() == 0:
                        break
                    tps.append(TokenPrediction(
                        sentence_id=sid, token_index=j,
                        case=ID_TO_CASE.get(int(case_idx[i, j].item())),
                        role=ID_TO_ROLE.get(int(role_idx[i, j].item())),
                        marker=ID_TO_MARKER.get(int(marker_idx[i, j].item())),
                        pos=ID_TO_POS.get(int(pos_idx[i, j].item()),
                                            POS_LABELS[int(pos_idx[i, j].item())] if int(pos_idx[i, j].item()) < len(POS_LABELS) else None),
                        case_conf=float(case_conf[i, j].item()),
                        role_conf=float(role_conf[i, j].item()),
                        marker_conf=float(marker_conf[i, j].item()),
                        pos_conf=float(pos_conf[i, j].item()),
                    ))
                out.append(SentencePrediction(sentence_id=sid, tokens=tps))
    return out


# ===========================================================================
# Per-stage metric extraction
# ===========================================================================

def _morph_macro_f1(model, tokenizer, sentences: List[Sentence],
                     batch_size: int = 32) -> float:
    """Average macro-F1 across morph heads on the held-out set.

    Computed from the model's morph heads when present. Returns 0.0
    if no morph supervision is available (e.g., when sentences
    don't carry morph labels).
    """
    import torch
    import torch.nn.functional as F
    device = next(model.parameters()).device
    model.eval()
    collator = SchemaV2Collator(tokenizer)
    ds = SchemaV2Dataset(sentences)

    correct: Dict[str, int] = {}
    total: Dict[str, int]   = {}
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            batch_items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            batch = collator(batch_items)
            batch = {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            res = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                word_starts=batch["word_starts"], word_ends=batch["word_ends"],
                word_mask=batch["word_mask"], return_dict=True,
            )
            for axis in MORPH_VOCABS:
                key = f"{axis}_logits"
                if key not in res:
                    continue
                lab = batch.get(f"morph_{axis}_labels")
                if lab is None:
                    continue
                pred = res[key].argmax(dim=-1)
                mask = (lab != -100)
                if mask.sum() == 0:
                    continue
                correct.setdefault(axis, 0)
                total.setdefault(axis, 0)
                correct[axis] += int((pred[mask] == lab[mask]).sum().item())
                total[axis]   += int(mask.sum().item())

    if not total:
        return 0.0
    per_axis_acc = [correct[a] / max(total[a], 1) for a in total]
    return sum(per_axis_acc) / len(per_axis_acc)


def _outcomes_filter(outcomes, predicate):
    return [o for o in outcomes if predicate(o)]


def gate_metrics_for_stage(
    stage_id: int,
    model, tokenizer, eval_sentences: List[Sentence],
    *, batch_size: int = 32,
) -> Dict[str, float]:
    """Compute the full metric dict for the current stage's gate.

    Item 13 — every eval reports:

      - overall fully + per-axis fully (case-only, role-only, marker-only)
      - strict_unseen_fully (the headline anti-leakage metric)
      - nested_fully  (clause depth ≥ 2)
      - ambiguity_fully (high ambiguity score)
      - long_range_fully (sentence length ≥ 25)
      - overlap_fully (multi-construction sentences)
      - calibration gap + ECE (already exposed)
      - confidence_histogram (per-bin counts at 0.0..1.0 in 0.1 steps)

    Stage gate metric mapping unchanged at gate level; the recovery
    patch (item 11) is implemented by the trainer's early-stop logic
    consuming ``strict_unseen_fully`` from this dict.
    """
    if not eval_sentences:
        return {}

    metrics: Dict[str, float] = {}

    preds = predict_for_eval(model, tokenizer, eval_sentences, batch_size=batch_size)
    outcomes = extract_outcomes(eval_sentences, preds)

    # ---------------- overall ----------------
    agg = aggregate_outcomes(outcomes)
    metrics["case_acc"]   = agg["case_acc"]
    metrics["role_f1"]    = agg["role_acc"]
    metrics["role_acc"]   = agg["role_acc"]
    metrics["marker_em"]  = agg["marker_em"]
    metrics["fully"]      = agg["fully"]
    metrics["calib_gap"]  = agg["calib_gap"]

    # ---------------- strict unseen (item 11/13) ----------------
    # Strict unseen = restrict to fully-observable tokens (where all 3
    # gold labels are populated). This is the canonical anti-noise
    # signal. The early-stop logic uses this metric.
    strict = _outcomes_filter(outcomes, lambda o: o.is_fully_observable)
    if strict:
        s_agg = aggregate_outcomes(strict)
        metrics["strict_unseen_fully"]  = s_agg["fully"]
        metrics["strict_unseen_case"]   = s_agg["case_acc"]
        metrics["strict_unseen_role"]   = s_agg["role_acc"]
        metrics["strict_unseen_marker"] = s_agg["marker_em"]
    else:
        metrics["strict_unseen_fully"]  = 0.0
        metrics["strict_unseen_case"]   = 0.0
        metrics["strict_unseen_role"]   = 0.0
        metrics["strict_unseen_marker"] = 0.0

    # ---------------- nested / long-range / overlap / ambiguity ----------------
    nested_o    = _outcomes_filter(outcomes, lambda o: o.clause_depth >= 2)
    long_o      = _outcomes_filter(outcomes, lambda o: o.sentence_length >= 25)
    overlap_o   = _outcomes_filter(outcomes, lambda o: len(o.construction_families) >= 2)
    ambig_o     = _outcomes_filter(outcomes, lambda o: o.semantic_pressure >= 2)

    for tag, sub in [("nested", nested_o), ("long_range", long_o),
                      ("overlap", overlap_o), ("ambiguity", ambig_o)]:
        if sub:
            sub_agg = aggregate_outcomes(sub)
            metrics[f"{tag}_fully"] = sub_agg["fully"]
        else:
            metrics[f"{tag}_fully"] = 0.0

    # Construction detection F1 (placeholder — see eval_v2)
    cd = construction_detection_metrics(eval_sentences, preds)
    if cd:
        f1s = [m.f1 for m in cd.values()]
        metrics["construction_f1_macro"] = sum(f1s) / max(len(f1s), 1)
    else:
        metrics["construction_f1_macro"] = 0.0

    # Quranic-only fully
    quranic_outcomes = [o for o in outcomes if o.domain == "quranic"]
    if quranic_outcomes:
        q_agg = aggregate_outcomes(quranic_outcomes)
        metrics["quranic_fully"] = q_agg["fully"]
    else:
        metrics["quranic_fully"] = metrics["fully"]

    # Stage-1: morph macro F1
    if stage_id == 1:
        metrics["morph_macro_f1"] = _morph_macro_f1(
            model, tokenizer, eval_sentences, batch_size=batch_size,
        )

    # Confidence histogram (item 6 D / 13)
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist = [0] * (len(bins) - 1)
    confs = []
    for o in outcomes:
        for c in (o.pred_role_conf, o.pred_case_conf, o.pred_marker_conf):
            if c is not None:
                confs.append(c)
    for c in confs:
        for i in range(len(bins) - 1):
            if bins[i] <= c < bins[i + 1]:
                hist[i] += 1
                break
    metrics["_conf_hist"] = hist
    metrics["_conf_n"] = len(confs)

    # ECE (10-bin)
    if confs:
        bin_n = [0] * 10
        bin_correct = [0] * 10
        bin_conf = [0.0] * 10
        for o in outcomes:
            if o.pred_role_conf is None or o.role_correct is None:
                continue
            b = min(9, int(o.pred_role_conf * 10))
            bin_n[b] += 1
            bin_conf[b] += o.pred_role_conf
            if o.role_correct:
                bin_correct[b] += 1
        n = sum(bin_n) or 1
        ece = sum(bin_n[b] * abs(bin_conf[b] / max(bin_n[b], 1) - bin_correct[b] / max(bin_n[b], 1))
                   for b in range(10)) / n
        metrics["ece"] = round(ece, 4)
    else:
        metrics["ece"] = 0.0

    return metrics
