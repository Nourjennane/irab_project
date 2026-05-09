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


def gate_metrics_for_stage(
    stage_id: int,
    model, tokenizer, eval_sentences: List[Sentence],
    *, batch_size: int = 32,
) -> Dict[str, float]:
    """Compute the metric dict for the current stage's gate.

    Stage gate metric mapping (from
    :data:`curriculum.config.DEFAULT_STAGES`):

      stage 1 → morph_macro_f1
      stage 2 → role_f1     (interpreted as role_acc)
      stage 3 → construction_f1_macro
      stages 4-6 → fully
      stage 7 → quranic_fully

    Returns a dict containing all of these so the scheduler can
    look up whichever it needs.
    """
    if not eval_sentences:
        return {}

    metrics: Dict[str, float] = {}

    # Always compute the per-field aggregates (cheap)
    preds = predict_for_eval(model, tokenizer, eval_sentences, batch_size=batch_size)
    outcomes = extract_outcomes(eval_sentences, preds)
    agg = aggregate_outcomes(outcomes)
    metrics["case_acc"]   = agg["case_acc"]
    metrics["role_f1"]    = agg["role_acc"]    # acc as F1 proxy
    metrics["role_acc"]   = agg["role_acc"]
    metrics["marker_em"]  = agg["marker_em"]
    metrics["fully"]      = agg["fully"]
    metrics["calib_gap"]  = agg["calib_gap"]

    # Construction detection F1 — predictions don't yet emit
    # ConstructionPrediction, so this returns zeros. Reserved for
    # the future when the trainer's eval emits constructions.
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

    # Stage-1: morph macro F1 (computed separately because morph
    # heads aren't part of the SentencePrediction adapter)
    if stage_id == 1:
        metrics["morph_macro_f1"] = _morph_macro_f1(
            model, tokenizer, eval_sentences, batch_size=batch_size,
        )

    return metrics
