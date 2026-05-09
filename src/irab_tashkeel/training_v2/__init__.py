"""training_v2 — curriculum-driven multi-task trainer (Step 7+).

Public API:

    config.TrainerConfig
    config.HeadLossWeights

    dataset.SchemaV2Dataset
    collator.SchemaV2Collator
    collator.MORPH_VOCABS, MORPH_TO_ID, IGNORE

    loss.compute_multi_head_loss(logits, labels, weights)

    trainer.StageTrainer
    trainer.TrainerState

    eval_hook.gate_metrics_for_stage(stage_id, model, tokenizer, sentences)
    eval_hook.predict_for_eval(model, tokenizer, sentences)

The trainer is the bridge between the data engine (data_v2) +
curriculum scheduler (curriculum) + frozen-baseline model class
(morphology.dep_aware_model) + eval_v2 metrics. It is backbone-
agnostic: any model with forward(input_ids, attention_mask,
word_starts, word_ends, word_mask, return_dict=True) → multi-head
logits dict can be plugged in.
"""
from .config import HeadLossWeights, TrainerConfig
from .dataset import SchemaV2Dataset
from .collator import (
    SchemaV2Collator, CollatorConfig, IGNORE,
    MORPH_VOCABS, MORPH_TO_ID,
)
from .loss import compute_multi_head_loss, MORPH_AXES
from .trainer import StageTrainer, TrainerState
from .eval_hook import gate_metrics_for_stage, predict_for_eval

__all__ = [
    "HeadLossWeights", "TrainerConfig",
    "SchemaV2Dataset",
    "SchemaV2Collator", "CollatorConfig", "IGNORE",
    "MORPH_VOCABS", "MORPH_TO_ID", "MORPH_AXES",
    "compute_multi_head_loss",
    "StageTrainer", "TrainerState",
    "gate_metrics_for_stage", "predict_for_eval",
]
