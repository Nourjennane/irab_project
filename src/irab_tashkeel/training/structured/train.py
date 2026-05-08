"""Train the multi-head structured i'rāb classifier on the distilled corpus.

This is Phase 3 of the project — the v1 rebuild. The model is an AraT5v2-base
encoder + 4 linear classification heads (case / role / marker / POS); training
data is the canonicalized 5K-sentence / 77K-word distill_v2 corpus.

Usage (HPC):
    python -m irab_tashkeel.training.structured.train \\
        --config configs/structured_v1_rebuild.yaml \\
        --output_dir runs/structured_v1_rebuild

Smoke test (laptop):
    python -m irab_tashkeel.training.structured.train \\
        --config configs/structured_v1_rebuild.yaml \\
        --output_dir runs/structured_smoke \\
        --smoke
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from irab_tashkeel.structured.dataset import (
    IGNORE,
    StructuredCollator,
    StructuredIrabDataset,
)
from irab_tashkeel.structured.model import StructuredIrabModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class StructuredConfig:
    encoder_name: str = "UBC-NLP/AraT5v2-base-1024"
    train_path: str = "data/structured_v1/train.jsonl"
    val_path: str = "data/structured_v1/val.jsonl"
    max_subwords: int = 320
    max_words: int = 64
    head_dropout: float = 0.1
    loss_weights: tuple = (1.0, 1.0, 1.0, 0.5)

    learning_rate: float = 5.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    logging_steps: int = 50
    eval_steps: int = 500
    save_total_limit: int = 1
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StructuredConfig":
        d = yaml.safe_load(Path(path).read_text())
        if isinstance(d.get("loss_weights"), list):
            d["loss_weights"] = tuple(d["loss_weights"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Per-head accuracy metric (computed inside Trainer.evaluate)
# ---------------------------------------------------------------------------
class StructuredTrainer(Trainer):
    """Subclass to (a) avoid HF's labels-kwarg path, (b) log per-head losses."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = model(**inputs, return_dict=True)
        loss = out["loss"]
        if return_outputs:
            return loss, out
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Run forward with labels so we get loss. Return (loss, logits, labels)
        # in a shape Trainer.evaluate can aggregate even when we have multiple
        # heads. We sidestep the default by computing metrics in compute_metrics
        # over a custom dict.
        with torch.no_grad():
            out = model(**inputs, return_dict=True)
            loss = out.get("loss")
        if prediction_loss_only:
            return (loss, None, None)
        # Stack per-head argmax preds + labels for compute_metrics.
        preds = {
            "case": out["case_logits"].argmax(dim=-1),
            "role": out["role_logits"].argmax(dim=-1),
            "marker": out["marker_logits"].argmax(dim=-1),
            "pos": out["pos_logits"].argmax(dim=-1),
            "word_mask": inputs["word_mask"],
        }
        labels = {
            "case": inputs["case_labels"],
            "role": inputs["role_labels"],
            "marker": inputs["marker_labels"],
            "pos": inputs["pos_labels"],
        }
        # Trainer expects tensors here; we stack to a single 5-channel tensor.
        # Channels: [case_pred, role_pred, marker_pred, pos_pred, word_mask]
        pred_t = torch.stack(
            [preds["case"], preds["role"], preds["marker"], preds["pos"], preds["word_mask"]], dim=-1
        )
        lab_t = torch.stack(
            [labels["case"], labels["role"], labels["marker"], labels["pos"]], dim=-1
        )
        return (loss, pred_t, lab_t)


def compute_metrics(eval_pred) -> dict:
    """Per-head accuracy + 4-correct rate over non-padded words."""
    preds_t, labels_t = eval_pred  # numpy arrays
    # preds_t: (N, W, 5), labels_t: (N, W, 4)
    case_pred = preds_t[..., 0]
    role_pred = preds_t[..., 1]
    marker_pred = preds_t[..., 2]
    pos_pred = preds_t[..., 3]
    word_mask = preds_t[..., 4].astype(bool)
    case_lab = labels_t[..., 0]
    role_lab = labels_t[..., 1]
    marker_lab = labels_t[..., 2]
    pos_lab = labels_t[..., 3]

    metrics = {}
    case_ok = (case_pred == case_lab) & word_mask
    role_ok = (role_pred == role_lab) & word_mask
    marker_ok = (marker_pred == marker_lab) & word_mask
    pos_ok = (pos_pred == pos_lab) & word_mask
    n = word_mask.sum().item()
    if n > 0:
        metrics["case_acc"] = float(case_ok.sum() / n)
        metrics["role_acc"] = float(role_ok.sum() / n)
        metrics["marker_acc"] = float(marker_ok.sum() / n)
        metrics["pos_acc"] = float(pos_ok.sum() / n)
        full = case_ok & role_ok & marker_ok
        metrics["fully"] = float(full.sum() / n)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--smoke", action="store_true", help="Run a 200-row smoke test")
    ap.add_argument("--smoke_n", type=int, default=200)
    ap.add_argument("--smoke_epochs", type=int, default=1)
    args = ap.parse_args()

    cfg = StructuredConfig.from_yaml(args.config)
    set_seed(cfg.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str)
    )

    print(f"[train] loading tokenizer + encoder from {cfg.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
    model = StructuredIrabModel(
        encoder_name=cfg.encoder_name,
        head_dropout=cfg.head_dropout,
        loss_weights=tuple(cfg.loss_weights),
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print(f"[train] datasets: train={cfg.train_path}  val={cfg.val_path}")
    train_ds = StructuredIrabDataset(cfg.train_path, tokenizer, max_subwords=cfg.max_subwords, max_words=cfg.max_words)
    val_ds = StructuredIrabDataset(cfg.val_path, tokenizer, max_subwords=cfg.max_subwords, max_words=cfg.max_words)

    if args.smoke:
        # Subsample to smoke_n sentences for a fast sanity run.
        train_ds._records = train_ds._records[: args.smoke_n]
        val_ds._records = val_ds._records[: max(20, args.smoke_n // 10)]
        cfg.num_train_epochs = args.smoke_epochs
        cfg.eval_steps = max(20, args.smoke_n // (cfg.per_device_train_batch_size * 2))
        cfg.logging_steps = 5

    print(f"[train] n_train_sents={len(train_ds)}  n_val_sents={len(val_ds)}")

    collator = StructuredCollator(pad_token_id=tokenizer.pad_token_id or 0)

    # eval_strategy was renamed in newer transformers; support both keys.
    train_args_dict = dict(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        bf16=cfg.bf16,
        optim=cfg.optim,
        report_to="none",
        save_strategy="epoch",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        label_names=["case_labels", "role_labels", "marker_labels", "pos_labels"],
    )
    # transformers >=4.30 renamed evaluation_strategy -> eval_strategy
    try:
        train_args = TrainingArguments(eval_strategy="steps", eval_steps=cfg.eval_steps, **train_args_dict)
    except TypeError:
        train_args = TrainingArguments(evaluation_strategy="steps", eval_steps=cfg.eval_steps, **train_args_dict)

    trainer = StructuredTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("[train] starting fit ...")
    train_result = trainer.train()
    print(f"[train] training metrics: {train_result.metrics}")

    print("[train] saving final model + tokenizer ...")
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    # Save model state + config in a way structured_predictor can re-load.
    torch.save(model.state_dict(), final_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(final_dir)
    (final_dir / "structured_config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    print("[train] done.")


if __name__ == "__main__":
    main()
