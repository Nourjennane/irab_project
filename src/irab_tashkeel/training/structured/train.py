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

    label_smoothing: float = 0.0
    role_class_weighting: str = "none"          # "none" | "sqrt_inv_freq" | "inv_freq"
    pooling_strategy: str = "mean"               # "mean" | "first"

    # Phase 4 — CRF on the role head
    use_crf_role: bool = False
    crf_init_from_bigrams: bool = True           # init transitions from training corpus

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

    # Best-checkpoint retention
    load_best_at_end: bool = True
    metric_for_best: str = "fully"               # one of compute_metrics keys
    greater_is_better: bool = True

    # ---- Phase 1 — morphology heads (opt-in; default OFF preserves rev 2 path) ----
    enable_morph_heads: bool = False
    morph_train_path: str = "data/morph_v1/train.jsonl"
    morph_val_path: str = "data/morph_v1/val.jsonl"
    # per-feature loss weights; default uniform 0.3 for every feature.
    morph_loss_weights: Optional[dict] = None
    # subset of morph features to enable (None = all 7).
    morph_heads_enabled: Optional[list] = None

    # ---- Phase 4a — taxonomy version (v3 = rev 2 / Phase 1 default; v4 = 34 labels) ----
    taxonomy: str = "v3"

    # ---- Phase 2 — soft morphology conditioning (opt-in; requires enable_morph_heads=True) ----
    # Mechanism ∈ {None, "none", "film", "additive", "concat_embed"}.
    # None / "none" = Phase 1 path (no conditioning, byte-identical).
    conditioning_mechanism: Optional[str] = None
    # When True, ``m`` is detached before entering the conditioning module so
    # gradients do NOT flow back through the morph heads. Reported as an ablation.
    conditioning_detached: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StructuredConfig":
        d = yaml.safe_load(Path(path).read_text())
        if isinstance(d.get("loss_weights"), list):
            d["loss_weights"] = tuple(d["loss_weights"])
        return cls(**d)


def _compute_role_class_weights(jsonl_path: str, mode: str,
                                taxonomy: str = "v3") -> Optional[torch.Tensor]:
    """Compute per-class weights for the role head from the training corpus.

    Args:
        jsonl_path: source corpus (v3 from data/structured_v1/, v4 from
            data/structured_v1_v4/).
        mode: "none" / "sqrt_inv_freq" / "inv_freq".
        taxonomy: "v3" (25 labels, default — rev 2 / Phase 1) or "v4"
            (34 labels — Phase 4a).
    """
    if mode == "none":
        return None
    import json
    from collections import Counter
    if taxonomy == "v4":
        from irab_tashkeel.structured.taxonomy_v4 import (
            ROLE_LABELS_V4 as ROLE_LABELS,
            ROLE_TO_ID_V4 as ROLE_TO_ID,
            N_ROLE_V4 as N_ROLE,
        )
    else:
        from irab_tashkeel.structured.schema import N_ROLE, ROLE_LABELS, ROLE_TO_ID
    cnt = Counter()
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for it in rec.get("items", []):
                r = it.get("role")
                if r in ROLE_TO_ID:
                    cnt[r] += 1
    weights = torch.zeros(N_ROLE, dtype=torch.float32)
    for label, idx in ROLE_TO_ID.items():
        c = cnt.get(label, 0)
        if mode == "sqrt_inv_freq":
            weights[idx] = (1.0 / (c + 1)) ** 0.5
        elif mode == "inv_freq":
            weights[idx] = 1.0 / (c + 1)
        else:
            raise ValueError(f"Unknown role_class_weighting: {mode}")
    weights = weights / weights.mean().clamp(min=1e-8)
    return weights


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
        # Stack per-head preds + labels for compute_metrics.
        # Role uses Viterbi when CRF is on, argmax otherwise.
        if getattr(model, "use_crf_role", False) and getattr(model, "role_crf", None) is not None:
            paths = model.role_crf.decode(out["role_logits"], inputs["word_mask"])
            B, W, _ = out["role_logits"].shape
            role_pred = torch.zeros((B, W), dtype=torch.long, device=out["role_logits"].device)
            for b, p in enumerate(paths):
                for j, t in enumerate(p):
                    role_pred[b, j] = t
        else:
            role_pred = out["role_logits"].argmax(dim=-1)
        preds = {
            "case": out["case_logits"].argmax(dim=-1),
            "role": role_pred,
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
    # Phase 4a: dispatch on taxonomy choice (v3 = rev 2 default; v4 = 34
    # labels). The role taxonomy + role-class-weight source travel together.
    if cfg.taxonomy == "v4":
        from irab_tashkeel.structured.taxonomy_v4 import (
            ROLE_TO_ID_V4 as ROLE_TO_ID_active,
            N_ROLE_V4 as N_ROLE_active,
        )
        print(f"[train] taxonomy=v4 (34 labels)")
    else:
        from irab_tashkeel.structured.schema import (
            ROLE_TO_ID as ROLE_TO_ID_active,
            N_ROLE as N_ROLE_active,
        )
        print(f"[train] taxonomy=v3 (25 labels) [rev 2 / Phase 1 default]")
    role_weight_source = cfg.train_path
    role_weights = _compute_role_class_weights(role_weight_source,
                                                cfg.role_class_weighting,
                                                taxonomy=cfg.taxonomy)
    if role_weights is not None:
        print(f"[train] role_class_weighting={cfg.role_class_weighting}  "
              f"min={role_weights.min().item():.3f}  max={role_weights.max().item():.3f}")
    print(f"[train] label_smoothing={cfg.label_smoothing}  pooling={cfg.pooling_strategy}  "
          f"use_crf_role={cfg.use_crf_role}  enable_morph_heads={cfg.enable_morph_heads}")

    # n_role passes through to the model so v4 (34) overrides the default v3 (25).
    n_role_kw = {"n_role": N_ROLE_active} if cfg.taxonomy == "v4" else {}
    if cfg.enable_morph_heads:
        # Phase 1 path — morph-augmented model + masked multi-task dataset.
        from irab_tashkeel.morphology.morph_model import MorphAugmentedStructuredModel
        morph_heads_enabled = set(cfg.morph_heads_enabled) if cfg.morph_heads_enabled else None
        model = MorphAugmentedStructuredModel(
            encoder_name=cfg.encoder_name,
            head_dropout=cfg.head_dropout,
            loss_weights=tuple(cfg.loss_weights),
            label_smoothing=cfg.label_smoothing,
            role_class_weights=role_weights,
            pooling_strategy=cfg.pooling_strategy,
            use_crf_role=cfg.use_crf_role,
            enable_morph_heads=True,
            morph_heads_enabled=morph_heads_enabled,
            morph_loss_weights=cfg.morph_loss_weights,
            conditioning_mechanism=cfg.conditioning_mechanism,
            conditioning_detached=cfg.conditioning_detached,
            **n_role_kw,
        )
        print(f"[train] morph heads enabled: {sorted(model.morph_heads_enabled)}")
        print(f"[train] morph loss weights: { {k: model.morph_loss_weights[k] for k in sorted(model.morph_heads_enabled)} }")
        if model.conditioning is not None:
            print(f"[train] conditioning: mechanism={model.conditioning_mechanism} "
                  f"detached={model.conditioning_detached} "
                  f"params={sum(p.numel() for p in model.conditioning.parameters())}")
    else:
        # Default rev-2 path — byte-identical to before Phase 1 / Phase 4a
        # when taxonomy=v3. With taxonomy=v4 it's the granularity-only branch.
        model = StructuredIrabModel(
            encoder_name=cfg.encoder_name,
            head_dropout=cfg.head_dropout,
            loss_weights=tuple(cfg.loss_weights),
            label_smoothing=cfg.label_smoothing,
            role_class_weights=role_weights,
            pooling_strategy=cfg.pooling_strategy,
            use_crf_role=cfg.use_crf_role,
            **n_role_kw,
        )

    if cfg.use_crf_role and cfg.crf_init_from_bigrams:
        from irab_tashkeel.structured.crf import compute_role_bigrams
        from irab_tashkeel.structured.schema import ROLE_TO_ID
        print("[train] computing role bigrams for CRF init ...")
        trans, start, end = compute_role_bigrams(cfg.train_path, ROLE_TO_ID)
        model.role_crf.init_from_bigrams(trans, start, end)
        print(f"[train] CRF init: trans range [{trans.min().item():.2f}, {trans.max().item():.2f}]; "
              f"start range [{start.min().item():.2f}, {start.max().item():.2f}]")
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    role_to_id_kw = {"role_to_id": ROLE_TO_ID_active} if cfg.taxonomy == "v4" else {}
    if cfg.enable_morph_heads:
        from irab_tashkeel.morphology.dataset import (
            MorphAwareStructuredIrabDataset, MorphAwareCollator,
        )
        print(f"[train] datasets (Phase 1, merged): train={cfg.morph_train_path}  val={cfg.morph_val_path}")
        train_ds = MorphAwareStructuredIrabDataset(cfg.morph_train_path, tokenizer,
                                                    max_subwords=cfg.max_subwords, max_words=cfg.max_words,
                                                    **role_to_id_kw)
        val_ds = MorphAwareStructuredIrabDataset(cfg.morph_val_path, tokenizer,
                                                  max_subwords=cfg.max_subwords, max_words=cfg.max_words,
                                                  **role_to_id_kw)
    else:
        print(f"[train] datasets: train={cfg.train_path}  val={cfg.val_path}")
        train_ds = StructuredIrabDataset(cfg.train_path, tokenizer,
                                          max_subwords=cfg.max_subwords, max_words=cfg.max_words,
                                          **role_to_id_kw)
        val_ds = StructuredIrabDataset(cfg.val_path, tokenizer,
                                        max_subwords=cfg.max_subwords, max_words=cfg.max_words,
                                        **role_to_id_kw)

    if args.smoke:
        # Subsample to smoke_n sentences for a fast sanity run.
        train_ds._records = train_ds._records[: args.smoke_n]
        val_ds._records = val_ds._records[: max(20, args.smoke_n // 10)]
        cfg.num_train_epochs = args.smoke_epochs
        cfg.eval_steps = max(20, args.smoke_n // (cfg.per_device_train_batch_size * 2))
        cfg.logging_steps = 5

    print(f"[train] n_train_sents={len(train_ds)}  n_val_sents={len(val_ds)}")

    if cfg.enable_morph_heads:
        from irab_tashkeel.morphology.dataset import MorphAwareCollator
        collator = MorphAwareCollator(pad_token_id=tokenizer.pad_token_id or 0)
    else:
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
        # If load_best_at_end, save_strategy and eval_strategy must match.
        # We always evaluate at epoch boundaries when load_best is on.
        load_best_model_at_end=cfg.load_best_at_end,
        metric_for_best_model=cfg.metric_for_best if cfg.load_best_at_end else None,
        greater_is_better=cfg.greater_is_better if cfg.load_best_at_end else None,
        remove_unused_columns=False,
        label_names=(
            ["case_labels", "role_labels", "marker_labels", "pos_labels"]
            + (["gender_labels", "number_labels", "definite_labels", "person_labels",
                "aspect_labels", "mood_labels", "voice_labels"]
               if cfg.enable_morph_heads else [])
        ),
        # T5EncoderModel keeps a shared input embedding (encoder.shared aliases
        # encoder.embed_tokens.weight); safetensors refuses to save tied tensors.
        # Use the regular torch pickle save which handles shared tensors fine.
        save_safetensors=False,
    )
    # transformers >=4.30 renamed evaluation_strategy -> eval_strategy
    eval_strat = "epoch" if cfg.load_best_at_end else "steps"
    extra = {} if eval_strat == "epoch" else {"eval_steps": cfg.eval_steps}
    try:
        train_args = TrainingArguments(eval_strategy=eval_strat, **extra, **train_args_dict)
    except TypeError:
        train_args = TrainingArguments(evaluation_strategy=eval_strat, **extra, **train_args_dict)

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
