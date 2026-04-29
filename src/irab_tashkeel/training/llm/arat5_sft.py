"""Full fine-tune AraT5v2-base-1024 on the i'rāb generation task (Stack B).

This is the lower-risk fallback in the research plan. AraT5v2-base is 296M
params, fits batch 16 on a single A100/H100, and per UBC-NLP converges 10×
faster than the original AraT5-base.

Run pattern (Bocconi HPC):
    sbatch scripts/slurm/train_arat5_sft.sbatch
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


def _import_stack():
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer,
        DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )
    return {
        "torch": torch, "Dataset": Dataset,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Seq2SeqTrainer": Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,
    }


@dataclass
class T5Config:
    model_id: str = "UBC-NLP/AraT5v2-base-1024"
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    n_epochs: int = 5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_input_length: int = 256
    max_target_length: int = 768
    label_smoothing_factor: float = 0.1
    curriculum: List[Optional[int]] = field(default_factory=lambda: [8, 16, None, None, None])
    dataset_cache: str = "data/cache/combined.pkl"
    val_split: float = 0.02
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "T5Config":
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def main():
    parser = argparse.ArgumentParser(description="AraT5v2 full FT for i'rāb generation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--report_to", default="none")
    args = parser.parse_args()

    cfg = T5Config.from_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "t5_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    s = _import_stack()
    torch = s["torch"]
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"Loading {cfg.model_id}")
    tok = s["AutoTokenizer"].from_pretrained(cfg.model_id)
    model = s["AutoModelForSeq2SeqLM"].from_pretrained(cfg.model_id)

    from .format import examples_to_pairs
    from ...data.build_dataset import load_examples

    cache = Path(cfg.dataset_cache)
    if not cache.exists():
        raise FileNotFoundError(f"Dataset cache not found: {cache}")
    examples = load_examples(cache)
    pairs = examples_to_pairs(examples)
    random.Random(cfg.seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * cfg.val_split))
    val_pairs = pairs[-n_val:]
    train_pairs = pairs[:-n_val]
    print(f"pairs total: {len(pairs)}  train: {len(train_pairs)}  val: {len(val_pairs)}")

    def tokenize(pair_list):
        rows = []
        for p in pair_list:
            inp = tok(
                f"{p.system}\n\n{p.user}",
                truncation=True, max_length=cfg.max_input_length,
            )
            tgt = tok(
                p.assistant,
                truncation=True, max_length=cfg.max_target_length,
            )
            inp["labels"] = tgt["input_ids"]
            rows.append({**inp, "n_words": p.n_words})
        return rows

    val_rows = tokenize(val_pairs)
    val_ds = s["Dataset"].from_list(val_rows).remove_columns(["n_words"])

    epoch_caps = cfg.curriculum or [None]
    if len(epoch_caps) != cfg.n_epochs:
        if len(epoch_caps) < cfg.n_epochs:
            epoch_caps = list(epoch_caps) + [epoch_caps[-1]] * (cfg.n_epochs - len(epoch_caps))
        else:
            epoch_caps = epoch_caps[: cfg.n_epochs]

    last_ckpt: Optional[str] = None
    for epoch_idx, cap in enumerate(epoch_caps):
        stage_pairs = [p for p in train_pairs if cap is None or p.n_words <= cap]
        print(f"\n=== Epoch {epoch_idx+1}/{cfg.n_epochs}  cap={cap}  pairs={len(stage_pairs)} ===")
        if not stage_pairs:
            continue
        train_rows = tokenize(stage_pairs)
        train_ds = s["Dataset"].from_list(train_rows).remove_columns(["n_words"])

        ta = s["Seq2SeqTrainingArguments"](
            output_dir=str(out / f"stage_{epoch_idx+1}"),
            num_train_epochs=1,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type="cosine",
            label_smoothing_factor=cfg.label_smoothing_factor,
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="epoch",
            predict_with_generate=False,
            bf16=True,
            optim="adamw_torch",
            report_to=args.report_to,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            seed=cfg.seed + epoch_idx,
        )
        collator = s["DataCollatorForSeq2Seq"](tok, model=model)
        trainer = s["Seq2SeqTrainer"](
            model=model, args=ta,
            train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=collator,
            processing_class=tok,
        )
        trainer.train(resume_from_checkpoint=last_ckpt)
        last_ckpt = str(out / f"stage_{epoch_idx+1}")

    # Final save
    final = out / "final"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"\n✓ saved fine-tuned T5 to {final}")


if __name__ == "__main__":
    main()
