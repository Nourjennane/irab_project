"""Phase 2 — fine-tune AraT5v2 on the *full per-word i'rāb* generation task.

Per-word seq2seq:
    Input:  أعرب: <word> | في: <sentence>
    Output: <full Arabic i'rāb prose>

Training data: `data/distill_v2/word_level.jsonl` produced by flattening
`data/distill_v2/distilled.jsonl` (Haiku 4.5 distilled, ~77K word rows).

Evaluation: at every epoch, log val loss; final eval is run separately
via `scripts/eval_arat5_irab.py` against Gazelle (so the val loss here
just guides checkpoint selection, not paper claims).

Run pattern:
    # smoke (small subset, 1 epoch)
    python -m irab_tashkeel.training.llm.irab_arat5_sft \\
        --config configs/irab_arat5v2_distill_v2.yaml \\
        --output runs/irab_arat5v2_smoke \\
        --limit_train 1000 --max_steps 100

    # full run
    python -m irab_tashkeel.training.llm.irab_arat5_sft \\
        --config configs/irab_arat5v2_distill_v2.yaml \\
        --output runs/irab_arat5v2_distill_v2
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

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
class IrabConfig:
    model_id: str = "UBC-NLP/AraT5v2-base-1024"
    pairs_path: str = "data/distill_v2/word_level.jsonl"
    val_split: float = 0.02

    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    n_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_input_length: int = 320
    max_target_length: int = 192     # i'rāb prose is longer than markers
    label_smoothing_factor: float = 0.1

    seed: int = 42
    optim: str = "adamw_torch"       # avoid bnb 8bit (4 prior HPC failures)

    # LoRA (used for AraT5v2-large to fit in 22-40GB MIG)
    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q", "k", "v", "o", "wi_0", "wi_1", "wo")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IrabConfig":
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _format_input(pair: dict, max_sent_len: int = 220) -> str:
    word = (pair.get("word") or "").strip()
    sent = (pair.get("sentence") or "").strip()
    if len(sent) > max_sent_len:
        sent = sent[:max_sent_len]
    return f"أعرب: {word} | في: {sent}"


def _load_pairs(path: Path | str) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("word") or not row.get("irab"):
                continue
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fine-tune AraT5v2 on full per-word i'rāb")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--limit_train", type=int, default=-1,
                        help="cap training rows (smoke runs)")
    parser.add_argument("--report_to", default="none")
    args = parser.parse_args()

    cfg = IrabConfig.from_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "irab_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    s = _import_stack()
    torch = s["torch"]
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"loading {cfg.model_id}", flush=True)
    tok = s["AutoTokenizer"].from_pretrained(cfg.model_id)
    model = s["AutoModelForSeq2SeqLM"].from_pretrained(cfg.model_id)

    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        print(f"  LoRA: r={cfg.lora_r} alpha={cfg.lora_alpha} targets={list(cfg.lora_target_modules)}", flush=True)
        lcfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    pairs = _load_pairs(cfg.pairs_path)
    print(f"raw pairs: {len(pairs)}", flush=True)
    random.Random(cfg.seed).shuffle(pairs)
    if args.limit_train > 0:
        pairs = pairs[: args.limit_train]
        print(f"smoke mode — limited to {len(pairs)} pairs", flush=True)

    n_val = max(1, int(len(pairs) * cfg.val_split))
    val_pairs = pairs[-n_val:]
    train_pairs = pairs[:-n_val]
    print(f"  train={len(train_pairs)}  val={len(val_pairs)}", flush=True)

    def encode(rows: List[dict]) -> List[Dict]:
        enc: List[Dict] = []
        for p in rows:
            inp = tok(_format_input(p),
                      truncation=True, max_length=cfg.max_input_length)
            tgt = tok(p["irab"],
                      truncation=True, max_length=cfg.max_target_length)
            inp["labels"] = tgt["input_ids"]
            enc.append(inp)
        return enc

    train_ds = s["Dataset"].from_list(encode(train_pairs))
    val_ds = s["Dataset"].from_list(encode(val_pairs))

    ta = s["Seq2SeqTrainingArguments"](
        output_dir=str(out),
        num_train_epochs=cfg.n_epochs,
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
        save_total_limit=2,
        eval_strategy="epoch",
        predict_with_generate=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim=cfg.optim,
        gradient_checkpointing=True,
        report_to=args.report_to,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        seed=cfg.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    collator = s["DataCollatorForSeq2Seq"](tok, model=model)
    import inspect
    _trainer_params = inspect.signature(s["Seq2SeqTrainer"].__init__).parameters
    _tok_kw = {"processing_class": tok} if "processing_class" in _trainer_params else {"tokenizer": tok}
    trainer = s["Seq2SeqTrainer"](
        model=model, args=ta,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
        **_tok_kw,
    )

    trainer.train()
    final = out / "final"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"\n✓ saved AraT5v2 i'rāb model to {final}", flush=True)


if __name__ == "__main__":
    main()
