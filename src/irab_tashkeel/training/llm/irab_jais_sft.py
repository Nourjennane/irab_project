"""Phase 2.4 — Jais (decoder-only Arabic+English LM) causal SFT for i'rāb.

Same training data (`data/distill_v2/word_level.jsonl`, ~77K rows) as the
seq2seq runs (AraT5v2-base, mT5-base), but formatted as a single
prompt+completion sequence and trained with masked-prompt loss (only the
i'rāb tokens contribute to the cross-entropy).

Prompt template:
    أعرب الكلمة التالية في سياقها.
    الكلمة: <word>
    الجملة: <sentence>
    الإعراب: <irab><eos>

The label tensor is `prompt_tokens × -100  +  irab_tokens × actual_id` so
that loss is only computed on the answer span. Prompts are truncated to
`max_input_length`; total sequence is capped at `max_total_length`.

Run pattern:
    # smoke
    python -m irab_tashkeel.training.llm.irab_jais_sft \\
        --config configs/irab_jais_1p3b.yaml \\
        --output runs/irab_jais_smoke \\
        --limit_train 1000 --max_steps 100

    # full
    sbatch scripts/slurm/45_smoke_irab_jais.sbatch
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
        AutoModelForCausalLM, AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer, TrainingArguments,
    )
    return {
        "torch": torch, "Dataset": Dataset,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


@dataclass
class JaisConfig:
    model_id: str = "inceptionai/jais-family-1p3b"
    pairs_path: str = "data/distill_v2/word_level.jsonl"
    val_split: float = 0.02

    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    n_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_input_length: int = 320       # prompt-only cap
    max_total_length: int = 512       # prompt + irab cap
    label_smoothing_factor: float = 0.0  # rare in causal LM
    seed: int = 42
    optim: str = "adamw_torch"

    # LoRA defaults — necessary for 1.3B+ to fit + train fast on a 40GB MIG
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # Jais-family is a GPT-2-style architecture; common LoRA targets:
    lora_target_modules: tuple = ("c_attn", "c_proj", "c_fc")
    # Whether to load in 4-bit (saves a lot of memory; needs bitsandbytes)
    load_in_4bit: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "JaisConfig":
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


PROMPT_TEMPLATE = (
    "أعرب الكلمة التالية في سياقها.\n"
    "الكلمة: {word}\n"
    "الجملة: {sentence}\n"
    "الإعراب: "
)


def _format_prompt(pair: dict, max_sent_len: int = 220) -> str:
    word = (pair.get("word") or "").strip()
    sent = (pair.get("sentence") or "").strip()
    if len(sent) > max_sent_len:
        sent = sent[:max_sent_len]
    return PROMPT_TEMPLATE.format(word=word, sentence=sent)


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
    parser = argparse.ArgumentParser(description="Causal-LM SFT for i'rāb (Jais)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--limit_train", type=int, default=-1)
    parser.add_argument("--report_to", default="none")
    args = parser.parse_args()

    cfg = JaisConfig.from_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "jais_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    s = _import_stack()
    torch = s["torch"]
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"loading {cfg.model_id}", flush=True)
    tok = s["AutoTokenizer"].from_pretrained(cfg.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {"trust_remote_code": True}
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = s["AutoModelForCausalLM"].from_pretrained(cfg.model_id, **model_kwargs)

    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        if cfg.load_in_4bit:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
        print(f"  LoRA: r={cfg.lora_r} alpha={cfg.lora_alpha} targets={list(cfg.lora_target_modules)}", flush=True)
        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
        )
        model = get_peft_model(model, lcfg)
        # PEFT + grad ckpt requires this to keep gradients flowing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
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

    eos_id = tok.eos_token_id

    def encode(rows: List[dict]) -> List[Dict]:
        enc: List[Dict] = []
        for p in rows:
            prompt = _format_prompt(p)
            target = (p.get("irab") or "").strip()
            # Tokenise prompt (no special tokens; we add eos manually below)
            prompt_ids = tok(prompt, add_special_tokens=False, truncation=True,
                              max_length=cfg.max_input_length)["input_ids"]
            target_ids = tok(target, add_special_tokens=False, truncation=True,
                              max_length=cfg.max_total_length - len(prompt_ids) - 1)["input_ids"]
            target_ids = target_ids + [eos_id] if eos_id is not None else target_ids
            input_ids = prompt_ids + target_ids
            # Mask prompt portion with -100 so loss is on response only
            labels = [-100] * len(prompt_ids) + target_ids
            attn_mask = [1] * len(input_ids)
            enc.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attn_mask,
            })
        return enc

    train_ds = s["Dataset"].from_list(encode(train_pairs))
    val_ds = s["Dataset"].from_list(encode(val_pairs))

    ta = s["TrainingArguments"](
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
        save_total_limit=1,
        save_only_model=True,
        eval_strategy="epoch",
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
        remove_unused_columns=False,
    )

    # Use DataCollatorForSeq2Seq even for causal LM — it pads input_ids and
    # labels consistently with -100 padding for label tensor.
    collator = s["DataCollatorForSeq2Seq"](tok, model=model, padding=True,
                                            label_pad_token_id=-100)

    import inspect
    _params = inspect.signature(s["Trainer"].__init__).parameters
    _tok_kw = {"processing_class": tok} if "processing_class" in _params else {"tokenizer": tok}
    trainer = s["Trainer"](
        model=model, args=ta,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
        **_tok_kw,
    )

    trainer.train()
    final = out / "final"
    if cfg.use_lora:
        # Save LoRA adapter only; merge later if needed
        model.save_pretrained(final)
    else:
        model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"\n✓ saved Jais i'rāb model to {final}", flush=True)


if __name__ == "__main__":
    main()
