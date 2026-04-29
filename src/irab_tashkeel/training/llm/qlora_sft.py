"""QLoRA SFT for Arabic LLM i'rāb generation (Stack A of the research plan).

Recommended base models (set via --model):
  - QCRI/Fanar-1-9B-Instruct          (best MSA per the plan; ~9B)
  - ALLaM-AI/ALLaM-7B-Instruct-preview (top AraLingBench; ~7B)
  - almaghrabima/Yehia-7B               (best Arabic syntax score; ~7B)

Hardware target: A100 40GB / 80GB or H100. Fits in 4-bit NF4 with LoRA r=32.

Run pattern (Bocconi HPC):
    sbatch scripts/slurm/train_qlora_fanar.sbatch

Or interactive on a debug GPU:
    python -m irab_tashkeel.training.llm.qlora_sft \\
        --config configs/llm_qlora.yaml --max_steps 50 --output runs/qlora_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# Lazy heavy imports — only load when actually training, not on --help.
def _import_training_stack():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    )
    return {
        "torch": torch, "Dataset": Dataset,
        "LoraConfig": LoraConfig, "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer, "TrainingArguments": TrainingArguments,
    }


@dataclass
class QLoraConfig:
    # Model
    model_id: str = "QCRI/Fanar-1-9B-Instruct"
    use_flash_attn_2: bool = True
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    # Training
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    n_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 1024
    # Curriculum (max_words per epoch). None means no cap.
    curriculum: List[Optional[int]] = field(default_factory=lambda: [8, 16, None])
    # Data
    dataset_cache: str = "data/cache/combined.pkl"
    val_split: float = 0.02
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QLoraConfig":
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def build_pairs(cache_path: Path, seed: int):
    """Load the cached dataset and convert all eligible examples to SFT pairs."""
    from ...data.build_dataset import load_examples
    from .format import examples_to_pairs

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Dataset cache not found: {cache_path}. "
            f"Run `python -m irab_tashkeel.training.cli --config configs/model_small.yaml --force-rebuild` "
            f"once to materialize it."
        )
    examples = load_examples(cache_path)
    pairs = examples_to_pairs(examples)
    random.Random(seed).shuffle(pairs)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT for Arabic i'rāb generation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=-1, help="cap training steps (smoke runs)")
    parser.add_argument("--report_to", type=str, default="none")
    args = parser.parse_args()

    cfg = QLoraConfig.from_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "qlora_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    # Heavy imports
    s = _import_training_stack()
    torch = s["torch"]
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- Tokenizer + model ----
    print(f"Loading tokenizer: {cfg.model_id}")
    tokenizer = s["AutoTokenizer"].from_pretrained(cfg.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = None
    if cfg.load_in_4bit:
        bnb = s["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )

    print(f"Loading model: {cfg.model_id}  (4-bit={cfg.load_in_4bit})")
    model_kwargs = dict(
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if cfg.use_flash_attn_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = s["AutoModelForCausalLM"].from_pretrained(cfg.model_id, **model_kwargs)
    model = s["prepare_model_for_kbit_training"](model)

    lora = s["LoraConfig"](
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=cfg.lora_target_modules,
    )
    model = s["get_peft_model"](model, lora)
    model.print_trainable_parameters()

    # ---- Data ----
    pairs = build_pairs(Path(cfg.dataset_cache), cfg.seed)
    print(f"SFT pairs total: {len(pairs)}")
    n_val = max(1, int(len(pairs) * cfg.val_split))
    val_pairs = pairs[-n_val:]
    train_pairs = pairs[:-n_val]
    print(f"  train: {len(train_pairs)}  val: {len(val_pairs)}")

    from .format import pair_to_chat

    def encode(pair_list):
        rows = []
        for p in pair_list:
            messages = pair_to_chat(p)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            rows.append({"text": text, "n_words": p.n_words})
        return rows

    val_rows = encode(val_pairs)
    val_ds = s["Dataset"].from_list(val_rows).map(
        lambda r: tokenizer(r["text"], truncation=True, max_length=cfg.max_seq_length),
        batched=False, remove_columns=["text", "n_words"],
    )

    # ---- Training: one Trainer per curriculum stage ----
    epoch_caps = cfg.curriculum or [None]
    if len(epoch_caps) != cfg.n_epochs:
        # Pad/truncate to n_epochs by repeating the last cap.
        if len(epoch_caps) < cfg.n_epochs:
            epoch_caps = list(epoch_caps) + [epoch_caps[-1]] * (cfg.n_epochs - len(epoch_caps))
        else:
            epoch_caps = epoch_caps[: cfg.n_epochs]

    last_ckpt: Optional[str] = None
    for epoch_idx, cap in enumerate(epoch_caps):
        stage_pairs = [p for p in train_pairs if cap is None or p.n_words <= cap]
        print(f"\n=== Epoch {epoch_idx+1}/{cfg.n_epochs}  cap={cap}  pairs={len(stage_pairs)} ===")
        if not stage_pairs:
            print("  no pairs at this cap; skipping stage")
            continue
        train_rows = encode(stage_pairs)
        train_ds = s["Dataset"].from_list(train_rows).map(
            lambda r: tokenizer(r["text"], truncation=True, max_length=cfg.max_seq_length),
            batched=False, remove_columns=["text", "n_words"],
        )
        ta = s["TrainingArguments"](
            output_dir=str(out / f"stage_{epoch_idx+1}"),
            num_train_epochs=1,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type,
            logging_steps=20,
            save_strategy="epoch",
            eval_strategy="no",
            bf16=True,
            optim="paged_adamw_8bit",
            report_to=args.report_to,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            seed=cfg.seed + epoch_idx,
            ddp_find_unused_parameters=False,
        )
        collator = s["DataCollatorForLanguageModeling"](tokenizer=tokenizer, mlm=False)
        trainer = s["Trainer"](
            model=model, args=ta,
            train_dataset=train_ds,
            data_collator=collator,
            processing_class=tokenizer,
        )
        trainer.train(resume_from_checkpoint=last_ckpt)
        last_ckpt = str(out / f"stage_{epoch_idx+1}")

    # ---- Final adapter save ----
    adapter_path = out / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n✓ saved LoRA adapter to {adapter_path}")


if __name__ == "__main__":
    main()
