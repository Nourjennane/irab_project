"""QLoRA SFT with **Unsloth + Liger Kernels + sequence packing** (~3-4× faster
than the vanilla `qlora_sft.py` for the same training data and quality).

Key differences from `qlora_sft.py`:
  - `unsloth.FastLanguageModel` replaces transformers' `AutoModelForCausalLM`
    + `BitsAndBytesConfig`. Unsloth ships hand-derived backward passes and
    fused 4-bit kernels (≈2× speed, 70% less VRAM per Unsloth's benchmarks).
  - `liger_kernel` patches RMSNorm / RoPE / fused cross-entropy (≈+20%
    throughput, -60% memory).
  - `trl.SFTTrainer` with `packing=True` packs multiple short examples per
    sequence — i'rāb pairs are short, so this is a 2× win on wall time.
  - `max_seq_length=512` (your i'rāb pairs fit comfortably).

The data path is identical to `qlora_sft.py` so you can A/B them on the same
cache. Curriculum is the same length-based 3-stage schedule.

Run pattern (Bocconi HPC, MIG 4g.40gb):
    sbatch scripts/slurm/30_train_qlora_fanar_unsloth.sbatch
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
    """Lazy import of the heavy training stack so --help is instant."""
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    # Liger patches happen on import side-effect when we apply them in main()
    from trl import SFTConfig, SFTTrainer
    return {
        "torch": torch, "Dataset": Dataset,
        "FastLanguageModel": FastLanguageModel,
        "SFTConfig": SFTConfig, "SFTTrainer": SFTTrainer,
    }


@dataclass
class UnslothConfig:
    model_id: str = "QCRI/Fanar-1-9B-Instruct"
    max_seq_length: int = 512                 # most i'rāb pairs are well under 512
    load_in_4bit: bool = True
    use_liger_kernels: bool = True
    use_packing: bool = True

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
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    curriculum: List[Optional[int]] = field(default_factory=lambda: [8, 16, None])

    # Data
    dataset_cache: str = "data/cache/combined.pkl"
    drop_templated: bool = True               # follow the plan's recommendation
    val_split: float = 0.02
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "UnslothConfig":
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def build_pairs(cache_path: Path, drop_templated: bool, seed: int):
    """Load the cached dataset and convert to SFT pairs.

    `drop_templated=True` excludes QAC-templated and PADT-templated examples,
    keeping only Yarob (manual) + distilled (LLM-teacher). This is what the
    research plan recommends: real data > template-derived data for fine-tuning.
    """
    from ...data.build_dataset import load_examples
    from .format import examples_to_pairs

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Dataset cache not found: {cache_path}. Run `python -m irab_tashkeel.training.cli` "
            f"once with --force-rebuild to materialize it."
        )
    examples = load_examples(cache_path)
    if drop_templated:
        kept = [e for e in examples if e.source in {"yarob", "distilled", "i3rab", "gazelle"}]
        print(f"  drop_templated: kept {len(kept)}/{len(examples)} examples (sources: yarob/distilled/i3rab)")
        examples = kept
    pairs = examples_to_pairs(examples)
    random.Random(seed).shuffle(pairs)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Unsloth + Liger + packing QLoRA SFT")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--report_to", type=str, default="none")
    args = parser.parse_args()

    cfg = UnslothConfig.from_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "unsloth_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    s = _import_stack()
    torch = s["torch"]
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- Liger Kernels (apply patches BEFORE loading the model) ----
    if cfg.use_liger_kernels:
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance  # type: ignore  # noqa: F401
            print("liger_kernel available — patches applied at model-load time")
        except Exception as e:
            print(f"⚠ liger_kernel not available ({e}); continuing without it")

    # ---- Model + tokenizer (Unsloth) ----
    print(f"Loading {cfg.model_id} via Unsloth (4-bit={cfg.load_in_4bit})")
    model, tokenizer = s["FastLanguageModel"].from_pretrained(
        model_name=cfg.model_id,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=None,           # auto: bf16 on Ampere/Hopper, fp16 elsewhere
    )

    # Liger fused-CE patch on the loaded instance, when available.
    if cfg.use_liger_kernels:
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance
            _apply_liger_kernel_to_instance(model=model)
            print("liger_kernel: patches applied to model")
        except Exception as e:
            print(f"⚠ liger patch failed ({e}); proceeding without")

    # LoRA via Unsloth's helper (uses rsLoRA-compatible scaling).
    model = s["FastLanguageModel"].get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        max_seq_length=cfg.max_seq_length,
    )

    # ---- Data (with the drop_templated knob) ----
    pairs = build_pairs(Path(cfg.dataset_cache), cfg.drop_templated, cfg.seed)
    print(f"SFT pairs total: {len(pairs)}")
    n_val = max(1, int(len(pairs) * cfg.val_split))
    val_pairs = pairs[-n_val:]
    train_pairs = pairs[:-n_val]
    print(f"  train: {len(train_pairs)}  val: {len(val_pairs)}")

    from .format import pair_to_chat

    def pair_to_text(p):
        msgs = pair_to_chat(p)
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    # ---- Per-stage curriculum trainer ----
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
        train_ds = s["Dataset"].from_list([{"text": pair_to_text(p)} for p in stage_pairs])

        sft_args = s["SFTConfig"](
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
            packing=cfg.use_packing,
            max_length=cfg.max_seq_length,
            dataset_text_field="text",
        )
        trainer = s["SFTTrainer"](
            model=model,
            args=sft_args,
            train_dataset=train_ds,
            processing_class=tokenizer,
        )
        trainer.train(resume_from_checkpoint=last_ckpt)
        last_ckpt = str(out / f"stage_{epoch_idx+1}")

    # ---- Save adapter ----
    adapter = out / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    print(f"\n✓ saved Unsloth-trained LoRA adapter to {adapter}")


if __name__ == "__main__":
    main()
