"""Mix A Phase 2 — fine-tune AraT5v2 on the *marker* sub-task only.

The Hybrid system (Claude RAG case+role + AraT5v2 marker) is the project's
headline contribution. Training a marker-specialist is much easier than full
i'rāb generation:

  - Input:  structured tag + sentence context + target word
  - Output: a single short phrase from the closed marker vocabulary
            (~200 unique phrases in our data; top-15 cover ≥90%)

Training data: `data/marker_pairs.jsonl` produced by
`evaluation.marker_extract`. ~8.8k pairs, 89.7% with a real marker, 11%
labelled `<NO_MARKER>` (mabniyy-mahall and unparsed Claude outputs).

Input template:
    أعرب علامة: <word> | في: <sentence> | الحالة: <case> | المحل: <role>

Output:
    <marker phrase>  or  <NO_MARKER>

Run pattern:
    # smoke (laptop, ≤100 examples, 1 epoch)
    python -m irab_tashkeel.training.llm.marker_arat5_sft \\
        --config configs/marker_arat5v2.yaml --output runs/marker_smoke \\
        --max_steps 50

    # full run on Bocconi
    sbatch scripts/slurm/33_train_marker_arat5v2.sbatch
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


NO_MARKER = "<NO_MARKER>"


@dataclass
class MarkerConfig:
    model_id: str = "UBC-NLP/AraT5v2-base-1024"
    pairs_path: str = "data/marker_pairs.jsonl"
    val_split: float = 0.05

    # Training
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    n_epochs: int = 5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_input_length: int = 256
    max_target_length: int = 32           # markers are short
    label_smoothing_factor: float = 0.1

    # Filtering
    drop_no_marker_in_train: bool = False  # set True to train only on markered words
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MarkerConfig":
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _format_input(pair: dict, max_sent_len: int = 200) -> str:
    """Build the structured input prompt for the seq2seq model.

    Conditioning fields (case, role) are passed even when missing — empty
    strings still teach the model "this combination saw any marker."
    """
    word = (pair.get("word") or "").strip()
    sent = (pair.get("sentence") or "").strip()
    if len(sent) > max_sent_len:
        sent = sent[:max_sent_len]
    case = (pair.get("case") or "").strip() or "-"
    role = (pair.get("role") or "").strip() or "-"
    return f"أعرب علامة: {word} | في: {sent} | الحالة: {case} | المحل: {role}"


def _load_pairs(path: Path | str) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("word") or not row.get("sentence"):
                continue
            if not row.get("marker_target"):
                continue
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fine-tune AraT5v2 on the marker sub-task")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--limit_train", type=int, default=-1,
                        help="cap training rows (smoke runs)")
    parser.add_argument("--report_to", default="none")
    args = parser.parse_args()

    cfg = MarkerConfig.from_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "marker_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    s = _import_stack()
    torch = s["torch"]
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"loading {cfg.model_id}")
    tok = s["AutoTokenizer"].from_pretrained(cfg.model_id)
    model = s["AutoModelForSeq2SeqLM"].from_pretrained(cfg.model_id)

    pairs = _load_pairs(cfg.pairs_path)
    print(f"raw pairs: {len(pairs)}")
    if cfg.drop_no_marker_in_train:
        pairs = [p for p in pairs if p["marker_target"] != NO_MARKER]
        print(f"after dropping <NO_MARKER>: {len(pairs)}")

    random.Random(cfg.seed).shuffle(pairs)
    if args.limit_train > 0:
        pairs = pairs[: args.limit_train]
        print(f"smoke mode — limited to {len(pairs)} pairs")

    n_val = max(1, int(len(pairs) * cfg.val_split))
    val_pairs = pairs[-n_val:]
    train_pairs = pairs[:-n_val]
    print(f"  train={len(train_pairs)}  val={len(val_pairs)}")

    def encode(rows: List[dict]) -> List[Dict]:
        enc: List[Dict] = []
        for p in rows:
            inp = tok(_format_input(p),
                      truncation=True, max_length=cfg.max_input_length)
            tgt = tok(p["marker_target"],
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
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        predict_with_generate=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="adamw_bnb_8bit",        # 4× smaller optimizer state vs adamw_torch
        gradient_checkpointing=True,
        report_to=args.report_to,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        seed=cfg.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    collator = s["DataCollatorForSeq2Seq"](tok, model=model)
    trainer = s["Seq2SeqTrainer"](
        model=model, args=ta,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tok,
    )

    trainer.train()
    final = out / "final"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"\n✓ saved marker model to {final}")


if __name__ == "__main__":
    main()
