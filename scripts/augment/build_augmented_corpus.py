"""Build augmented training corpus: distill_v2 + synthetic rare constructions.

Concatenates:
- data/structured_v1/{train,val}.jsonl  (distill_v2)
- data/structured_v1_augmented/synthetic/all_synthetic.jsonl  (Phase #39)

→ data/structured_v1_augmented/{train,val}.jsonl

Validation split: 5% of synthetic goes into val (matches distill_v2's
val ratio); rest into train. Original distill_v2 train/val split is
preserved exactly (no remixing).

Usage:
    python scripts/augment/build_augmented_corpus.py \\
        --in_train data/structured_v1/train.jsonl \\
        --in_val   data/structured_v1/val.jsonl \\
        --in_synth data/structured_v1_augmented/synthetic/all_synthetic.jsonl \\
        --out_dir  data/structured_v1_augmented/ \\
        --val_frac_synth 0.05 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: Path) -> List[Dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def write_jsonl(records: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_train", required=True)
    ap.add_argument("--in_val", required=True)
    ap.add_argument("--in_synth", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_frac_synth", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)

    train_orig = load_jsonl(Path(args.in_train))
    val_orig = load_jsonl(Path(args.in_val))
    synth = load_jsonl(Path(args.in_synth))

    rng.shuffle(synth)
    n_synth_val = max(1, int(len(synth) * args.val_frac_synth))
    synth_val = synth[:n_synth_val]
    synth_train = synth[n_synth_val:]

    train_combined = train_orig + synth_train
    val_combined = val_orig + synth_val
    rng.shuffle(train_combined)
    rng.shuffle(val_combined)

    write_jsonl(train_combined, out_dir / "train.jsonl")
    write_jsonl(val_combined, out_dir / "val.jsonl")

    summary = {
        "train_orig": len(train_orig),
        "val_orig": len(val_orig),
        "synth_total": len(synth),
        "synth_train": len(synth_train),
        "synth_val": len(synth_val),
        "train_combined": len(train_combined),
        "val_combined": len(val_combined),
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Built augmented corpus:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n→ {out_dir}/train.jsonl")
    print(f"→ {out_dir}/val.jsonl")


if __name__ == "__main__":
    main()
