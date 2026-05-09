"""Per-backbone training driver — runs a single backbone through one
of three benchmark configurations.

Configurations (per ``docs/roadmap/backbone_upgrade.md``):

  - ``phase1_transfer`` : retrain only Phase 1 morph heads on the
                           existing supervision; compare UD-PADT macro-F1
                           against the frozen baseline's 98.4%.
  - ``phase3_transfer`` : retrain Phase 1 morph + Phase 3 dep features;
                           compare Gazelle case + marker against 56.7 / 44.8.
  - ``phase3a_full``    : full Phase 3-A retrain on the same recipe;
                           compare overall headlines against 25.2 / 14.9.

The driver writes results to
``runs/backbone_benchmark/<backbone_id>/<config>/`` so the
matrix renderer can pick them up.

Usage::

    PYTHONPATH=src python scripts/backbones/run_backbone.py \\
        --backbone arabart-base \\
        --config   phase3a_full \\
        --epochs   6 \\
        --output_root runs/backbone_benchmark
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    help="backbone id from src/irab_tashkeel/backbones/registry.py")
    ap.add_argument("--config", required=True,
                    choices=("phase1_transfer", "phase3_transfer", "phase3a_full"))
    ap.add_argument("--output_root", default=str(ROOT / "runs" / "backbone_benchmark"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from irab_tashkeel.backbones import get_backbone

    spec = get_backbone(args.backbone)
    print("=" * 70)
    print(f"Backbone benchmark: {spec.backbone_id} ({spec.n_params_est})")
    print(f"  hf_model_id: {spec.hf_model_id}")
    print(f"  arch:        {spec.arch_family}")
    print(f"  pretraining: {spec.arabic_pretraining}")
    print(f"  config:      {args.config}")
    print("=" * 70)

    out_dir = Path(args.output_root) / args.backbone / args.config
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest written before any heavy work — useful for failed jobs.
    manifest = {
        "backbone_id": spec.backbone_id,
        "hf_model_id": spec.hf_model_id,
        "arch_family": spec.arch_family,
        "arabic_pretraining": spec.arabic_pretraining,
        "config": args.config,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "bf16": args.bf16, "seed": args.seed,
        "started": time.time(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    # ----- Lazy torch import -----
    import torch
    from transformers import AutoTokenizer

    # The training_v2 entry point already supports arbitrary --encoder_name;
    # delegate to it and let it handle the model construction. We only need
    # to set the env / argv appropriately.
    print("\nInvoking training_v2/train_curriculum.py via subprocess...")
    import subprocess
    cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts" / "training_v2" / "train_curriculum.py"),
        "--output_root", str(out_dir),
        "--encoder_name", spec.hf_model_id,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
    ]
    if args.bf16:
        cmd.append("--bf16")
    # phase1_transfer / phase3_transfer would gate on stage 1 / stage 3
    # respectively; for now, both paths run the full curriculum and the
    # comparison renderer slices the per-stage metrics. phase3a_full
    # similarly runs the full schedule.
    rc = subprocess.call(cmd, env=None)
    print(f"\nsubprocess exit code: {rc}")

    manifest["finished"] = time.time()
    manifest["exit_code"] = rc
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
