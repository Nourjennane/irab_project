#!/bin/bash
#SBATCH --job-name=irab-pc-base
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3415496/irab_project/logs/per_construction_phase3a_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/per_construction_phase3a_%j.err

# Per-construction eval on Phase 3-A baseline. Establishes the "before"
# numbers for kāna sisters / inna sisters / istithnāʾ / mawṣūl / iḍāfa /
# Quranic proxy on Gazelle + MASAQ. Augmentation v2 will be measured
# against these baseline numbers.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL=runs/phase3a_491240/final
OUT_BASE=runs/per_construction_phase3a

mkdir -p "$OUT_BASE/gazelle" "$OUT_BASE/masaq"

echo "=== Per-construction eval: Gazelle ==="
PYTHONPATH=src python -u scripts/structured/eval_per_construction.py \
    --model "$MODEL" \
    --eval gazelle \
    --out_dir "$OUT_BASE/gazelle"

echo
echo "=== Per-construction eval: MASAQ ==="
PYTHONPATH=src python -u scripts/structured/eval_per_construction.py \
    --model "$MODEL" \
    --eval masaq \
    --out_dir "$OUT_BASE/masaq"

echo
echo "=== Done. Summaries: ==="
ls -la "$OUT_BASE/gazelle/per_construction_summary."{json,md} 2>/dev/null
ls -la "$OUT_BASE/masaq/per_construction_summary."{json,md} 2>/dev/null
