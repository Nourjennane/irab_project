#!/bin/bash
#SBATCH --job-name=irab-p2-v3-add
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/home/3415496/irab_project/logs/train_phase2_v3_additive_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_phase2_v3_additive_%j.err

# Phase 2 Run 2 — v3 + additive bias + soft + joint. Mechanism ablation.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

OUT=runs/phase2_v3_additive_${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)" > "$OUT/provenance.txt"
echo "config=configs/phase2_v3_additive_joint.yaml" >> "$OUT/provenance.txt"
echo "submitted=$(date -Iseconds)" >> "$OUT/provenance.txt"
cp configs/phase2_v3_additive_joint.yaml "$OUT/config_snapshot.yaml"

python -u -m irab_tashkeel.training.structured.train \
    --config configs/phase2_v3_additive_joint.yaml \
    --output_dir "$OUT"

echo "=== Phase 2 v3 + additive (Run 2) done. final model in $OUT/final ==="
ls -la "$OUT/final" | head -15
du -sh "$OUT" 2>/dev/null || true
