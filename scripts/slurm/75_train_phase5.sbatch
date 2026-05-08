#!/bin/bash
#SBATCH --job-name=irab-p5
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/3415496/irab_project/logs/train_phase5_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_phase5_%j.err

# Phase 5 Run 1 (gate) — hierarchical case decoder, joint training.
# Layered on Phase 3-A (production: 56.7 / 41.3 / 44.8 / 20.1).
# Decision gate (soft): beat Phase 3-A on >=2 of {case, role-F1, fully},
# no regression > 1.0 pp.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

OUT=runs/phase5_${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)" > "$OUT/provenance.txt"
echo "config=configs/phase5_case_hierarchy.yaml" >> "$OUT/provenance.txt"
echo "submitted=$(date -Iseconds)" >> "$OUT/provenance.txt"
cp configs/phase5_case_hierarchy.yaml "$OUT/config_snapshot.yaml"

python -u -m irab_tashkeel.training.structured.train \
    --config configs/phase5_case_hierarchy.yaml \
    --output_dir "$OUT"

echo "=== Phase 5 done. final model in $OUT/final ==="
ls -la "$OUT/final" | head -15
du -sh "$OUT" 2>/dev/null || true
