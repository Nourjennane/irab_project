#!/bin/bash
#SBATCH --job-name=irab-p6
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/3415496/irab_project/logs/train_phase6_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_phase6_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

OUT=runs/phase6_${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)" > "$OUT/provenance.txt"
echo "config=configs/phase6_marker_hierarchy.yaml" >> "$OUT/provenance.txt"
cp configs/phase6_marker_hierarchy.yaml "$OUT/config_snapshot.yaml"

python -u -m irab_tashkeel.training.structured.train \
    --config configs/phase6_marker_hierarchy.yaml \
    --output_dir "$OUT"

echo "=== Phase 6 done ==="
ls -la "$OUT/final" | head -15
