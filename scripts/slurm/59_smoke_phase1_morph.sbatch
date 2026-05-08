#!/bin/bash
#SBATCH --job-name=irab-p1-smoke
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/home/3415496/irab_project/logs/smoke_phase1_morph_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/smoke_phase1_morph_%j.err

# Phase 1 smoke — 100 sentences, 1 epoch. Verifies:
#  - merged corpus loads
#  - masked multi-task forward works (no NaN)
#  - per-head losses appear in train log
#  - both UD-PADT and distill_v2 examples seen in same batch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

OUT=runs/phase1_morph_smoke_${SLURM_JOB_ID}

python -u -m irab_tashkeel.training.structured.train \
    --config configs/phase1_morphology.yaml \
    --output_dir "$OUT" \
    --smoke --smoke_n 100 --smoke_epochs 1

echo "=== smoke done ==="
ls -la "$OUT" 2>/dev/null | head -10
