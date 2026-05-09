#!/bin/bash
#SBATCH --job-name=irab-nextgen
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3415496/irab_project/logs/nextgen_train_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/nextgen_train_%j.err

# Next-gen curriculum training entry point.
# Resumes automatically from the latest stage checkpoint if present.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/nextgen

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Run for a single 4-hour SLURM slot. Re-submit to continue across stages.
# Note: fp32 warm-start (Phase 3-A trained in fp32). Use of --bf16 caused
# NaN loss in 491616. Stick with fp32 unless we re-train Phase 3-A in bf16.
# Lower LR (1e-5) since we're warm-starting from a converged checkpoint;
# at 5e-5 the optimiser can blow up when paired with the new head weighting.
PYTHONPATH=src python -u scripts/training_v2/train_curriculum.py \
    --output_root runs/nextgen \
    --warm_start  runs/phase3a_491240/final \
    --batch_size  16 \
    --lr          1e-5 \
    --eval_every  200 \
    --max_total_steps 60000 \
    --eval_max_sentences 100

echo "=== nextgen curriculum training step finished ==="
ls -la runs/nextgen/ 2>&1
