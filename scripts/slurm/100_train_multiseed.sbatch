#!/bin/bash
#SBATCH --job-name=irab-mseed
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/3415496/irab_project/logs/multiseed_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/multiseed_%j.err

# Multi-seed ablation of the validated recovery training. Submit this
# script multiple times with --export=SEED=<n> to get parallel runs.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SEED="${SEED:-0}"
mkdir -p logs runs/multiseed_s${SEED}

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHONPATH=src python -u scripts/training_v2/train_curriculum.py \
    --output_root runs/multiseed_s${SEED} \
    --warm_start  runs/phase3a_491240/final \
    --batch_size 16 --lr 1e-5 \
    --eval_every 200 \
    --max_total_steps 60000 \
    --eval_max_sentences 100 \
    --seed ${SEED} \
    --use_hard_failure_sampler \
    --label_smoothing 0.05 \
    --entropy_reg_lambda 0.01 \
    --consistency_lambda 0.20 \
    --fully_aux_lambda 0.50 \
    --use_ema \
    --early_stop_patience 3 \
    --use_swa --swa_start_step 2000 \
    --use_llrd --llrd_decay 0.85

echo "=== seed ${SEED} done ==="
