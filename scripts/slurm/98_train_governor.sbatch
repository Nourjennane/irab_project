#!/bin/bash
#SBATCH --job-name=irab-gov
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3415496/irab_project/logs/gov_train_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/gov_train_%j.err

# Governor-head training. Warm-starts from validated_recovery (clean,
# leak-free baseline) and adds the biaffine governor head + attachment
# contrastive loss. Strict no-leakage assertions enforced.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/nextgen_governor

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHONPATH=src python -u scripts/training_v2/train_curriculum.py \
    --output_root runs/nextgen_governor \
    --warm_start  runs/validated_nextgen_recovery \
    --batch_size  16 \
    --lr          1e-5 \
    --eval_every  200 \
    --max_total_steps 60000 \
    --eval_max_sentences 100 \
    --enable_governor_head \
    --attachment_contrastive_lambda 0.1 \
    --use_hard_failure_sampler \
    --label_smoothing     0.05 \
    --entropy_reg_lambda  0.01 \
    --consistency_lambda  0.20 \
    --fully_aux_lambda    0.50 \
    --use_ema \
    --early_stop_patience 3 \
    --use_swa --swa_start_step 2000 \
    --use_llrd --llrd_decay 0.85

echo "=== governor training step finished ==="
ls -la runs/nextgen_governor/ 2>&1
