#!/bin/bash
#SBATCH --job-name=irab-recovery
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3415496/irab_project/logs/recovery_train_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/recovery_train_%j.err

# Generalization-recovery curriculum training.
# Strict no-leakage: gazelle_test, masaq_quranic, ud_padt_test never enter
# the training pool. Eval set is loaded SEPARATELY from disk.
#
# All 14 recovery-patch items enabled by default flags. Toggle each off
# via the explicit CLI flag if you want to ablate.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/nextgen_recovery

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHONPATH=src python -u scripts/training_v2/train_curriculum.py \
    --output_root runs/nextgen_recovery \
    --warm_start  runs/phase3a_491240/final \
    --batch_size  16 \
    --lr          1e-5 \
    --eval_every  200 \
    --max_total_steps 60000 \
    --eval_max_sentences 100 \
    --use_hard_failure_sampler \
    --contrastive_lambda  0.15 \
    --enable_graph_refiner \
    --label_smoothing     0.05 \
    --entropy_reg_lambda  0.01 \
    --consistency_lambda  0.20 \
    --fully_aux_lambda    0.50 \
    --construction_dropout_p 0.12 \
    --use_ema \
    --early_stop_patience 3 \
    --use_swa --swa_start_step 2000 \
    --use_llrd --llrd_decay 0.85

echo "=== recovery curriculum training step finished ==="
ls -la runs/nextgen_recovery/ 2>&1
