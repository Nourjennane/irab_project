#!/bin/bash
#SBATCH --job-name=irab-bb
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3415496/irab_project/logs/backbone_%A_%a.out
#SBATCH --error=/home/3415496/irab_project/logs/backbone_%A_%a.err
#SBATCH --array=0-9

# Backbone benchmark — Step 6.
# Array task index 0..9 maps to one backbone in the registry order
# (see src/irab_tashkeel/backbones/registry.py).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/backbone_benchmark

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=0     # tools may need to download new backbones
export TRANSFORMERS_OFFLINE=0

# Map array index → backbone_id (matches registry order)
BACKBONES=(
    "arat5v2-base"
    "arat5v2-large"
    "arabart-base"
    "camelbert-msa"
    "camelbert-ca"
    "camelbert-mix"
    "arabert-large-v02"
    "mt5-base"
    "mt5-large"
    "xlm-r-base"
)

BB="${BACKBONES[$SLURM_ARRAY_TASK_ID]}"
echo "=== array task $SLURM_ARRAY_TASK_ID → backbone $BB ==="

PYTHONPATH=src python -u scripts/backbones/run_backbone.py \
    --backbone "$BB" \
    --config   phase3a_full \
    --batch_size 32 \
    --lr 5e-5 \
    --bf16

echo "=== backbone $BB run finished ==="
