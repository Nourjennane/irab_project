#!/bin/bash
#SBATCH --job-name=irab-smoke
#SBATCH --partition=debug_gpunew
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/smoke_%j.out

# 15-minute smoke test on a debug A100. Verifies:
#   - dataset cache loads
#   - per-word decoder + Lightning trainer runs 1 epoch
# Run this once before launching the long training jobs.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4
source activate irab

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Quick 1-epoch smoke on the small config
python -m irab_tashkeel.training.cli \
    --config configs/model_small.yaml \
    --output-dir runs/smoke_${SLURM_JOB_ID}
