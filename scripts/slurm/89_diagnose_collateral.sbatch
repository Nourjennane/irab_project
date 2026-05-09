#!/bin/bash
#SBATCH --job-name=irab-clt
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/home/3415496/irab_project/logs/diagnose_collateral_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/diagnose_collateral_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/phaseR2_collateral

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHONPATH=src python -u scripts/structured/diagnose_collateral.py \
    --out_dir runs/phaseR2_collateral

echo "=== diagnostic complete ==="
ls -la runs/phaseR2_collateral/
