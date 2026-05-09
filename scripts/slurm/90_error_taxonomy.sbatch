#!/bin/bash
#SBATCH --job-name=irab-tax
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/home/3415496/irab_project/logs/error_taxonomy_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/error_taxonomy_%j.err

# Step 16 — error taxonomy population pass on Phase 3-A
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs docs/error_analysis_v2

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PYTHONPATH=src python -u scripts/error_taxonomy/build_taxonomy.py \
    --model runs/phase3a_491240/final \
    --out_dir docs/error_analysis_v2 \
    --with_stanza

echo
echo "=== taxonomy build complete ==="
ls -la docs/error_analysis_v2/
