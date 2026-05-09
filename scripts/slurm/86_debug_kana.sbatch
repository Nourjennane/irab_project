#!/bin/bash
#SBATCH --job-name=irab-pR2-dbg
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3415496/irab_project/logs/debug_kana_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/debug_kana_%j.err

# Phase R2 — KanaReasoner failure trace dump (Gazelle + MASAQ)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/phaseR2_debug

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== Gazelle kana trace dump ==="
PYTHONPATH=src python -u scripts/structured/debug_kana_failures.py \
    --eval gazelle \
    --out_dir runs/phaseR2_debug/kana_failures_gazelle

echo
echo "=== MASAQ kana trace dump (kana-only filter; for comparison) ==="
PYTHONPATH=src python -u scripts/structured/debug_kana_failures.py \
    --eval masaq \
    --out_dir runs/phaseR2_debug/kana_failures_masaq

echo
echo "=== Debug dump complete ==="
ls -la runs/phaseR2_debug/
