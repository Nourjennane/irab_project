#!/bin/bash
#SBATCH --job-name=irab-pR-eval
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/home/3415496/irab_project/logs/eval_phaseR_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/eval_phaseR_%j.err

# Phase R — R-0 (lambda=0 sanity) + R-C (lambda=0.3 default) per-construction eval
# on Phase 3-A. Pre-built grammar memory at data/grammar_memory/.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL=runs/phase3a_491240/final
MEMORY=data/grammar_memory/

# R-0: lambda=0 (must reproduce Phase 3-A baseline byte-equivalent on construction-touched spans)
echo "=== R-0 (lambda=0 sanity) Gazelle ==="
PYTHONPATH=src python -u scripts/structured/eval_per_construction.py \
    --model "$MODEL" --eval gazelle \
    --use_retrieval --retrieval_memory "$MEMORY" --retrieval_lambda 0.0 --retrieval_k 5 \
    --out_dir runs/phaseR_R0/gazelle

echo
echo "=== R-0 (lambda=0 sanity) MASAQ ==="
PYTHONPATH=src python -u scripts/structured/eval_per_construction.py \
    --model "$MODEL" --eval masaq \
    --use_retrieval --retrieval_memory "$MEMORY" --retrieval_lambda 0.0 --retrieval_k 5 \
    --out_dir runs/phaseR_R0/masaq

echo
echo "=== R-C (lambda=0.3 default full system) Gazelle ==="
PYTHONPATH=src python -u scripts/structured/eval_per_construction.py \
    --model "$MODEL" --eval gazelle \
    --use_retrieval --retrieval_memory "$MEMORY" --retrieval_lambda 0.3 --retrieval_k 5 \
    --out_dir runs/phaseR_RC/gazelle

echo
echo "=== R-C (lambda=0.3 default full system) MASAQ ==="
PYTHONPATH=src python -u scripts/structured/eval_per_construction.py \
    --model "$MODEL" --eval masaq \
    --use_retrieval --retrieval_memory "$MEMORY" --retrieval_lambda 0.3 --retrieval_k 5 \
    --out_dir runs/phaseR_RC/masaq

echo
echo "=== Phase R eval done ==="
ls -la runs/phaseR_R0/gazelle/ runs/phaseR_RC/gazelle/ 2>/dev/null
