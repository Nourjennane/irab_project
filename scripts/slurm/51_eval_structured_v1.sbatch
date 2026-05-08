#!/bin/bash
#SBATCH --job-name=irab-struc-eval
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/home/3415496/irab_project/logs/eval_structured_v1_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/eval_structured_v1_%j.err

# Phase 3 / v1 rebuild — Gazelle + MASAQ eval, two configurations
# (heads-only vs heads + 4 symbolic constraints).
# Pass the final/ dir as the first argument.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

MODEL_DIR="${1:-}"
if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: pass model dir as first arg, e.g. sbatch ... /path/to/runs/structured_v1_rebuild_xxx/final"
    exit 2
fi

bash scripts/structured/eval_structured_v1.sh "$MODEL_DIR"
