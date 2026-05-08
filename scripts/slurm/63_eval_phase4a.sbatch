#!/bin/bash
#SBATCH --job-name=irab-p4a-eval
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3415496/irab_project/logs/eval_phase4a_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/eval_phase4a_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_DIR="${1:?usage: $0 <model_final_dir>}"

JOB_ID="$(basename "$(dirname "$MODEL_DIR")" | grep -oE '[0-9]+$' || echo unknown)"
VARIANT="$(basename "$(dirname "$MODEL_DIR")" | sed 's/_[0-9]*$//')"
OUT="runs/phase4a_eval_${VARIANT}_${JOB_ID}"

mkdir -p "$OUT/gazelle" "$OUT/masaq" "$OUT/gazelle_constrained"

echo "=== Phase 4a 4-stream eval ==="
echo "model:   $MODEL_DIR"
echo "variant: $VARIANT"
echo "out:     $OUT"
echo

# Gazelle (heads only)
python -u scripts/structured/eval_phase4a.py \
    --model "$MODEL_DIR" --eval gazelle --out_dir "$OUT/gazelle"

# Gazelle + constraints (Phase 1 already shows +0.7-0.8 fully from constraints)
python -u scripts/structured/eval_phase4a.py \
    --model "$MODEL_DIR" --eval gazelle --apply_constraints \
    --out_dir "$OUT/gazelle_constrained"

# MASAQ
python -u scripts/structured/eval_phase4a.py \
    --model "$MODEL_DIR" --eval masaq --out_dir "$OUT/masaq"

echo
echo "=== Phase 4a eval done. ==="
ls "$OUT"/*/phase4a_summary.json 2>/dev/null
