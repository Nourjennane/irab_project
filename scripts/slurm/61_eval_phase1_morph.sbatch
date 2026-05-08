#!/bin/bash
#SBATCH --job-name=irab-p1-eval
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/home/3415496/irab_project/logs/eval_phase1_morph_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/eval_phase1_morph_%j.err

# Phase 1 eval: morphology accuracy + i'rāb regression check vs rev 2.
# Uses the same iʿrāb eval pipeline as rev 2 (apples-to-apples) plus a
# new per-feature morphology eval on UD-PADT held-out test split.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_DIR="${1:-}"
if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: pass model dir as first arg, e.g. sbatch ... runs/phase1_morph_490987/final"
    exit 2
fi

JOB_ID="$(basename "$(dirname "$MODEL_DIR")" | grep -oE '[0-9]+$' || echo unknown)"
OUT="runs/phase1_morph_eval_${JOB_ID}"
mkdir -p "$OUT/gazelle" "$OUT/masaq" "$OUT/morph"

echo "================================================"
echo "  Phase 1 eval — model: $MODEL_DIR"
echo "  Output: $OUT"
echo "================================================"

# 1. Morphology eval on UD-PADT test set
echo ""
echo "--- 1. Morphology eval (UD-PADT test, n=680) ---"
python -u scripts/morphology/eval_morphology.py \
    --model "$MODEL_DIR" \
    --conllu data/ud_padt/ar_padt-ud-test.conllu \
    --out_dir "$OUT/morph"

# 2. i'rāb regression check on Gazelle (vs rev 2)
echo ""
echo "--- 2. Gazelle iʿrāb (heads only + +constraints, vs rev 2 ablation) ---"
python -u -m irab_tashkeel.evaluation.run_baselines \
    --eval gazelle \
    --baselines structured_v1,structured_v1_constrained \
    --structured_v1_path "$MODEL_DIR" \
    --out "$OUT/gazelle"

# 3. MASAQ deltas
echo ""
echo "--- 3. MASAQ iʿrāb (n=5,007) ---"
python -u -m irab_tashkeel.evaluation.run_baselines \
    --eval masaq \
    --baselines structured_v1,structured_v1_constrained \
    --structured_v1_path "$MODEL_DIR" \
    --out "$OUT/masaq"

echo ""
echo "================================================"
echo "  Phase 1 eval done."
echo "    Morph: $OUT/morph/morphology_summary.json"
echo "    Gazelle: $OUT/gazelle/summary.json"
echo "    MASAQ:   $OUT/masaq/summary.json"
echo "================================================"
