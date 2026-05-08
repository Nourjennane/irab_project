#!/bin/bash
#SBATCH --job-name=irab-p2-evlt
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/home/3415496/irab_project/logs/eval_phase2_light_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/eval_phase2_light_%j.err

# Phase 2 LIGHTWEIGHT eval -- mechanism-comparison view only. ~5 min vs full eval's ~22 min.
# Drops:
#   - MASAQ runs (cross-register stability is mechanism-orthogonal; only needed for ship candidate)
#   - constrained baselines (mechanism comparison runs heads-only)
#   - 4-stream + stress on constraints/MASAQ
# Keeps:
#   - Gazelle gate metrics (case + role-F1 + marker + fully) heads-only
#   - 4-stream + stress on Gazelle heads (rare-F1, head-F1, calibration gap)
#   - UD-PADT morph macro (verify Phase 1 morph heads still healthy)
#
# Usage: sbatch scripts/slurm/70b_eval_phase2_light.sbatch <runs/phase2_*_<JOBID>/final>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_DIR="${1:?usage: $0 <runs/phase2_*/final>}"
JOB_ID="$(basename "$(dirname "$MODEL_DIR")" | grep -oE '[0-9]+$' || echo unknown)"
VARIANT="$(basename "$(dirname "$MODEL_DIR")" | sed 's/_[0-9]*$//')"
OUT="runs/phase2_eval_${VARIANT}_${JOB_ID}"
mkdir -p "$OUT/gazelle" "$OUT/morph"

echo "=== Phase 2 LIGHTWEIGHT eval: $VARIANT (job $JOB_ID) ==="

echo
echo "--- Gazelle gate (heads only) ---"
python -u -m irab_tashkeel.evaluation.run_baselines \
    --eval gazelle \
    --baselines structured_v1 \
    --structured_v1_path "$MODEL_DIR" \
    --out "$OUT/gazelle"

echo
echo "--- Gazelle 4-stream + stress (heads only) ---"
python -u scripts/structured/eval_phase4a.py \
    --model "$MODEL_DIR" \
    --eval gazelle \
    --out_dir "$OUT/gazelle"

echo
echo "--- UD-PADT morphology ---"
python -u scripts/morphology/eval_morphology.py \
    --model "$MODEL_DIR" \
    --conllu data/ud_padt/ar_padt-ud-test.conllu \
    --out_dir "$OUT/morph"

echo
echo "=== Phase 2 lightweight eval done ($VARIANT) ==="
ls -la "$OUT"/gazelle/summary.json "$OUT"/gazelle/phase4a_summary.json "$OUT"/morph/morphology_summary.json 2>/dev/null
