#!/bin/bash
#SBATCH --job-name=irab-p2-eval
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/home/3415496/irab_project/logs/eval_phase2_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/eval_phase2_%j.err

# Phase 2 eval (any of the 5 mechanism/training-mode/taxonomy variants).
# For a given trained model:
#  1. Gazelle headline numbers via run_baselines (case, role-F1, marker, fully)
#     -- this is what feeds the decision gate.
#  2. MASAQ cross-register stability via run_baselines.
#  3. Stream A/B/C/D + stress table via eval_phase4a.py on Gazelle + MASAQ
#     -- captures rare-role, head-role, calibration, long-tail-collapse.
#  4. UD-PADT morphology accuracy -- verify Phase 1 morph heads still healthy.
#
# Usage: sbatch scripts/slurm/70_eval_phase2.sbatch <runs/phase2_*_<JOBID>/final>

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
mkdir -p "$OUT/gazelle" "$OUT/gazelle_constrained" "$OUT/masaq" "$OUT/morph"

echo "================================================"
echo "  Phase 2 eval"
echo "  variant: $VARIANT  job: $JOB_ID"
echo "  model:   $MODEL_DIR"
echo "  out:     $OUT"
echo "================================================"

echo
echo "--- 1. Gazelle headline (heads + heads+constraints) ---"
python -u -m irab_tashkeel.evaluation.run_baselines \
    --eval gazelle \
    --baselines structured_v1,structured_v1_constrained \
    --structured_v1_path "$MODEL_DIR" \
    --out "$OUT/gazelle"

echo
echo "--- 2. MASAQ cross-register ---"
python -u -m irab_tashkeel.evaluation.run_baselines \
    --eval masaq \
    --baselines structured_v1,structured_v1_constrained \
    --structured_v1_path "$MODEL_DIR" \
    --out "$OUT/masaq"

echo
echo "--- 3. 4-stream + stress on Gazelle (heads only) ---"
python -u scripts/structured/eval_phase4a.py \
    --model "$MODEL_DIR" \
    --eval gazelle \
    --out_dir "$OUT/gazelle"

echo
echo "--- 4. 4-stream + stress on Gazelle (with constraints) ---"
python -u scripts/structured/eval_phase4a.py \
    --model "$MODEL_DIR" \
    --eval gazelle \
    --apply_constraints \
    --out_dir "$OUT/gazelle_constrained"

echo
echo "--- 5. 4-stream + stress on MASAQ (heads only) ---"
python -u scripts/structured/eval_phase4a.py \
    --model "$MODEL_DIR" \
    --eval masaq \
    --out_dir "$OUT/masaq"

echo
echo "--- 6. UD-PADT morph accuracy ---"
python -u scripts/morphology/eval_morphology.py \
    --model "$MODEL_DIR" \
    --conllu data/ud_padt/ar_padt-ud-test.conllu \
    --out_dir "$OUT/morph"

echo
echo "================================================"
echo "  Phase 2 eval done."
echo "    Gazelle headline: $OUT/gazelle/summary.json"
echo "    MASAQ headline:   $OUT/masaq/summary.json"
echo "    4-stream + stress: $OUT/{gazelle,gazelle_constrained,masaq}/phase4a_summary.json"
echo "    Morphology:        $OUT/morph/morphology_summary.json"
echo "================================================"
ls -1 "$OUT"/*/summary.json "$OUT"/*/phase4a_summary.json 2>/dev/null
