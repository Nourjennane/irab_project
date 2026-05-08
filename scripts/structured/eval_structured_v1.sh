#!/usr/bin/env bash
# Phase 3 / v1 rebuild evaluation runner.
#
# After the structured training job finishes (final/ in
# runs/structured_v1_rebuild_<JOBID>/), run this to score the rebuild on
# Gazelle and MASAQ in four configurations:
#
#   1. structured_v1               (heads only, no symbolic constraints)
#   2. structured_v1_constrained   (heads + 4 logit-bias rerankers)
#
# Usage:
#   bash scripts/structured/eval_structured_v1.sh path/to/runs/structured_v1_rebuild_<JOBID>/final
#
set -euo pipefail
MODEL_DIR="${1:?usage: $0 <model_final_dir>}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: $MODEL_DIR does not exist" >&2
    exit 2
fi

JOB_ID="$(basename "$(dirname "$MODEL_DIR")" | grep -oE '[0-9]+$' || echo unknown)"
OUT_BASE="runs/structured_v1_eval_${JOB_ID}"

mkdir -p "$OUT_BASE/gazelle" "$OUT_BASE/masaq"

echo "================================================"
echo "  Phase 3 / v1 rebuild eval"
echo "  Model: $MODEL_DIR"
echo "  Outputs: $OUT_BASE"
echo "================================================"

# ---- Gazelle (n=134), both ablations ----
echo ""
echo "--- Gazelle: structured_v1 + structured_v1_constrained ---"
python -u -m irab_tashkeel.evaluation.run_baselines \
    --eval gazelle \
    --baselines structured_v1,structured_v1_constrained \
    --structured_v1_path "$MODEL_DIR" \
    --out "$OUT_BASE/gazelle"

# ---- MASAQ (n=5,007 words), both ablations ----
echo ""
echo "--- MASAQ: structured_v1 + structured_v1_constrained ---"
python -u -m irab_tashkeel.evaluation.run_baselines \
    --eval masaq \
    --baselines structured_v1,structured_v1_constrained \
    --structured_v1_path "$MODEL_DIR" \
    --out "$OUT_BASE/masaq"

echo ""
echo "================================================"
echo "  Eval done. Summaries in:"
echo "    $OUT_BASE/gazelle/summary.json"
echo "    $OUT_BASE/masaq/summary.json"
echo "================================================"
