#!/bin/bash
#SBATCH --job-name=irab-pR2v2
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/home/3415496/irab_project/logs/eval_phaseR2v2_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/eval_phaseR2v2_%j.err

# Phase R2-v2 — KanaReasoner with fixes 1-4 (formula bug, conf guard, POS-aware skip, canonical case)
# Compare against R2-A baseline (kana-only, no fixes).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/phaseR2v2

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL=runs/phase3a_491240/final
MEMORY=data/grammar_memory/

run_eval() {
    local tag="$1"; shift
    local eval_set="$1"; shift
    local extra_args="$*"
    local outdir="runs/phaseR2v2/${tag}/${eval_set}"
    echo
    echo "=================================================="
    echo "  ${tag} — ${eval_set}"
    echo "=================================================="
    PYTHONPATH=src python -u scripts/structured/eval_per_construction.py \
        --model "$MODEL" --eval "$eval_set" \
        --out_dir "$outdir" \
        ${extra_args}
}

# v2-A: KanaReasoner only with all fixes (canonical case + per-field skip + log-odds)
run_eval "v2-A" gazelle "--use_structural_reasoning --retrieval_memory $MEMORY --enabled_families kana_sisters --dump_traces"
run_eval "v2-A" masaq   "--use_structural_reasoning --retrieval_memory $MEMORY --enabled_families kana_sisters"

# Re-run kana debug dump with v2 to compare per-particle outcomes
echo
echo "=== Re-running kana debug dump on v2 ==="
PYTHONPATH=src python -u scripts/structured/debug_kana_failures.py \
    --eval gazelle \
    --out_dir runs/phaseR2v2/debug_gazelle
PYTHONPATH=src python -u scripts/structured/debug_kana_failures.py \
    --eval masaq \
    --out_dir runs/phaseR2v2/debug_masaq

echo
echo "=== Phase R2-v2 eval complete ==="
ls -la runs/phaseR2v2/
