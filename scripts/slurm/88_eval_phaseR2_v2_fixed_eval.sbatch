#!/bin/bash
#SBATCH --job-name=irab-pR2v2f
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/home/3415496/irab_project/logs/eval_phaseR2v2_fixed_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/eval_phaseR2v2_fixed_%j.err

# R2-v2.1 with FIXED evaluator (kana-aware role extraction).
# Compare Phase 3-A baseline AND R2-v2.1 on the corrected gold.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/phaseR2v2_fixed

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
    local outdir="runs/phaseR2v2_fixed/${tag}/${eval_set}"
    echo
    echo "=================================================="
    echo "  ${tag} — ${eval_set} (FIXED evaluator)"
    echo "=================================================="
    PYTHONPATH=src python -u scripts/structured/eval_per_construction.py \
        --model "$MODEL" --eval "$eval_set" \
        --out_dir "$outdir" \
        ${extra_args}
}

# Phase 3-A baseline on FIXED evaluator
run_eval "P3A_fixed" gazelle ""
run_eval "P3A_fixed" masaq ""

# R2-v2.1 on FIXED evaluator
run_eval "v2.1_fixed" gazelle "--use_structural_reasoning --retrieval_memory $MEMORY --enabled_families kana_sisters"
run_eval "v2.1_fixed" masaq   "--use_structural_reasoning --retrieval_memory $MEMORY --enabled_families kana_sisters"

echo
echo "=== R2-v2.1 with fixed evaluator complete ==="
ls -la runs/phaseR2v2_fixed/
