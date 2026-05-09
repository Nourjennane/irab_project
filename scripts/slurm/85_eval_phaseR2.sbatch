#!/bin/bash
#SBATCH --job-name=irab-pR2-eval
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/home/3415496/irab_project/logs/eval_phaseR2_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/eval_phaseR2_%j.err

# Phase R2 — retrieval-guided structural reasoning evaluation grid
#
# R2-0: sanity (no reasoners enabled — must reproduce Phase 3-A baseline)
# R2-A: KanaReasoner only on Gazelle + MASAQ
# R2-B: Kana + Istithna
# R2-C: All 5 reasoners (full system)
#
# tau_high=0.75, tau_med=0.50, lambda_strong=1.5 (defaults from design doc)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/phaseR2

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
    local outdir="runs/phaseR2/${tag}/${eval_set}"
    echo
    echo "=================================================="
    echo "  ${tag} — ${eval_set}"
    echo "=================================================="
    PYTHONPATH=src python -u scripts/structured/eval_per_construction.py \
        --model "$MODEL" --eval "$eval_set" \
        --out_dir "$outdir" \
        ${extra_args}
}

# R2-0: structural reasoning DISABLED (just runs Phase 3-A baseline as a control)
run_eval "R2-0" gazelle ""
run_eval "R2-0" masaq ""

# R2-A: KanaReasoner only
run_eval "R2-A" gazelle "--use_structural_reasoning --retrieval_memory $MEMORY --enabled_families kana_sisters --dump_traces"
run_eval "R2-A" masaq   "--use_structural_reasoning --retrieval_memory $MEMORY --enabled_families kana_sisters"

# R2-B: Kana + Istithna
run_eval "R2-B" gazelle "--use_structural_reasoning --retrieval_memory $MEMORY --enabled_families kana_sisters,istithna --dump_traces"
run_eval "R2-B" masaq   "--use_structural_reasoning --retrieval_memory $MEMORY --enabled_families kana_sisters,istithna"

# R2-C: all 5 reasoners (default — no enabled_families flag)
run_eval "R2-C" gazelle "--use_structural_reasoning --retrieval_memory $MEMORY --dump_traces"
run_eval "R2-C" masaq   "--use_structural_reasoning --retrieval_memory $MEMORY"

echo
echo "=== Phase R2 eval complete ==="
ls -la runs/phaseR2/
