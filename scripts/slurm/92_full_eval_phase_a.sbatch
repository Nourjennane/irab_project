#!/bin/bash
#SBATCH --job-name=irab-eval-phasea
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/home/3415496/irab_project/logs/full_eval_phasea_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/full_eval_phasea_%j.err

# Phase A — Full Independent Evaluation.
# Runs Phase 3-A baseline AND nextgen stage_7 over each held-out
# test set with the same evaluator. Writes per-(model × dataset)
# JSON shards, then aggregates into the final report.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs docs/final_eval/raw

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTHONPATH=src python -u scripts/eval/run_full_eval_v2.py \
    --checkpoints \
        phase3a:runs/phase3a_491240/final \
        stage7:runs/nextgen/stage_7/final \
    --datasets \
        gazelle:data_v2/annotated/gazelle_test/all.jsonl \
        masaq:data_v2/annotated/masaq_quranic/all.jsonl \
        ud_test:data_v2/annotated/ud_padt_test/all.jsonl \
    --output_root docs/final_eval/raw \
    --batch_size 16 \
    --seed 0

PYTHONPATH=src python -u scripts/eval/aggregate_full_eval.py \
    --raw_dir docs/final_eval/raw \
    --out_dir docs/final_eval \
    --baseline phase3a \
    --candidate stage7

echo "=== Phase A full eval complete ==="
ls -la docs/final_eval/ docs/final_eval/raw/
