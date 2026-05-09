#!/bin/bash
#SBATCH --job-name=irab-eval-gov
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3415496/irab_project/logs/full_eval_gov_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/full_eval_gov_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs docs/final_eval_governor/raw

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHONPATH=src python -u scripts/eval/run_full_eval_v2.py \
    --checkpoints \
        phase3a:runs/phase3a_491240/final \
        recovery:runs/validated_nextgen_recovery \
        governor:runs/nextgen_governor/stage_7/final \
    --datasets \
        gazelle:data_v2/annotated/gazelle_test/all.jsonl \
        masaq:data_v2/annotated/masaq_quranic/all.jsonl \
        ud_test:data_v2/annotated/ud_padt_test/all.jsonl \
    --output_root docs/final_eval_governor/raw \
    --batch_size 16 \
    --seed 0

PYTHONPATH=src python -u scripts/eval/aggregate_full_eval.py \
    --raw_dir docs/final_eval_governor/raw \
    --out_dir docs/final_eval_governor \
    --baseline phase3a \
    --candidate governor

# Run failure analysis on the governor model to compare confusion deltas
PYTHONPATH=src python -u scripts/analysis/run_failure_analysis.py \
    --checkpoint runs/nextgen_governor/stage_7/final \
    --out_dir docs/failure_analysis_governor \
    --datasets gazelle_test masaq_quranic

echo "=== governor full eval finished ==="
