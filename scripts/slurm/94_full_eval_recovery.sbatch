#!/bin/bash
#SBATCH --job-name=irab-eval-recv
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3415496/irab_project/logs/full_eval_recovery_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/full_eval_recovery_%j.err

# Independent eval of the recovery checkpoint vs Phase 3-A baseline,
# on the FULL clean held-out sets (no caps).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs docs/final_eval_recovery/raw

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHONPATH=src python -u scripts/eval/run_full_eval_v2.py \
    --checkpoints \
        phase3a:runs/phase3a_491240/final \
        recovery:runs/nextgen_recovery/stage_7/final \
    --datasets \
        gazelle:data_v2/annotated/gazelle_test/all.jsonl \
        masaq:data_v2/annotated/masaq_quranic/all.jsonl \
        ud_test:data_v2/annotated/ud_padt_test/all.jsonl \
    --output_root docs/final_eval_recovery/raw \
    --batch_size 16 \
    --seed 0

PYTHONPATH=src python -u scripts/eval/aggregate_full_eval.py \
    --raw_dir docs/final_eval_recovery/raw \
    --out_dir docs/final_eval_recovery \
    --baseline phase3a \
    --candidate recovery

echo "=== recovery full eval finished ==="
ls -la docs/final_eval_recovery/
