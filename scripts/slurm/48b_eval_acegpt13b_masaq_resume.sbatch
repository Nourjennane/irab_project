#!/bin/bash
#SBATCH --job-name=ace13b-masaq
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3415496/irab_project/logs/eval_acegpt13b_masaq_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/eval_acegpt13b_masaq_%j.err

# AceGPT-13B MASAQ eval — resume mode.
#
# stud QoS caps wall time at 4h, and a full MASAQ pass takes ~12-15h
# at ~9 sec/word with max_new_tokens=96. This sbatch is meant to be
# submitted multiple times in succession; each call picks up where
# the previous one stopped and appends to the existing predictions
# JSONL via the --resume flag in run_baselines.py.
#
# Run pattern:
#   sbatch 48b_eval_acegpt13b_masaq_resume.sbatch
#   # wait for it to finish (will hit 4h timeout for the first 2-3 jobs)
#   sbatch 48b_eval_acegpt13b_masaq_resume.sbatch
#   # repeat until the final scored count = 624 verses / 5007 words

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ADAPTER=runs/irab_acegpt13b_distill_v2_487888/final

echo "=== MASAQ eval (resume) — $(date) ==="
wc -l runs/baseline_eval_masaq_acegpt13b/acegpt_irab.predictions.jsonl 2>/dev/null \
    || echo "no existing predictions yet — starting from verse 1"

python -u -m irab_tashkeel.evaluation.run_baselines \
    --eval masaq --baselines acegpt_irab \
    --acegpt_irab_path "$ADAPTER" \
    --acegpt_base_model /home/3415496/acegpt13b \
    --out runs/baseline_eval_masaq_acegpt13b \
    --resume

echo
echo "=== JOB DONE — $(date) ==="
wc -l runs/baseline_eval_masaq_acegpt13b/acegpt_irab.predictions.jsonl
