#!/bin/bash
#SBATCH --job-name=irab-ace13b-eval
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3415496/irab_project/logs/eval_acegpt13b_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/eval_acegpt13b_%j.err

# AceGPT-13B QLoRA eval on Gazelle + MASAQ subsets.
# Loads base in 4-bit NF4 (matching training), attaches PEFT adapter.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ADAPTER=runs/irab_acegpt13b_distill_v2_487888/final

echo "=== GAZELLE eval ==="
python -u -m irab_tashkeel.evaluation.run_baselines \
    --eval gazelle --baselines acegpt_irab \
    --acegpt_irab_path "$ADAPTER" \
    --acegpt_base_model /home/3415496/acegpt13b \
    --out runs/baseline_eval_acegpt13b_gazelle

echo
echo "=== MASAQ eval (5K words, ~80 min) ==="
python -u -m irab_tashkeel.evaluation.run_baselines \
    --eval masaq --baselines acegpt_irab \
    --acegpt_irab_path "$ADAPTER" \
    --acegpt_base_model /home/3415496/acegpt13b \
    --out runs/baseline_eval_masaq_acegpt13b

echo
echo "=== DONE ==="
ls -la runs/baseline_eval_acegpt13b_gazelle/ runs/baseline_eval_masaq_acegpt13b/
