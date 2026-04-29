#!/bin/bash
#SBATCH --job-name=irab-qlora-allam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=logs/qlora_allam_%j.out

# Stack A alternative — QLoRA SFT on ALLaM-7B-Instruct.
# 7B fits comfortably on A100 40GB. ~10h.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4
source activate irab

export HF_HOME=$HOME/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nvidia-smi

OUT=runs/qlora_allam_${SLURM_JOB_ID}
python -m irab_tashkeel.training.llm.qlora_sft \
    --config configs/llm_qlora_allam.yaml \
    --output "$OUT"

echo "✓ done. adapter in $OUT/adapter"
