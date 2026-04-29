#!/bin/bash
#SBATCH --job-name=irab-qlora-fanar
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/qlora_fanar_%j.out

# Stack A — QLoRA SFT on Fanar-1-9B-Instruct.
# Target: H100 (long_gpunew). ~14h. Falls back to A100 (long_gpu) if H100 queue is full.
# To use A100 instead, change --partition=long_gpu.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4
source activate irab

export HF_HOME=$HOME/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
# Reduce CPU thread oversubscription on big models
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nvidia-smi

OUT=runs/qlora_fanar_${SLURM_JOB_ID}
python -m irab_tashkeel.training.llm.qlora_sft \
    --config configs/llm_qlora_fanar.yaml \
    --output "$OUT"

echo "✓ done. adapter in $OUT/adapter"
