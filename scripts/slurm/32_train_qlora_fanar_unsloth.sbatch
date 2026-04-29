#!/bin/bash
#SBATCH --job-name=irab-unsloth
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/unsloth_fanar_%j.out

# Unsloth + Liger + sequence-packing QLoRA on Fanar-1-9B.
# ~3-4× faster than the vanilla qlora_sft path. Trains only on real data
# (Yarob + distilled), drops templated QAC/PADT.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4
source activate irab

# Install Unsloth + Liger if not present (idempotent).
python -c "import unsloth" 2>/dev/null || pip install --quiet \
    "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"
python -c "import liger_kernel" 2>/dev/null || pip install --quiet liger-kernel

export HF_HOME=$HOME/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nvidia-smi

OUT=runs/unsloth_fanar_${SLURM_JOB_ID}
python -m irab_tashkeel.training.llm.qlora_unsloth \
    --config configs/llm_qlora_fanar_unsloth.yaml \
    --output "$OUT"

echo "✓ done. adapter in $OUT/adapter"
