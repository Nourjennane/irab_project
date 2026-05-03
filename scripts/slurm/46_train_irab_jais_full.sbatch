#!/bin/bash
#SBATCH --job-name=irab-jais-full
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/home/3415496/irab_project/logs/train_jais_full_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_jais_full_%j.err

# Phase 2.4 FULL — Jais-family-1.3B LoRA SFT, 77,534 examples, 3 epochs.
# Submitted only after the smoke (487433) passes.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

echo "=== env check ==="
python -c "import torch, transformers, datasets, accelerate, peft, irab_tashkeel; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'peft', peft.__version__, 'transformers', transformers.__version__)"
nvidia-smi -L
echo

OUT=runs/irab_jais_distill_v2_${SLURM_JOB_ID}
python -u -m irab_tashkeel.training.llm.irab_jais_sft \
    --config configs/irab_jais_1p3b.yaml \
    --output "$OUT"

echo "=== full done. final adapter in $OUT/final ==="
ls -la "$OUT" | head
du -sh "$OUT"
