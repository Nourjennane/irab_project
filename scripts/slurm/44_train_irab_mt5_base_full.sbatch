#!/bin/bash
#SBATCH --job-name=irab-mt5-full
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/3415496/irab_project/logs/train_mt5_base_full_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_mt5_base_full_%j.err

# Phase 2.3 FULL — mT5-base (~580M) full FT, 77,534 examples, 3 epochs.
# Smoke 487431 passed: train 12.71->9.75, eval 7.94, COMPLETED clean
# (disk fix: save_total_limit=1 + save_only_model=True).
# Extrapolated full-run wall: ~4-5h on A100 40GB MIG slice.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

echo "=== env check ==="
python -c "import torch, transformers, datasets, accelerate, irab_tashkeel; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'transformers', transformers.__version__)"
nvidia-smi -L
echo

OUT=runs/irab_mt5_base_distill_v2_${SLURM_JOB_ID}
python -u -m irab_tashkeel.training.llm.irab_arat5_sft \
    --config configs/irab_mt5_base.yaml \
    --output "$OUT"

echo "=== full done. final model in $OUT/final ==="
ls -la "$OUT" | head
du -sh "$OUT"
