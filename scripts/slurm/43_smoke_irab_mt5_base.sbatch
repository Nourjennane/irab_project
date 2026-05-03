#!/bin/bash
#SBATCH --job-name=irab-mt5-smoke
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3415496/irab_project/logs/smoke_mt5_base_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/smoke_mt5_base_%j.err

# Phase 2.3 SMOKE — mT5-base (~580M) full FT, 1000 examples, 100 steps.

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

OUT=runs/irab_mt5_base_smoke_${SLURM_JOB_ID}
python -u -m irab_tashkeel.training.llm.irab_arat5_sft \
    --config configs/irab_mt5_base.yaml \
    --output "$OUT" \
    --limit_train 1000 --max_steps 100

echo "=== smoke done. final model in $OUT/final ==="
ls -la "$OUT" | head
du -sh "$OUT"
