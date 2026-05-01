#!/bin/bash
#SBATCH --job-name=irab-arat5-smoke
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3415496/irab_project/logs/smoke_arat5_irab_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/smoke_arat5_irab_%j.err

# Phase 2.1 SMOKE — AraT5v2-base full-irab FT, 1000 examples, 1 epoch.
# Goal: verify training pipeline + checkpoint save/load on the stud-MIG slice
# before committing to the multi-hour full run. Per the directive's Phase 1.5
# rule: smoke-test before any expensive training.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

echo "=== env check ==="
python -c "import torch, transformers, datasets, accelerate; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'transformers', transformers.__version__)"
nvidia-smi -L
echo

if [ ! -s data/distill_v2/word_level.jsonl ]; then
    echo "ERROR: data/distill_v2/word_level.jsonl missing"
    exit 3
fi

OUT=runs/irab_arat5_smoke_${SLURM_JOB_ID}
python -u -m irab_tashkeel.training.llm.irab_arat5_sft \
    --config configs/irab_arat5v2_distill_v2.yaml \
    --output "$OUT" \
    --limit_train 1000 \
    --max_steps 100

echo "=== smoke done. final model in $OUT/final ==="
ls -la "$OUT" | head -20
