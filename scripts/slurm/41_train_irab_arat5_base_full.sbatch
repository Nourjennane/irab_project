#!/bin/bash
#SBATCH --job-name=irab-arat5-full
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/3415496/irab_project/logs/train_arat5_irab_full_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_arat5_irab_full_%j.err

# Phase 2.1 FULL — AraT5v2-base full-irab FT, 77,534 examples, 3 epochs.
# Smoke 487232 passed: train 9.66->3.80, eval 3.21->3.02, throughput 26.6/s
# on A100 40GB MIG slice. Extrapolated runtime: ~2.4h (well under 6h limit).

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

if [ ! -s data/distill_v2/word_level.jsonl ]; then
    echo "ERROR: data/distill_v2/word_level.jsonl missing"
    exit 3
fi

OUT=runs/irab_arat5v2_distill_v2_${SLURM_JOB_ID}
python -u -m irab_tashkeel.training.llm.irab_arat5_sft \
    --config configs/irab_arat5v2_distill_v2.yaml \
    --output "$OUT"

echo "=== full done. final model in $OUT/final ==="
ls -la "$OUT" | head -10
du -sh "$OUT"
