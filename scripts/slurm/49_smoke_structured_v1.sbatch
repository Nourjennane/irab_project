#!/bin/bash
#SBATCH --job-name=irab-struc-smoke
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3415496/irab_project/logs/smoke_structured_v1_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/smoke_structured_v1_%j.err

# Phase 3 smoke — multi-head structured i'rāb classifier.
# 200 sentences, 1 epoch, no save. Verifies pipeline imports/forwards/losses.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

echo "=== env check ==="
python -c "import torch, transformers, irab_tashkeel; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'transformers', transformers.__version__)"
nvidia-smi -L
echo

if [ ! -s data/structured_v1/train.jsonl ]; then
    echo "ERROR: data/structured_v1/train.jsonl missing - did you run scripts/structured/build_structured_corpus.py ?"
    exit 3
fi

OUT=runs/structured_v1_smoke_${SLURM_JOB_ID}
python -u -m irab_tashkeel.training.structured.train \
    --config configs/structured_v1_rebuild.yaml \
    --output_dir "$OUT" \
    --smoke --smoke_n 200 --smoke_epochs 1

echo "=== smoke done. ==="
ls -la "$OUT" | head -20
du -sh "$OUT" 2>/dev/null || true
