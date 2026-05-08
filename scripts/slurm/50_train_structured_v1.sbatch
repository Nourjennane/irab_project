#!/bin/bash
#SBATCH --job-name=irab-struc-train
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/3415496/irab_project/logs/train_structured_v1_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_structured_v1_%j.err

# Phase 3 — multi-head structured i'rāb classifier on AraT5v2-base encoder.
# Full training: 4,750 sentences (77K word-judgments), 3 epochs.
# Estimated runtime: ~3-4h on stud MIG slice (encoder-only + 4 small heads,
# no decoder pass — substantially cheaper than the AraT5 seq2seq FT).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

# Compute nodes have no internet; the HF cache is already populated. Force
# offline mode to avoid hangs on huggingface.co lookups.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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

OUT=runs/structured_v1_rebuild_${SLURM_JOB_ID}
python -u -m irab_tashkeel.training.structured.train \
    --config configs/structured_v1_rebuild.yaml \
    --output_dir "$OUT"

echo "=== done. final model in $OUT/final ==="
ls -la "$OUT/final" | head -15
du -sh "$OUT" 2>/dev/null || true
