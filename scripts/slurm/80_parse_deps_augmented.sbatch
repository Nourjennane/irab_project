#!/bin/bash
#SBATCH --job-name=irab-p39-parse
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3415496/irab_project/logs/parse_deps_augmented_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/parse_deps_augmented_%j.err

# Phase #39 -- Stanza-parse the augmented corpus (distill_v2 + 1,400 synthetic).
# Output: data/morph_v1_augmented_dep/{train,val}.jsonl
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs data/morph_v1_augmented_dep

module load miniconda3
module load cuda/12.4 || true
source activate irab

if [ ! -d "$HOME/.cache/stanza/1.11.0/resources/ar/depparse" ]; then
    echo "ERROR: Stanza Arabic dep-parser not cached"
    exit 4
fi

PYTHONPATH=src python -u scripts/morphology/parse_deps.py \
    --in_train data/morph_v1_augmented/train.jsonl \
    --in_val   data/morph_v1_augmented/val.jsonl \
    --out_dir  data/morph_v1_augmented_dep/ \
    --use_gpu

echo "=== Phase #39 dep parse done ==="
ls -la data/morph_v1_augmented_dep/ 2>/dev/null | head
cat data/morph_v1_augmented_dep/parse_summary.json 2>/dev/null
