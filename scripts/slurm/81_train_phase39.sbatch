#!/bin/bash
#SBATCH --job-name=irab-p39
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/3415496/irab_project/logs/train_phase39_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_phase39_%j.err

# Phase #39 -- Phase 3-A architecture trained on augmented corpus
# (distill_v2 + 1,400 synthetic rare constructions). Same hyperparameters
# as phase3a_491240 (production). Decision gate: targeted constructions
# improve >= +5 pp word-fully on Gazelle subset; overall Gazelle within
# +/- 1 pp of 56.7 / 41.3 / 44.8 / 20.1.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [ ! -s data/morph_v1_augmented_dep/train.jsonl ]; then
    echo "ERROR: data/morph_v1_augmented_dep/train.jsonl missing -- run 80_parse_deps_augmented.sbatch first"
    exit 3
fi

OUT=runs/phase39_${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)" > "$OUT/provenance.txt"
echo "config=configs/phase39_augmented.yaml" >> "$OUT/provenance.txt"
cp configs/phase39_augmented.yaml "$OUT/config_snapshot.yaml"

python -u -m irab_tashkeel.training.structured.train \
    --config configs/phase39_augmented.yaml \
    --output_dir "$OUT"

echo "=== Phase #39 done ==="
ls -la "$OUT/final" | head -15
