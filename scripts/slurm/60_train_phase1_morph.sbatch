#!/bin/bash
#SBATCH --job-name=irab-phase1-morph
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/3415496/irab_project/logs/train_phase1_morph_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_phase1_morph_%j.err

# Phase 1 — multi-head structured i'rāb classifier + 7 auxiliary morphology heads.
# Joint masked multi-task training over UD-PADT (morph) + distill_v2 (i'rāb).
# Strict separation from rev 2: outputs go to runs/phase1_morph_<JOBID>/.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

# Compute nodes have no internet; HF cache is already populated.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== env check ==="
python -c "import torch, transformers, irab_tashkeel; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'transformers', transformers.__version__)"
nvidia-smi -L
echo

if [ ! -s data/morph_v1/train.jsonl ]; then
    echo "ERROR: data/morph_v1/train.jsonl missing — run scripts/morphology/build_morphology_corpus.sh first"
    exit 3
fi

OUT=runs/phase1_morph_${SLURM_JOB_ID}
mkdir -p "$OUT"

# Save provenance: git commit + config + seed + label set
echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)" > "$OUT/provenance.txt"
echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)" >> "$OUT/provenance.txt"
echo "seed=42" >> "$OUT/provenance.txt"
echo "config=configs/phase1_morphology.yaml" >> "$OUT/provenance.txt"
echo "submitted=$(date -Iseconds)" >> "$OUT/provenance.txt"
cp configs/phase1_morphology.yaml "$OUT/config_snapshot.yaml"

python -u -m irab_tashkeel.training.structured.train \
    --config configs/phase1_morphology.yaml \
    --output_dir "$OUT"

echo "=== Phase 1 train done. final model in $OUT/final ==="
ls -la "$OUT/final" | head -15
du -sh "$OUT" 2>/dev/null || true
