#!/bin/bash
#SBATCH --job-name=irab-p4a
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/3415496/irab_project/logs/train_phase4a_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_phase4a_%j.err

# Phase 4a — 34-label v4 taxonomy. One sbatch handles both arms;
# CONFIG argument selects the variant:
#   - configs/phase4a_taxonomy_no_morph.yaml  (granularity-only)
#   - configs/phase4a_taxonomy_full.yaml      (granularity + morph)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CONFIG="${1:-}"
if [ -z "$CONFIG" ]; then
    echo "ERROR: pass config path, e.g. sbatch ... configs/phase4a_taxonomy_no_morph.yaml"
    exit 2
fi
if [ ! -s "$CONFIG" ]; then
    echo "ERROR: config $CONFIG missing"
    exit 3
fi

# Variant tag for the run dir name
VARIANT="$(basename "$CONFIG" .yaml | sed 's/^phase4a_//')"
OUT="runs/phase4a_${VARIANT}_${SLURM_JOB_ID}"
mkdir -p "$OUT"

# Provenance
echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)" > "$OUT/provenance.txt"
echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)" >> "$OUT/provenance.txt"
echo "config=$CONFIG" >> "$OUT/provenance.txt"
echo "seed=42" >> "$OUT/provenance.txt"
echo "variant=$VARIANT" >> "$OUT/provenance.txt"
echo "submitted=$(date -Iseconds)" >> "$OUT/provenance.txt"
cp "$CONFIG" "$OUT/config_snapshot.yaml"

echo "=== env check ==="
python -c "import torch, transformers, irab_tashkeel; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'transformers', transformers.__version__)"
nvidia-smi -L
echo

python -u -m irab_tashkeel.training.structured.train \
    --config "$CONFIG" \
    --output_dir "$OUT"

echo "=== Phase 4a ($VARIANT) train done. final model in $OUT/final ==="
ls -la "$OUT/final" | head -15
du -sh "$OUT" 2>/dev/null || true
