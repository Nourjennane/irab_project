#!/bin/bash
#SBATCH --job-name=irab-p3-a
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/3415496/irab_project/logs/train_phase3a_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_phase3a_%j.err

# Phase 3 Run 1 — gate run. Phase 1 (rev 2 + 7 morph heads) + dep features
# (Stanza-parsed offline on distill_v2 records in morph_v1_dep).
# Decision gate: case >= 53.0, role-F1 >= 43.0, fully >= 19.4 vs Phase 1.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== env check ==="
python -c "import torch, transformers, irab_tashkeel; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'transformers', transformers.__version__)"
nvidia-smi -L
echo

if [ ! -s data/morph_v1_dep/train.jsonl ]; then
    echo "ERROR: data/morph_v1_dep/train.jsonl missing -- run scripts/slurm/71_parse_deps.sbatch first"
    exit 3
fi

OUT=runs/phase3a_${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)" > "$OUT/provenance.txt"
echo "config=configs/phase3_dep_aware.yaml" >> "$OUT/provenance.txt"
echo "submitted=$(date -Iseconds)" >> "$OUT/provenance.txt"
cp configs/phase3_dep_aware.yaml "$OUT/config_snapshot.yaml"

python -u -m irab_tashkeel.training.structured.train \
    --config configs/phase3_dep_aware.yaml \
    --output_dir "$OUT"

echo "=== Phase 3-A done. final model in $OUT/final ==="
ls -la "$OUT/final" | head -15
du -sh "$OUT" 2>/dev/null || true
