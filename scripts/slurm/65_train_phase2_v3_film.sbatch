#!/bin/bash
#SBATCH --job-name=irab-p2-v3-film
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/home/3415496/irab_project/logs/train_phase2_v3_film_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_phase2_v3_film_%j.err

# Phase 2 Run 1 — v3 + FiLM + soft + joint. THE GATE RUN.
# Builds on Phase 1 (rev 2 + 7 morph heads) by adding identity-init FiLM
# conditioning on the iʿrāb decoders. Decision gate: case ≥ 53.0,
# role-F1 ≥ 43.0, fully ≥ 19.4. If this run regresses Phase 1 we abort
# the rest of the Phase 2 grid.
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

OUT=runs/phase2_v3_film_${SLURM_JOB_ID}
mkdir -p "$OUT"

echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)" > "$OUT/provenance.txt"
echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)" >> "$OUT/provenance.txt"
echo "seed=42" >> "$OUT/provenance.txt"
echo "config=configs/phase2_v3_film_joint.yaml" >> "$OUT/provenance.txt"
echo "submitted=$(date -Iseconds)" >> "$OUT/provenance.txt"
cp configs/phase2_v3_film_joint.yaml "$OUT/config_snapshot.yaml"

python -u -m irab_tashkeel.training.structured.train \
    --config configs/phase2_v3_film_joint.yaml \
    --output_dir "$OUT"

echo "=== Phase 2 v3 + FiLM (Run 1, gate) done. final model in $OUT/final ==="
ls -la "$OUT/final" | head -15
du -sh "$OUT" 2>/dev/null || true
