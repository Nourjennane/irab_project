#!/bin/bash
#SBATCH --job-name=irab-marker-yarob
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/marker_yarob_%j.out

# Marker FT v2 — Yarob-only training to test the teacher-bias hypothesis.
# 1,643 hand-authored Yarob marker pairs, 10 epochs, otherwise same recipe.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4
source activate irab

python -c "import transformers, accelerate, datasets, bitsandbytes" || {
    echo "ERROR: training stack missing"; exit 2;
}

export HF_HOME=$HOME/.hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

nvidia-smi

# Sanity: marker_pairs_yarob_only.jsonl must exist locally and be pushed
if [ ! -s data/marker_pairs_yarob_only.jsonl ]; then
    echo "ERROR: data/marker_pairs_yarob_only.jsonl missing"
    exit 3
fi

OUT=runs/marker_yarob_${SLURM_JOB_ID}
python -m irab_tashkeel.training.llm.marker_arat5_sft \
    --config configs/marker_arat5v2_yarob.yaml \
    --output "$OUT"

echo "✓ done. final marker model in $OUT/final"
