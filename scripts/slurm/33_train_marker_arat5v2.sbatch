#!/bin/bash
#SBATCH --job-name=irab-marker
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/marker_arat5v2_%j.out

# Mix A Phase 2 — fine-tune AraT5v2-base on marker prediction.
# 8.8k pairs × 5 epochs at effective batch 16. Estimated ~2.5h on full A100,
# ~3-4h on stud MIG slice (4g.40gb).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4
source activate irab

# Compute nodes typically have no outbound internet, so we don't pip-install
# here. The training stack must be present in the env beforehand:
#   ssh login-node && module load miniconda3 && conda activate irab && \
#       pip install "transformers>=4.46,<4.55" accelerate datasets bitsandbytes sentencepiece anthropic
python -c "import transformers, accelerate, datasets, bitsandbytes" || {
    echo "ERROR: training stack missing from conda env 'irab' — run pip install on the login node first"
    exit 2
}

export HF_HOME=$HOME/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

# Sanity: marker_pairs.jsonl must exist (run evaluation.marker_extract first
# locally, then commit + pull, OR run the extractor here).
if [ ! -s data/marker_pairs.jsonl ]; then
    echo "data/marker_pairs.jsonl missing; building from distilled + yarob"
    python -m irab_tashkeel.evaluation.marker_extract \
        --distilled data/distilled_irab.jsonl \
        --yarob_dir data/yarob_src \
        --out data/marker_pairs.jsonl
fi

OUT=runs/marker_arat5v2_${SLURM_JOB_ID}
python -m irab_tashkeel.training.llm.marker_arat5_sft \
    --config configs/marker_arat5v2.yaml \
    --output "$OUT"

echo "✓ done. final marker model in $OUT/final"
