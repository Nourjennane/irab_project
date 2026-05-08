#!/bin/bash
#SBATCH --job-name=irab-pR-build
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3415496/irab_project/logs/build_grammar_memory_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/build_grammar_memory_%j.err

# Phase R — build the grammar memory from Phase 3-A's training corpus.
# Output: data/grammar_memory/{family}/instances.jsonl + embeddings.faiss
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs data/grammar_memory

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# faiss-cpu must be available
python -c "import faiss; print('faiss', faiss.__version__)" || pip install faiss-cpu

PYTHONPATH=src python -u scripts/grammar_memory/build_memory.py \
    --model runs/phase3a_491240/final \
    --in_corpus data/morph_v1_dep/train.jsonl \
    --out_dir data/grammar_memory/

echo
echo "=== build done ==="
ls -la data/grammar_memory/
cat data/grammar_memory/_build_summary.json 2>/dev/null
