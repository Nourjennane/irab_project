#!/bin/bash
#SBATCH --job-name=irab-p3-parse
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3415496/irab_project/logs/parse_deps_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/parse_deps_%j.err

# Phase 3 — offline Stanza UD parse over distill_v2 records in morph_v1.
# Output: data/morph_v1_dep/{train,val}.jsonl with per-word dep info on
# distill_v2 records (UD-PADT records are passed through unchanged with
# has_dep=False).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs data/morph_v1_dep

module load miniconda3
module load cuda/12.4 || true
source activate irab

# Stanza models are cached at ~/.cache/stanza after the initial
# stanza.download('ar') run on the login node. Compute nodes have no
# internet, so the cache must be populated before this job runs.
if [ ! -d "$HOME/.cache/stanza/1.11.0/resources/ar/depparse" ]; then
    echo "ERROR: Stanza Arabic dep-parser not cached at ~/.cache/stanza/1.11.0/resources/ar"
    echo "Run on the login node first:"
    echo "  python -c 'import stanza; stanza.download(\"ar\")'"
    exit 4
fi

PYTHONPATH=src python -u scripts/morphology/parse_deps.py \
    --in_train data/morph_v1/train.jsonl \
    --in_val   data/morph_v1/val.jsonl \
    --out_dir  data/morph_v1_dep/ \
    --use_gpu

echo
echo "=== Phase 3 dep parse done ==="
ls -la data/morph_v1_dep/ 2>/dev/null | head
cat data/morph_v1_dep/parse_summary.json 2>/dev/null
