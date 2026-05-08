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

# Stanza needs network the first time to download Arabic models. Once
# cached, we can run offline.
export STANZA_RESOURCES_DIR="${STANZA_RESOURCES_DIR:-$HOME/stanza_resources}"

# If models aren't cached, attempt online download (login node has
# internet); if we're on a compute node without network, skip — caller
# should re-run on the login node first.
python -c "import stanza, os; \
ar = os.path.join(os.environ.get('STANZA_RESOURCES_DIR', os.path.expanduser('~/stanza_resources')), 'ar'); \
import os; (os.path.isdir(ar) or stanza.download('ar', verbose=False))"

PYTHONPATH=src python -u scripts/morphology/parse_deps.py \
    --in_train data/morph_v1/train.jsonl \
    --in_val   data/morph_v1/val.jsonl \
    --out_dir  data/morph_v1_dep/ \
    --use_gpu

echo
echo "=== Phase 3 dep parse done ==="
ls -la data/morph_v1_dep/ 2>/dev/null | head
cat data/morph_v1_dep/parse_summary.json 2>/dev/null
