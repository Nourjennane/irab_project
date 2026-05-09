#!/bin/bash
#SBATCH --job-name=irab-supan
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/home/3415496/irab_project/logs/supan_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/supan_%j.err

# Run failure analysis + per-bucket hard-eval against the validated
# recovery checkpoint.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHONPATH=src python -u scripts/analysis/run_failure_analysis.py \
    --checkpoint runs/validated_nextgen_recovery \
    --out_dir docs/failure_analysis \
    --datasets gazelle_test masaq_quranic

PYTHONPATH=src python -u scripts/analysis/run_hard_eval_report.py \
    --checkpoint runs/validated_nextgen_recovery \
    --hard_root data_v2/hard_eval \
    --out_dir docs/hard_eval

echo "=== supervision-phase analysis done ==="
ls -la docs/failure_analysis docs/hard_eval
