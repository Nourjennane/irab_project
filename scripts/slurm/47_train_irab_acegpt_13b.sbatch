#!/bin/bash
#SBATCH --job-name=irab-acegpt13b
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/home/3415496/irab_project/logs/train_acegpt13b_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/train_acegpt13b_%j.err

# Phase 2.4 ext — AceGPT-13B (FreedomIntelligence/AceGPT-13B) QLoRA SFT.
# Substituted for Jais-13B (all 4 Jais-13B repos gated 403). ONE attempt
# with hard 24h budget per directive — no retries.
#
# Pre-flight: model weights MUST be downloaded to ~/.cache/huggingface/hub
# on a login node before this job runs (compute nodes have no internet).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

echo "=== env check ==="
python -c "import torch, transformers, datasets, accelerate, peft, bitsandbytes, irab_tashkeel; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'peft', peft.__version__, 'transformers', transformers.__version__, \
'bnb', bitsandbytes.__version__)"
nvidia-smi -L
echo

# Model loaded from local flat dir (BeeGFS scratch is unwritable from this user
# and hf_hub's POSIX file locks fail there; downloaded via wget into /home).
# Force offline mode so HF doesn't try to phone home from compute node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT=runs/irab_acegpt13b_distill_v2_${SLURM_JOB_ID}
python -u -m irab_tashkeel.training.llm.irab_jais_sft \
    --config configs/irab_acegpt_13b.yaml \
    --output "$OUT"

echo "=== training done. final adapter in $OUT/final ==="
ls -la "$OUT" | head
du -sh "$OUT"
