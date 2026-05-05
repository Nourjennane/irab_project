#!/bin/bash
#SBATCH --job-name=irab-ace13b-smoke
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/3415496/irab_project/logs/smoke_acegpt13b_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/smoke_acegpt13b_%j.err

# Phase 2.5 SMOKE — AceGPT-13B QLoRA, 1000 examples / 100 steps.
# Verifies: 4-bit nf4 loading, Llama-2 LoRA target_modules, gradient flow,
# tokenizer integration, save path. Burns ~30 min. Cheap insurance vs the
# 18+h full job hitting an unrecoverable error.

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

# Force offline mode so HF doesn't try to phone home from compute node
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT=runs/irab_acegpt13b_smoke_${SLURM_JOB_ID}
python -u -m irab_tashkeel.training.llm.irab_jais_sft \
    --config configs/irab_acegpt_13b.yaml \
    --output "$OUT" \
    --limit_train 1000 \
    --max_steps 100

echo "=== smoke done. final adapter in $OUT/final ==="
ls -la "$OUT" | head
du -sh "$OUT"
