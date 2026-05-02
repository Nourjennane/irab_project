#!/bin/bash
#SBATCH --job-name=irab-arat5-large-smoke
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/3415496/irab_project/logs/smoke_arat5_large_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/smoke_arat5_large_%j.err

# Phase 2.3 SMOKE — AraT5v2-large (1.2B) LoRA r=32, 1000 examples, 100 steps.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

echo "=== env check ==="
python -c "import torch, transformers, datasets, accelerate, peft, irab_tashkeel; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
'peft', peft.__version__, 'transformers', transformers.__version__)"
nvidia-smi -L
echo

OUT=runs/irab_arat5_large_smoke_${SLURM_JOB_ID}
python -u -m irab_tashkeel.training.llm.irab_arat5_sft \
    --config configs/irab_arat5v2_large_lora.yaml \
    --output "$OUT" \
    --limit_train 1000 --max_steps 100

echo "=== smoke done. final adapter in $OUT/final ==="
ls -la "$OUT" | head
du -sh "$OUT"
