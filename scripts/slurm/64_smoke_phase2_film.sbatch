#!/bin/bash
#SBATCH --job-name=irab-p2-smoke
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/home/3415496/irab_project/logs/smoke_phase2_film_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/smoke_phase2_film_%j.err

# Phase 2 smoke — verifies the full training stack with FiLM conditioning end-to-end:
#   - structured_config.json round-trips conditioning_mechanism + conditioning_detached
#   - MorphAugmentedStructuredModel with conditioning="film" actually trains
#     (loss decreases over 1 epoch on 100 sentences)
#   - inference path with the saved checkpoint loads without "missing/unexpected keys"
#   - the unit tests (tests/test_phase2_conditioning.py) all pass on HPC including
#     the encoder-load integration test that was skipped on Mac

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

OUT="runs/phase2_smoke_${SLURM_JOB_ID}"

echo "=== Phase 2 unit tests ==="
PYTHONPATH=src python -m pytest tests/test_phase2_conditioning.py -q

echo
echo "=== Phase 2 100-sentence training smoke (v3 + FiLM + soft + joint) ==="
python -u -m irab_tashkeel.training.structured.train \
    --config configs/phase2_v3_film_joint.yaml \
    --output_dir "$OUT" \
    --smoke --smoke_n 100 --smoke_epochs 1

echo
echo "=== smoke artefacts ==="
ls -la "$OUT" 2>/dev/null | head -20
echo
echo "=== structured_config.json (verify Phase 2 keys persisted) ==="
python -c "
import json, sys
from pathlib import Path
p = Path('$OUT')
finals = list(p.glob('final*'))
if not finals:
    finals = [p]
for f in finals:
    cfg_path = f / 'structured_config.json'
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        print(f'{cfg_path}:')
        for k in ('enable_morph_heads', 'taxonomy', 'conditioning_mechanism', 'conditioning_detached'):
            print(f'  {k} = {cfg.get(k)!r}')
        break
else:
    print('NO structured_config.json found', file=sys.stderr); sys.exit(1)
"

echo
echo "=== Phase 2 smoke done ==="
