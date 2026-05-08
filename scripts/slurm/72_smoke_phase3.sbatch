#!/bin/bash
#SBATCH --job-name=irab-p3-smoke
#SBATCH --partition=stud
#SBATCH --account=3415496
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/home/3415496/irab_project/logs/smoke_phase3_%j.out
#SBATCH --error=/home/3415496/irab_project/logs/smoke_phase3_%j.err

# Phase 3 smoke — 100-sentence/1-epoch full training stack with dep features.
# Verifies:
#  - data/morph_v1_dep/{train,val}.jsonl loads with dep fields
#  - DepAwareStructuredModel forward works on a real batch
#  - structured_config.json round-trips enable_dep_features=true
#  - 14 unit tests in tests/test_phase3_dep_features.py pass on HPC
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs

module load miniconda3
module load cuda/12.4 || true
source activate irab

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

OUT="runs/phase3_smoke_${SLURM_JOB_ID}"

echo "=== Phase 3 unit tests ==="
PYTHONPATH=src python -m pytest tests/test_phase3_dep_features.py -q

echo
echo "=== Phase 3 100-sentence training smoke (Phase 1 + dep features) ==="
python -u -m irab_tashkeel.training.structured.train \
    --config configs/phase3_dep_aware.yaml \
    --output_dir "$OUT" \
    --smoke --smoke_n 100 --smoke_epochs 1

echo
echo "=== smoke artefacts ==="
ls -la "$OUT" 2>/dev/null | head -20
echo
echo "=== structured_config.json (verify Phase 3 keys persisted) ==="
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
        for k in ('enable_morph_heads', 'enable_dep_features', 'taxonomy', 'conditioning_mechanism'):
            print(f'  {k} = {cfg.get(k)!r}')
        break
else:
    print('NO structured_config.json found', file=sys.stderr); sys.exit(1)
"
echo
echo "=== Phase 3 smoke done ==="
