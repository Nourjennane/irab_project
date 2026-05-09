# Single-Command Reproduction Recipe

This file documents the minimum sequence to reproduce every
production result in this repository from a fresh clone.

## Prerequisites

```bash
git clone <repo-url> irab_project && cd irab_project
pip install -e ".[dev]"
# or for inference / demo only:
# pip install fastapi uvicorn torch transformers numpy
```

Python 3.11 + a modern PyTorch stack. No bespoke build tools needed.

## A. Build the schema_v2 corpus

(One-time. Reads raw sources + writes `data_v2/annotated/<source>/all.jsonl`.)

```bash
python scripts/data_v2/build_schema_v2_corpus.py
python scripts/data_v2/build_provenance_manifest.py
```

The provenance manifest enforces the train/dev/test split policy at
load time. Three runtime assertions in
`src/irab_tashkeel/curriculum/{config,sampler}.py` provide
defence-in-depth against the leakage class.

## B. Train Phase 3-A baseline (warm-start for everything)

(Skip if you can reuse our checkpoint at `runs/phase3a_491240/final/`.)

This is the **prior** production model. The recovery training warm-starts
from it.

```bash
# legacy script under scripts/training/ — see project history
```

## C. Train the validated recovery checkpoint (the production model)

```bash
PYTHONPATH=src python scripts/training_v2/train_curriculum.py \
    --output_root runs/nextgen_recovery \
    --warm_start  runs/phase3a_491240/final \
    --batch_size 16 --lr 1e-5 \
    --use_hard_failure_sampler \
    --label_smoothing 0.05 --entropy_reg_lambda 0.01 \
    --consistency_lambda 0.20 --fully_aux_lambda 0.50 \
    --use_ema --early_stop_patience 3 \
    --use_swa --swa_start_step 2000 \
    --use_llrd --llrd_decay 0.85
```

(SLURM: `sbatch scripts/slurm/93_train_curriculum_recovery.sbatch`)

Wall-clock on a single GPU: ~25–45 min.

## D. Freeze the validated checkpoint

```bash
python scripts/freeze_validated_checkpoint.py \
    --src runs/nextgen_recovery/stage_7/final \
    --dst runs/validated_nextgen_recovery \
    --skip_onnx
```

Writes `REPRODUCIBILITY_MANIFEST.json`, `model_torchscript.pt`,
`model_fp16.pt` into the validated directory.

## E. Run the independent full eval

```bash
PYTHONPATH=src python scripts/eval/run_full_eval_v2.py \
    --checkpoints \
        phase3a:runs/phase3a_491240/final \
        recovery:runs/validated_nextgen_recovery \
    --datasets \
        gazelle:data_v2/annotated/gazelle_test/all.jsonl \
        masaq:data_v2/annotated/masaq_quranic/all.jsonl \
    --output_root docs/final_eval_recovery/raw \
    --batch_size 16 --seed 0

PYTHONPATH=src python scripts/eval/aggregate_full_eval.py \
    --raw_dir docs/final_eval_recovery/raw \
    --out_dir docs/final_eval_recovery \
    --baseline phase3a --candidate recovery
```

(SLURM: `sbatch scripts/slurm/94_full_eval_recovery.sbatch`)

## F. Run the leakage audit

```bash
PYTHONPATH=src python scripts/eval/leakage_audit.py
```

Writes `docs/leakage_audit/{leakage_report.md, leakage_summary.json,
overlap_tables.csv, suspicious_examples.jsonl}`.

## G. Run the failure analysis (the central finding)

```bash
PYTHONPATH=src python scripts/analysis/run_failure_analysis.py \
    --checkpoint runs/validated_nextgen_recovery \
    --datasets gazelle_test masaq_quranic \
    --out_dir docs/failure_analysis
```

Writes the failure-analysis report set, including the `mudaaf_ilayh`
confusion family detail.

## H. Run the hard-eval per-bucket report

```bash
PYTHONPATH=src python scripts/data_v2/build_hard_eval.py
PYTHONPATH=src python scripts/analysis/run_hard_eval_report.py \
    --checkpoint runs/validated_nextgen_recovery \
    --hard_root data_v2/hard_eval \
    --out_dir docs/hard_eval
```

## I. Build the ambiguity-corpus mining queue

```bash
PYTHONPATH=src python scripts/data_v2/mine_ambiguity_candidates.py
```

Writes `data_v2/ambiguity_corpus/<kind>/queue.jsonl` for the seven
ambiguity kinds. (No model needed — pure dataset analysis from the
failure-analysis JSON output.)

## J. Reproduce the negative results (optional)

If you also want to reproduce the documented negative results:

```bash
# Graph integration:
sbatch scripts/slurm/95_train_graph.sbatch
sbatch scripts/slurm/96_full_eval_graph.sbatch

# Governor head:
sbatch scripts/slurm/98_train_governor.sbatch
sbatch scripts/slurm/99_full_eval_governor.sbatch
```

Both write to dedicated output directories so they do not overwrite
the production checkpoint.

## K. Launch the demo

```bash
pip install fastapi uvicorn
PYTHONPATH=src uvicorn demo.backend.main:app --port 8000
# open http://localhost:8000
```

## L. Launch the annotation server (when annotators are ready)

```bash
PYTHONPATH=src uvicorn irab_tashkeel.annotation.annotation_server:app --port 8001
# open http://localhost:8001
```

## Reproducibility manifests

Every frozen artifact carries:

- `REPRODUCIBILITY_MANIFEST.json` — git commit hash, env versions,
  dataset sha256s at training time
- `metrics.json` / `eval_tables.json` — full Phase A eval slice
- `calibration.json` — per-field reliability bins + ECE
- `training_manifest.json` — training summary + config
- `git_commit.txt`, `environment.txt`

To verify any artifact: read the JSON, compare the commit hash and
dataset sha256s against your environment.
