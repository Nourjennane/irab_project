# Arabic Iʿrāb — Demo App

FastAPI backend serving the validated nextgen stage_7 model + a single-page
frontend with six tabs (Sentence Analysis, Grammar Graph, Reasoning Trace,
Construction Breakdown, Evaluation Dashboard, Model Comparison).

## Quick start

```bash
# Requires the validated checkpoint at runs/validated_nextgen_stage7/
# (run scripts/freeze_validated_checkpoint.py first if it doesn't exist;
# the demo also accepts runs/nextgen/stage_7/final/ if you symlink it).

pip install fastapi uvicorn
PYTHONPATH=src uvicorn demo.backend.main:app --reload --port 8000
# open http://localhost:8000
```

## Architecture

```
demo/
├── backend/
│   ├── main.py        FastAPI routes (/api/analyze, /api/compare, /api/eval_metrics, ...)
│   └── inference.py   ModelHolder (lazy-loads checkpoints), analyze_sentence
└── static/
    └── index.html     single-page frontend (Tailwind via CDN, vanilla JS)
```

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/analyze`         | `{text, checkpoint}`  | per-token analysis |
| POST | `/api/compare`         | `{text}`              | phase3a vs stage7 side-by-side |
| GET  | `/api/eval_metrics`    | —                     | Phase A eval tables |
| GET  | `/api/leakage_summary` | —                     | Phase B audit summary |
| GET  | `/api/sample`          | —                     | sample sentences (MSA / Quranic / nested / ambiguous) |
| GET  | `/api/health`          | —                     | available checkpoints |
| GET  | `/`                    | —                     | static frontend |

## Tabs

1. **Sentence Analysis** — per-token case / role / marker / POS with confidence bars and color-coded case pills
2. **Grammar Graph** — DOT-format dependency-style graph (rendered text; can be wired to viz-js)
3. **Reasoning Trace** — per-token narrative from structured labels (template-rendered, not generative)
4. **Construction Breakdown** — detected kana/inna/idafa families
5. **Evaluation Dashboard** — Phase A independent eval tables + Phase B leakage audit
6. **Model Comparison** — Phase 3-A vs stage_7 side-by-side on the same sentence

## Production notes

- The model is loaded lazily on first request to keep cold-start fast.
- For production, swap the FP32 checkpoint for the FP16 export
  (`runs/validated_nextgen_stage7/model_fp16.pt`) — halves memory and
  speeds up CPU inference noticeably.
- ONNX inference path is scaffolded but not wired into the backend by default;
  switch by setting `IRAB_USE_ONNX=1` (TODO).
- The grammar-graph viewer uses raw DOT for now; integrate viz-js or
  cytoscape.js for a polished visualization.
