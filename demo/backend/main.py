"""FastAPI demo backend for the validated nextgen iʿrāb model.

Endpoints:

  POST /api/analyze         — full per-token analysis of an Arabic sentence
  POST /api/compare         — Phase 3-A vs recovery vs leaked stage_7
  GET  /api/eval_metrics    — Phase A eval tables
  GET  /api/leakage_summary — Phase B audit summary
  GET  /api/permissive_eval — eval_v3 permissive scoring summary
  GET  /api/calibration     — temperature scaling fits + reliability bins
  GET  /api/sample          — sample sentences (MSA / Quranic / Idafa / etc.)
  GET  /api/hard_cases      — curated hard examples gallery
  GET  /api/health          — checkpoints + project banner
  GET  /                    — single-page frontend
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from .inference import (
    HARD_CASES, SAMPLE_SENTENCES, ModelHolder,
    analyze_sentence, compare_sentence,
)


app = FastAPI(
    title="Arabic Iʿrāb — Honest Grammatical Reasoning",
    version="2.1.0",
    description="Per-token grammatical analysis with construction-aware reasoning.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = ROOT / "demo" / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

holder = ModelHolder()


class AnalyzeRequest(BaseModel):
    text: str
    checkpoint: str = "recovery"


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    if not req.text.strip():
        raise HTTPException(400, "text must be non-empty")
    try:
        return analyze_sentence(holder, req.text, checkpoint=req.checkpoint)
    except FileNotFoundError as e:
        raise HTTPException(503, f"checkpoint not available: {e}")


@app.post("/api/compare")
def compare(req: AnalyzeRequest) -> Dict[str, Any]:
    if not req.text.strip():
        raise HTTPException(400, "text must be non-empty")
    return compare_sentence(holder, req.text)


@app.get("/api/eval_metrics")
def eval_metrics() -> Dict[str, Any]:
    """Validated recovery's headline metrics + the negative-result deltas."""
    out: Dict[str, Any] = {"summary": {}}

    def _load(p: Path):
        return json.loads(p.read_text()) if p.exists() else None

    rec = _load(ROOT / "docs" / "final_eval_recovery" / "final_eval_tables.json")
    if rec:
        out["recovery_eval"] = rec

    # Curated headline (so the dashboard doesn't have to slice the raw)
    out["summary"]["headline"] = {
        "gazelle":  {"phase3a": {"case": 0.638, "role": 0.575, "marker": 0.684, "fully": 0.459},
                      "recovery": {"case": 0.646, "role": 0.613, "marker": 0.653, "fully": 0.459},
                      "graph":    {"case": 0.638, "role": 0.613, "marker": 0.653, "fully": 0.459},
                      "governor": {"case": 0.661, "role": 0.600, "marker": 0.684, "fully": 0.459}},
        "masaq":    {"phase3a": {"case": 0.835, "role": 0.778, "marker": 0.718, "fully": 0.675},
                      "recovery": {"case": 0.848, "role": 0.807, "marker": 0.710, "fully": 0.711},
                      "graph":    {"case": 0.845, "role": 0.813, "marker": 0.715, "fully": 0.707},
                      "governor": {"case": 0.844, "role": 0.805, "marker": 0.707, "fully": 0.714}},
    }
    out["summary"]["calibration_ece"] = {
        "recovery": {"case": 0.42, "role": 0.49, "marker": 0.60},
    }
    out["summary"]["leaked_stage7"] = {
        "masaq_fully": 0.999, "masaq_calib_gap": 0.9998,
        "gazelle_fully": 0.377,
        "note": "Contamination case study — gazelle_test + masaq_quranic in training pool",
    }
    return out


@app.get("/api/leakage_summary")
def leakage_summary() -> Dict[str, Any]:
    p = ROOT / "docs" / "leakage_audit" / "leakage_summary.json"
    if not p.exists():
        return {"status": "not yet generated"}
    return json.loads(p.read_text())


@app.get("/api/permissive_eval")
def permissive_eval() -> Dict[str, Any]:
    p = ROOT / "docs" / "permissive_eval" / "permissive_eval.json"
    if not p.exists():
        return {"status": "not yet generated; "
                "run scripts/analysis/auto_annotate_ambiguities.py"}
    return json.loads(p.read_text())


@app.get("/api/calibration")
def calibration() -> Dict[str, Any]:
    p = ROOT / "docs" / "calibration" / "temperature_fits.json"
    if not p.exists():
        return {"status": "not yet generated; "
                "run scripts/calibration/run_temperature_scaling.py"}
    return json.loads(p.read_text())


@app.get("/api/failure_summary")
def failure_summary() -> Dict[str, Any]:
    """Top role / case / marker confusions on the production model."""
    p = ROOT / "docs" / "failure_analysis" / "summary.json"
    if not p.exists():
        return {"status": "not yet generated"}
    d = json.loads(p.read_text())
    # Trim to top-20 confusions per axis
    confusions = {}
    for axis in ("case", "role", "marker"):
        top = d.get("confusions", {}).get(axis, {}).get("top", [])
        confusions[axis] = top[:20]
    return {
        "n_records":    d.get("n_failure_records"),
        "buckets":      d.get("buckets"),
        "confusions":   confusions,
        "calibration":  d.get("calibration"),
    }


@app.get("/api/sample")
def sample():
    return SAMPLE_SENTENCES


@app.get("/api/hard_cases")
def hard_cases():
    return HARD_CASES


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "2.1.0",
        "checkpoints_available": holder.available_checkpoints(),
        "project": "Arabic Iʿrāb — Honest Grammatical Reasoning",
        "production_checkpoint": "validated_nextgen_recovery",
    }
