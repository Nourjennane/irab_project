"""FastAPI demo backend for the validated nextgen iʿrāb model.

Endpoints:

  POST /api/analyze         — full per-token analysis of an Arabic sentence
  POST /api/compare         — Phase 3-A vs stage_7 side-by-side
  GET  /api/eval_metrics    — serve final_eval_tables.json (Phase A output)
  GET  /api/leakage_summary — serve leakage_summary.json (Phase B output)
  GET  /api/sample          — sample sentences (MSA / Quranic / nested / ambiguous)
  GET  /                    — single-page frontend
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from .inference import (
    SAMPLE_SENTENCES, ModelHolder, analyze_sentence, compare_sentence,
)


app = FastAPI(
    title="Arabic Iʿrāb — Validated Nextgen Demo",
    version="2.1.0",
    description="Per-token grammatical analysis with construction-aware reasoning.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = ROOT / "demo" / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Lazy model holders — instantiated on first request.
holder = ModelHolder()


class AnalyzeRequest(BaseModel):
    text: str
    checkpoint: str = "stage7"   # "stage7" | "phase3a"


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
    p = ROOT / "docs" / "final_eval" / "final_eval_tables.json"
    if not p.exists():
        return {"status": "not yet generated — run Phase A eval first"}
    return json.loads(p.read_text())


@app.get("/api/leakage_summary")
def leakage_summary() -> Dict[str, Any]:
    p = ROOT / "docs" / "leakage_audit" / "leakage_summary.json"
    if not p.exists():
        return {"status": "not yet generated"}
    return json.loads(p.read_text())


@app.get("/api/sample")
def sample() -> List[Dict[str, str]]:
    return SAMPLE_SENTENCES


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "checkpoints_available": holder.available_checkpoints(),
    }
