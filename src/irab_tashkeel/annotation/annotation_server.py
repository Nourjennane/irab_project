"""Lightweight FastAPI annotation server for the ambiguity queue.

Run::

    PYTHONPATH=src uvicorn irab_tashkeel.annotation.annotation_server:app \\
        --port 8001

Endpoints:

  GET  /api/queues                 — list all kinds + their counts
  GET  /api/queue/{kind}/pending   — fetch up to 50 pending items
  POST /api/queue/{kind}/confirm   — body: {ambiguity_id, annotator_id, notes}
  POST /api/queue/{kind}/reject    — body: {ambiguity_id, annotator_id, reason}
  POST /api/queue/{kind}/edit      — body: {ambiguity_id, annotator_id, example}
  GET  /api/queue/{kind}/disagreements — show items where annotators differ

The frontend (``static/annotation.html``) is a single-page UI: pick
a kind, see one item at a time, click "Confirm" / "Reject" / "Edit
analysis" / "Mark ambiguous". Annotator id is set once at top.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..ambiguity.schema import AmbiguityExample, AmbiguityKind
from .disagreement_resolution import collect_annotations, resolve_majority
from .review_queue import ReviewQueue


ROOT = Path(__file__).resolve().parents[3]
QUEUE_ROOT = ROOT / "data_v2" / "ambiguity_corpus"
STATIC_DIR = Path(__file__).resolve().parent / "static"


app = FastAPI(title="Iʿrāb Ambiguity Annotation", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


_queues: Dict[str, ReviewQueue] = {}


def _q(kind: str) -> ReviewQueue:
    if kind not in [k.value for k in AmbiguityKind]:
        raise HTTPException(404, f"unknown kind {kind}")
    if kind not in _queues:
        _queues[kind] = ReviewQueue(QUEUE_ROOT, kind)
    return _queues[kind]


class ConfirmReq(BaseModel):
    ambiguity_id: str
    annotator_id: str
    notes: str = ""


class RejectReq(BaseModel):
    ambiguity_id: str
    annotator_id: str
    reason: str = ""


class EditReq(BaseModel):
    ambiguity_id: str
    annotator_id: str
    example: Dict[str, Any]


@app.get("/api/queues")
def list_queues():
    out = {}
    for k in AmbiguityKind:
        try:
            q = _q(k.value)
            out[k.value] = q.stats()
        except Exception:
            out[k.value] = {"error": "no queue"}
    return out


@app.get("/api/queue/{kind}/pending")
def get_pending(kind: str, limit: int = 50):
    q = _q(kind)
    items = q.pending(limit=limit)
    return {"kind": kind, "n": len(items),
            "items": [{"state": it.state,
                       "example": it.example.to_dict()} for it in items]}


@app.post("/api/queue/{kind}/confirm")
def confirm(kind: str, req: ConfirmReq):
    q = _q(kind)
    ok = q.confirm(req.ambiguity_id, annotator_id=req.annotator_id, notes=req.notes)
    if not ok:
        raise HTTPException(404, "ambiguity_id not found")
    return {"ok": True, "stats": q.stats()}


@app.post("/api/queue/{kind}/reject")
def reject(kind: str, req: RejectReq):
    q = _q(kind)
    ok = q.reject(req.ambiguity_id, annotator_id=req.annotator_id, reason=req.reason)
    if not ok:
        raise HTTPException(404, "ambiguity_id not found")
    return {"ok": True, "stats": q.stats()}


@app.post("/api/queue/{kind}/edit")
def edit(kind: str, req: EditReq):
    q = _q(kind)
    edited = AmbiguityExample.from_dict(req.example)
    ok = q.edit(req.ambiguity_id, edited, annotator_id=req.annotator_id)
    if not ok:
        raise HTTPException(404, "ambiguity_id not found")
    return {"ok": True, "stats": q.stats()}


@app.get("/api/queue/{kind}/disagreements")
def disagreements(kind: str):
    by_id = collect_annotations(QUEUE_ROOT, kind)
    out = []
    for amb_id, anns in by_id.items():
        if len(anns) < 2:
            continue
        chosen, dis = resolve_majority(anns)
        out.append({
            "ambiguity_id": amb_id,
            "n_annotators": dis.n_annotators,
            "annotator_ids": dis.annotator_ids,
            "primary_signatures": dis.primary_signatures,
            "needs_escalation": dis.needs_escalation,
        })
    return {"kind": kind, "items": out}


@app.get("/")
def root():
    idx = STATIC_DIR / "annotation.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"status": "annotation server", "static_missing": True}


# Mount static directory if present
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
