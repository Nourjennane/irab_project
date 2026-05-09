"""Persistent annotation review queue.

Backed by a JSONL file under ``data_v2/ambiguity_corpus/<kind>/``,
plus parallel ``confirmed.jsonl``, ``rejected.jsonl``,
``edited.jsonl`` for the post-annotation states.

Single-writer model — the annotation server holds an in-memory copy
and rewrites the JSONL on each accept/reject/edit. Concurrent
annotators should be coordinated externally or via a small file lock.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..ambiguity.schema import AmbiguityExample


@dataclass
class QueueItem:
    example: AmbiguityExample
    state: str = "pending"      # pending | confirmed | edited | rejected
    annotator_id: str = ""
    notes: str = ""


class ReviewQueue:
    """In-memory queue with JSONL backing files."""

    def __init__(self, root: Path, kind: str):
        self.root = Path(root) / kind
        self.root.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        self.items: List[QueueItem] = []
        self._load()

    def _load(self) -> None:
        # Initial source: the auto-mined candidates
        queue_file = self.root / "queue.jsonl"
        if queue_file.exists():
            for line in queue_file.open():
                ex = AmbiguityExample.from_dict(json.loads(line))
                self.items.append(QueueItem(example=ex))
        # Overlay confirmed / rejected / edited states
        for fname, state in (
            ("confirmed.jsonl", "confirmed"),
            ("rejected.jsonl",  "rejected"),
            ("edited.jsonl",    "edited"),
        ):
            f = self.root / fname
            if not f.exists():
                continue
            for line in f.open():
                d = json.loads(line)
                aid = d.get("ambiguity_id")
                for it in self.items:
                    if it.example.ambiguity_id == aid:
                        it.state = state
                        it.annotator_id = d.get("annotator_id", "")
                        it.notes = d.get("notes", "")
                        if state == "edited":
                            it.example = AmbiguityExample.from_dict(d.get("example", {}))
                        break

    def pending(self, limit: int = 50) -> List[QueueItem]:
        return [it for it in self.items if it.state == "pending"][:limit]

    def confirm(self, ambiguity_id: str, *, annotator_id: str, notes: str = "") -> bool:
        for it in self.items:
            if it.example.ambiguity_id != ambiguity_id:
                continue
            it.state = "confirmed"
            it.annotator_id = annotator_id
            it.notes = notes
            self._append("confirmed.jsonl", {
                "ambiguity_id": ambiguity_id,
                "annotator_id": annotator_id,
                "notes": notes,
            })
            return True
        return False

    def reject(self, ambiguity_id: str, *, annotator_id: str,
               reason: str = "") -> bool:
        for it in self.items:
            if it.example.ambiguity_id != ambiguity_id:
                continue
            it.state = "rejected"
            it.annotator_id = annotator_id
            it.notes = reason
            self._append("rejected.jsonl", {
                "ambiguity_id": ambiguity_id,
                "annotator_id": annotator_id,
                "notes": reason,
            })
            return True
        return False

    def edit(self, ambiguity_id: str, edited: AmbiguityExample,
             *, annotator_id: str) -> bool:
        for it in self.items:
            if it.example.ambiguity_id != ambiguity_id:
                continue
            it.example = edited
            it.state = "edited"
            it.annotator_id = annotator_id
            self._append("edited.jsonl", {
                "ambiguity_id": ambiguity_id,
                "annotator_id": annotator_id,
                "example": edited.to_dict(),
            })
            return True
        return False

    def _append(self, fname: str, record: Dict) -> None:
        with (self.root / fname).open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def stats(self) -> Dict[str, int]:
        c = {"pending": 0, "confirmed": 0, "edited": 0, "rejected": 0}
        for it in self.items:
            c[it.state] = c.get(it.state, 0) + 1
        return c
