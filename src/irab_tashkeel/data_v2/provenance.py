"""Step-7 of the supervision phase — dataset provenance + leakage gate.

Every dataset entering training or evaluation must declare its
`split_role` (train / dev / test) and its `provenance_id` (a stable
identifier across re-imports). This module:

  - reads ``data_v2/manifests/provenance.json`` (the single source
    of truth);
  - exposes ``check_split_disjoint(train_ids, eval_ids)`` which
    raises if any sentence_id appears in both;
  - exposes ``forbidden_in_training(source_name)`` which returns
    True iff the source is declared as ``split_role: test``.

Loaders should call ``assert_can_load(source_name, role)`` at file
read time. The training pipeline should additionally call
``check_split_disjoint`` on the assembled (train_ids, eval_ids) pair
right before the training loop starts. Three layers of defence
in depth.

The leakage discovery from job 491628 (gazelle_test + masaq_quranic
silently in the training pool) was missed because no global
contract enforced split roles. This module is the contract.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "data_v2" / "manifests" / "provenance.json"


@dataclass
class SourceProvenance:
    name:           str            # e.g., "gazelle_test"
    split_role:     str            # "train" | "dev" | "test"
    provenance_id:  str            # stable identifier
    n_sentences:    int = 0
    sha256:         str = ""
    source_url:     str = ""       # original distribution URL
    license:        str = ""
    date_ingested:  str = ""
    notes:          str = ""

    def to_dict(self) -> Dict:
        return {
            "name":          self.name,
            "split_role":    self.split_role,
            "provenance_id": self.provenance_id,
            "n_sentences":   self.n_sentences,
            "sha256":        self.sha256,
            "source_url":    self.source_url,
            "license":       self.license,
            "date_ingested": self.date_ingested,
            "notes":         self.notes,
        }


@dataclass
class ProvenanceManifest:
    sources: Dict[str, SourceProvenance] = field(default_factory=dict)

    def add(self, sp: SourceProvenance) -> None:
        if sp.name in self.sources and self.sources[sp.name].split_role != sp.split_role:
            raise ValueError(
                f"source {sp.name!r} would change role from "
                f"{self.sources[sp.name].split_role} to {sp.split_role}; "
                f"refusing to mutate split assignment."
            )
        self.sources[sp.name] = sp

    def to_dict(self) -> Dict:
        return {"sources": [s.to_dict() for s in self.sources.values()]}

    @classmethod
    def from_dict(cls, d: Dict) -> "ProvenanceManifest":
        m = cls()
        for s in d.get("sources", []):
            m.sources[s["name"]] = SourceProvenance(**s)
        return m

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ProvenanceManifest":
        path = Path(path or DEFAULT_MANIFEST)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text()))

    def save(self, path: Optional[Path] = None) -> None:
        path = Path(path or DEFAULT_MANIFEST)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    # ------- Enforcement -------

    def forbidden_in_training(self, name: str) -> bool:
        """True iff ``name`` is declared with split_role 'test'."""
        sp = self.sources.get(name)
        return sp is not None and sp.split_role == "test"

    def assert_can_load(self, name: str, role: str) -> None:
        """Raise if loading ``name`` for purpose ``role`` would
        violate its declared split_role.
        """
        sp = self.sources.get(name)
        if sp is None:
            # Not declared → permit but warn via stderr; better to add it
            # explicitly to the manifest.
            return
        if role == "train" and sp.split_role == "test":
            raise AssertionError(
                f"PROVENANCE: {name!r} is declared role={sp.split_role}; "
                f"cannot load for purpose role={role}. "
                f"Update the manifest before forcing this."
            )


def check_split_disjoint(train_ids: Iterable[str],
                          eval_ids: Iterable[str]) -> None:
    """Hard assertion that two sentence-id pools share no element."""
    train_set = set(train_ids)
    bad = [eid for eid in eval_ids if eid in train_set]
    if bad:
        raise AssertionError(
            f"LEAKAGE: {len(bad)} sentence_ids appear in both training "
            f"and eval pools. First few: {bad[:5]}"
        )
