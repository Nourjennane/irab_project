"""Build the canonical data_v2/manifests/provenance.json.

Walks ``data_v2/annotated/<source>/all.jsonl`` for each known source,
computes sha256 + line count, and writes the manifest with declared
split_role.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from irab_tashkeel.data_v2.provenance import (
    ProvenanceManifest, SourceProvenance,
)


# Hard-coded split assignments — anyone who edits these is changing
# the project's evaluation contract and must do so explicitly.
SPLIT_ROLES = {
    "distill_v2":      "train",
    "ud_padt_train":   "train",
    "ud_padt_dev":     "dev",
    "ud_padt_test":    "test",
    "gazelle_test":    "test",
    "masaq_quranic":   "test",
}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(ROOT / "data_v2" / "annotated"))
    ap.add_argument("--out", default=str(ROOT / "data_v2" / "manifests" / "provenance.json"))
    args = ap.parse_args()

    manifest = ProvenanceManifest()
    today = date.today().isoformat()

    for name, role in SPLIT_ROLES.items():
        p = Path(args.data_root) / name / "all.jsonl"
        if not p.exists():
            print(f"  [warn] missing {p} — skipping")
            continue
        n = sum(1 for _ in p.open())
        sp = SourceProvenance(
            name=name, split_role=role,
            provenance_id=f"{name}:v1",
            n_sentences=n,
            sha256=_sha256(p),
            date_ingested=today,
        )
        manifest.add(sp)
        print(f"  {name:20} role={role:5} n={n:6} sha256={sp.sha256[:12]}...")

    manifest.save(Path(args.out))
    print(f"\n✓ wrote {args.out}")


if __name__ == "__main__":
    main()
