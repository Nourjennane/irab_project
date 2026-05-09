"""Build the canonical schema_v2 corpus from all available sources.

Runs the full data_v2 pipeline:

  source loaders → construction detector → curriculum metadata → JSONL

Per source we produce ``data_v2/annotated/<source>/all.jsonl`` (a
single canonical file). Train/dev/test splits are produced
separately by ``scripts/data_v2/split_corpus.py``.

The resulting JSONL is the canonical next-generation training
dataset; it preserves provenance, source quality, parser-vs-gold
distinction, alternative analyses, construction overlap, and all
graph-ready structural slots.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from irab_tashkeel.data_v2.constructions.detector import detect_constructions_pass
from irab_tashkeel.data_v2.loaders import distill2, gazelle, masaq, ud_padt  # register
from irab_tashkeel.data_v2.loaders.base import get_loader
from irab_tashkeel.data_v2.metadata import difficulty
from irab_tashkeel.data_v2.schema_v2 import write_jsonl


SOURCES = [
    ("distill_v2",       {"split": None}),    # frozen-baseline silver MSA
    ("gazelle_test",     {"split": None}),    # held-out gold MSA
    ("masaq_quranic",    {"split": None}),    # gold Quranic
    ("ud_padt",          {"split": "train"}), # gold treebank — train
    ("ud_padt",          {"split": "dev"}),
    ("ud_padt",          {"split": "test"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out_root", default=str(ROOT / "data_v2" / "annotated"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit per-source sentences (debug)")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="Subset of source ids to build")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "started": time.time(),
        "sources": {},
    }

    for source_id, kwargs in SOURCES:
        if args.sources and source_id not in args.sources:
            continue

        loader_cls = get_loader(source_id)
        try:
            if "split" in kwargs and kwargs["split"]:
                loader = loader_cls(root=args.root, split=kwargs["split"])
                tag = f"{source_id}_{kwargs['split']}"
            else:
                loader = loader_cls(root=args.root)
                tag = source_id
        except Exception as e:
            print(f"[skip] {source_id} ({kwargs}): {e}")
            continue

        print(f"\n=== {tag} ===")
        t0 = time.time()
        sentences = loader.load_all()
        n_loaded = len(sentences)
        print(f"  loaded {n_loaded} sentences in {time.time() - t0:.1f}s")
        if not n_loaded:
            print(f"  [skip] empty")
            continue

        if args.limit:
            sentences = sentences[: args.limit]
            print(f"  limited to {len(sentences)}")

        # Construction detection
        t0 = time.time()
        n_with = detect_constructions_pass(sentences)
        print(f"  constructions detected in {n_with}/{len(sentences)} "
              f"in {time.time() - t0:.1f}s")

        # Curriculum metadata
        t0 = time.time()
        difficulty.populate_all(sentences)
        print(f"  metadata populated in {time.time() - t0:.1f}s")

        # Diagnostic counts
        family_counts = Counter()
        difficulty_counts = Counter()
        for s in sentences:
            for c in s.constructions:
                family_counts[c.family] += 1
            difficulty_counts[s.curriculum.difficulty_level] += 1

        # Write
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "all.jsonl"
        t0 = time.time()
        write_jsonl(str(out_path), sentences)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  wrote {out_path} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")

        summary["sources"][tag] = {
            "n_sentences": len(sentences),
            "n_with_constructions": n_with,
            "family_counts": dict(family_counts),
            "difficulty_counts": dict(sorted(difficulty_counts.items())),
            "out_path": str(out_path),
            "size_mb": round(size_mb, 1),
        }

    summary["finished"] = time.time()
    summary["total_seconds"] = round(summary["finished"] - summary["started"], 1)

    summary_path = out_root / "build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSummary: {summary_path}")
    print(json.dumps({k: v for k, v in summary["sources"].items()
                       }, indent=2, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
