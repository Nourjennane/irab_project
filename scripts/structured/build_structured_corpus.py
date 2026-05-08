"""Build the structured-prediction training corpus from distill_v2.

Reads ``data/distill_v2/distilled.jsonl`` (5,000 sentence-level rows from the
Haiku-distilled corpus), canonicalizes each per-word i'rāb item to the small
schema in :mod:`irab_tashkeel.structured.schema`, and writes train / val splits
to ``data/structured_v1/{train,val}.jsonl``.

Each output row is a sentence-level :class:`SentenceIrab` JSON line:

    {
      "sentence": "كما يوجد ممر حديث ...",
      "items": [
        {"word": "كما", "case": "mabni", "role": "harf_jarr",
         "marker": "sukun", "pos": "particle", "irab_prose": "..."},
        ...
      ]
    }

Sentences are kept as a unit (not flattened to words) because the encoder
needs surrounding context. ``--val_frac 0.05`` is the default split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irab_tashkeel.structured.schema import (  # noqa: E402
    canonicalize_case,
    canonicalize_role,
    canonicalize_marker,
    derive_pos,
)
from irab_tashkeel.structured.word_irab import SentenceIrab, WordIrab  # noqa: E402


def iter_distilled(path: Path) -> Iterable[Dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def canonicalize_sentence(rec: Dict) -> SentenceIrab:
    items: List[WordIrab] = []
    for it in rec.get("items", []):
        items.append(
            WordIrab(
                word=it.get("word", ""),
                case=canonicalize_case(it.get("case")),
                role=canonicalize_role(it.get("role")),
                marker=canonicalize_marker(it.get("marker")),
                pos=derive_pos(it.get("role"), it.get("irab"), raw_pos=it.get("pos")),
                irab_prose=it.get("irab"),
            )
        )
    return SentenceIrab(sentence=rec.get("sentence", ""), items=items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/distill_v2/distilled.jsonl")
    ap.add_argument("--out_dir", default="data/structured_v1")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences: List[SentenceIrab] = []
    case_cnt: Counter = Counter()
    role_cnt: Counter = Counter()
    marker_cnt: Counter = Counter()
    pos_cnt: Counter = Counter()
    n_words = 0
    n_full_label = 0

    for rec in iter_distilled(in_path):
        sent = canonicalize_sentence(rec)
        sentences.append(sent)
        for w in sent.items:
            n_words += 1
            case_cnt[w.case] += 1
            role_cnt[w.role] += 1
            marker_cnt[w.marker] += 1
            pos_cnt[w.pos] += 1
            if w.has_full_label():
                n_full_label += 1

    rng = random.Random(args.seed)
    rng.shuffle(sentences)
    n_val = max(1, int(round(len(sentences) * args.val_frac)))
    val = sentences[:n_val]
    train = sentences[n_val:]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    with train_path.open("w") as fh:
        for s in train:
            fh.write(s.to_json_line() + "\n")
    with val_path.open("w") as fh:
        for s in val:
            fh.write(s.to_json_line() + "\n")

    stats = {
        "n_sentences": len(sentences),
        "n_train_sentences": len(train),
        "n_val_sentences": len(val),
        "n_words": n_words,
        "n_full_label": n_full_label,
        "full_label_rate": n_full_label / max(n_words, 1),
        "case": dict(case_cnt.most_common()),
        "role": dict(role_cnt.most_common()),
        "marker": dict(marker_cnt.most_common()),
        "pos": dict(pos_cnt.most_common()),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    print(f"sentences: {len(sentences)} (train {len(train)} / val {len(val)})")
    print(f"words: {n_words:,}  full-label: {n_full_label:,} ({100*n_full_label/n_words:.2f}%)")
    print(f"wrote: {train_path}, {val_path}, {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
