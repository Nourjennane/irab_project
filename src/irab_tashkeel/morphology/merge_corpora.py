"""Merge UD-PADT (morphology) + distill_v2 (i'rāb) into a unified corpus.

Each output JSONL line carries:
  * the surface sentence + per-word records
  * an i'rāb block (if from distill_v2) — case/role/marker/pos labels
  * a morph block (if from UD-PADT)   — gender/number/.../UPOS-derived pos
  * **per-head presence flags** at the example level

Per-head presence flags drive the masked multi-task loss in the trainer:
when ``has_irab=False`` the four i'rāb head losses are masked to zero on
that example (and the morph heads are active); vice-versa when
``has_morph=False``.  No example has BOTH active in Phase 1 (we don't have
parallel morph + i'rāb annotations for the same surface words; future
phases could merge by surface alignment).

Output schema (one JSON record per sentence):

    {
      "sentence":   "...",
      "source":     "UD-PADT" | "distill_v2",
      "has_irab":   true | false,
      "has_morph":  true | false,
      "items": [
        {
          "word": "...",

          # i'rāb (only when has_irab=True)
          "case":   "...",  "role":   "...",
          "marker": "...",  "pos":    "...",

          # morphology (only when has_morph=True)
          "gender":   "m|f|und",
          "number":   "sg|dual|pl|und",
          "definite": "def|indef|cons|und",
          "person":   "1|2|3|und",
          "aspect":   "imp|perf|und",
          "mood":     "ind|imp_mood|sub|jus|und",
          "voice":    "act|pass|und",
          "upos":     "NOUN|VERB|...",      # raw UD UPOS, kept for inspection
          "pos_ud":   "noun|verb|...",      # canonical 6-class derived from UPOS

          # NB: when both has_irab=True and has_morph=True (future phase),
          # the morph "pos_ud" and the i'rāb "pos" may disagree; the trainer
          # uses the i'rāb "pos" because it matches rev 2's existing POS head.
        },
        ...
      ]
    }

The trainer's dataset class reads this directly; no further preprocessing
needed.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irab_tashkeel.morphology.schema import MORPH_FEATURES
from irab_tashkeel.morphology.ud_loader import parse_conllu
from irab_tashkeel.morphology.word_morph import SentenceMorph, WordMorph
from irab_tashkeel.structured.word_irab import SentenceIrab, WordIrab


def _ud_to_record(sent: SentenceMorph) -> Dict:
    """Convert a UD-PADT SentenceMorph to a unified-corpus record."""
    items = []
    for w in sent.items:
        items.append({
            "word": w.word,
            "gender": w.gender, "number": w.number, "definite": w.definite,
            "person": w.person, "aspect": w.aspect, "mood": w.mood,
            "voice": w.voice,
            "upos": w.upos, "pos_ud": w.pos,
        })
    return {
        "sentence": sent.sentence,
        "source": "UD-PADT",
        "sent_id": sent.sent_id,
        "has_irab": False,
        "has_morph": True,
        "items": items,
    }


def _irab_to_record(sent: SentenceIrab) -> Dict:
    items = []
    for w in sent.items:
        items.append({
            "word": w.word,
            "case": w.case, "role": w.role, "marker": w.marker, "pos": w.pos,
            "irab_prose": w.irab_prose,
        })
    return {
        "sentence": sent.sentence,
        "source": "distill_v2",
        "has_irab": True,
        "has_morph": False,
        "items": items,
    }


def merge(
    ud_paths: List[Path],
    irab_paths: List[Path],
    out_path: Path,
    *,
    val_frac: float = 0.05,
    seed: int = 42,
) -> Dict:
    """Merge UD-PADT + distill_v2 streams into one shuffled JSONL.

    Returns a stats dict describing the corpus composition.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    records: List[Dict] = []
    n_ud = 0
    for p in ud_paths:
        for s in parse_conllu(p):
            records.append(_ud_to_record(s))
            n_ud += 1

    n_irab = 0
    for p in irab_paths:
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                s = SentenceIrab.from_json_line(line)
                records.append(_irab_to_record(s))
                n_irab += 1

    rng.shuffle(records)
    n_val = max(1, int(round(len(records) * val_frac)))
    val = records[:n_val]
    train = records[n_val:]

    train_path = out_path
    val_path = out_path.parent / out_path.name.replace("train", "val")
    if val_path == train_path:
        val_path = train_path.with_suffix(".val.jsonl")

    with train_path.open("w") as fh:
        for r in train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with val_path.open("w") as fh:
        for r in val:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats: per-source counts in train / val
    src_train = Counter(r["source"] for r in train)
    src_val = Counter(r["source"] for r in val)
    total_words = sum(len(r["items"]) for r in records)

    stats = {
        "n_sentences_total": len(records),
        "n_train": len(train),
        "n_val": len(val),
        "n_ud_total": n_ud,
        "n_irab_total": n_irab,
        "n_words_total": total_words,
        "train_by_source": dict(src_train),
        "val_by_source": dict(src_val),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    stats_path = train_path.parent / "merge_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ud_train", default="data/ud_padt/ar_padt-ud-train.conllu")
    ap.add_argument("--ud_dev", default="data/ud_padt/ar_padt-ud-dev.conllu")
    ap.add_argument("--irab_train", default="data/structured_v1/train.jsonl")
    ap.add_argument("--irab_val", default="data/structured_v1/val.jsonl")
    ap.add_argument("--out", default="data/morph_v1/train.jsonl")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stats = merge(
        ud_paths=[Path(args.ud_train), Path(args.ud_dev)],
        irab_paths=[Path(args.irab_train), Path(args.irab_val)],
        out_path=Path(args.out),
        val_frac=args.val_frac,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
