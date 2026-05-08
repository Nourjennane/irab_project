"""Phase 4a corpus builder: re-canonicalise distill_v2 against taxonomy_v4.

Reads the same distill_v2/distilled.jsonl source as the v3 pipeline; emits
sentence-level JSONL with v4 (34-label) role canonicalisation. Writes to
``data/structured_v1_v4/{train,val}.jsonl``. Identical seed (42) and
val_frac (0.05) as ``build_structured_corpus.py`` so train/val sentences
are byte-identical between v3 and v4 — only the role label differs.

This deliberate sentence-level identity makes per-sentence ablation between
rev 2 (v3, frozen), Phase 1 (v3, frozen), Phase 4-no-morph (v4), and Phase
4-full (v4 + morph) directly comparable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irab_tashkeel.structured.schema import (
    canonicalize_case, canonicalize_marker, derive_pos,
)
from irab_tashkeel.structured.taxonomy_v4 import canonicalize_role_v4
from irab_tashkeel.structured.word_irab import SentenceIrab, WordIrab


def _canonicalize_sentence_v4(rec: dict) -> SentenceIrab:
    items = []
    for it in rec.get("items", []):
        items.append(
            WordIrab(
                word=it.get("word", ""),
                case=canonicalize_case(it.get("case")),
                role=canonicalize_role_v4(it.get("role")),  # v4 instead of v3
                marker=canonicalize_marker(it.get("marker")),
                pos=derive_pos(it.get("role"), it.get("irab"), raw_pos=it.get("pos")),
                irab_prose=it.get("irab"),
            )
        )
    return SentenceIrab(sentence=rec.get("sentence", ""), items=items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/distill_v2/distilled.jsonl")
    ap.add_argument("--out_dir", default="data/structured_v1_v4")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences: list[SentenceIrab] = []
    role_cnt: Counter = Counter()
    n_words = 0

    with in_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sent = _canonicalize_sentence_v4(rec)
            sentences.append(sent)
            for w in sent.items:
                role_cnt[w.role] += 1
                n_words += 1

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

    # Per-class support: train + val combined.
    train_role: Counter = Counter()
    val_role: Counter = Counter()
    for s in train:
        for w in s.items:
            train_role[w.role] += 1
    for s in val:
        for w in s.items:
            val_role[w.role] += 1

    # Sort by descending train support
    from irab_tashkeel.structured.taxonomy_v4 import ROLE_LABELS_V4
    print(f"\n=== Phase 4a v4 corpus stats ===")
    print(f"sentences: {len(sentences)} (train {len(train)} / val {len(val)})")
    print(f"words: {n_words:,}")
    print(f"\n{'role':<18} {'train':>7} {'val':>5}")
    print("-" * 35)
    for label in sorted(ROLE_LABELS_V4, key=lambda r: -train_role[r]):
        t = train_role[label]
        v = val_role[label]
        print(f"{label:<18} {t:>7} {v:>5}")

    # Stats summary
    stats = {
        "n_sentences": len(sentences),
        "n_train": len(train),
        "n_val": len(val),
        "n_words_total": n_words,
        "train_by_role": dict(train_role),
        "val_by_role": dict(val_role),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nwrote {train_path}, {val_path}, {out_dir/'stats.json'}")


if __name__ == "__main__":
    main()
