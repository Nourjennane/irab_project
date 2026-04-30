"""Construction-type classification + per-category error breakdown for Gazelle.

The 30-sentence eval is small but heterogeneous — some are simple
verbal sentences, others have iḍāfa chains, mood-shifting particles, or
exception (استثناء) constructions that stress the i'rāb model differently.
A flat 134-word table hides these failure modes.

This module:
  1. Heuristically classifies each gold sentence into one or more
     construction tags (a sentence can belong to multiple categories).
  2. Joins these tags onto each per-word judgment.
  3. Produces a per-category × per-system breakdown of fully_correct_word.

Tags (overlapping):
  - VERBAL              starts with a finite verb (verb-initial sentence)
  - NOMINAL             nominal/equational sentence (no leading verb)
  - IDAFA_HEAVY         contains ≥2 consecutive nouns in genitive (chain)
  - PREPOSITIONAL       contains a preposition (حرف جر)
  - PARTICLE_MOOD       contains a mood-shifting particle (إن، أن، لن، لم،
                        كان، …) or a sister of inna/kana
  - EXCEPTION           contains an istithnāʾ marker (سوى، إلا، عدا، …)

Heuristics use the gold i'rāb prose itself (we have it!) so we don't need
a separate parser. Tags are coarse but consistent across systems.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from .stats import (
    WordJudgment, build_judgments, mcnemar_exact, paired_bootstrap_proportion_delta,
    per_judgment_scores, bootstrap_proportion,
)


# ---------------------------------------------------------------------------
# Tag heuristics
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    return re.sub(r"[ً-ْٰ]", "", s)


PARTICLE_KEYS = ("إن", "أن", "لن", "لم", "لا", "كان", "ليس", "أصبح", "صار",
                 "بات", "ظل", "ما زال", "ما برح", "ما فتئ", "ليت", "لعل",
                 "كأن", "لكن")
EXCEPTION_KEYS = ("سوى", "إلا", "عدا", "خلا", "حاشا", "غير")


def classify_sentence(sentence: str, gold_irab_concat: str) -> Set[str]:
    """Return the set of construction tags for one sentence."""
    sn = _norm(sentence).strip()
    irab = _norm(gold_irab_concat)
    tags: Set[str] = set()

    # VERBAL vs NOMINAL: look at the first non-punctuation word and the
    # gold i'rāb of the first word.
    first_word_irab = irab.split("\n", 1)[0] if irab else ""
    if "فعل ماض" in first_word_irab or "فعل مضارع" in first_word_irab or "فعل أمر" in first_word_irab:
        tags.add("VERBAL")
    elif "مبتدأ" in first_word_irab:
        tags.add("NOMINAL")

    # IDAFA_HEAVY: ≥2 instances of "مضاف إليه" in the gold i'rāb.
    if irab.count("مضاف إليه") >= 2:
        tags.add("IDAFA_HEAVY")

    # PREPOSITIONAL: has a preposition.
    if "حرف جر" in irab:
        tags.add("PREPOSITIONAL")

    # PARTICLE_MOOD: gold or surface mentions a mood-shifting particle.
    for k in PARTICLE_KEYS:
        if re.search(rf"(^|[^ء-ي]){re.escape(k)}([^ء-ي]|$)", sn):
            tags.add("PARTICLE_MOOD")
            break

    # EXCEPTION: istithnāʾ.
    for k in EXCEPTION_KEYS:
        if re.search(rf"(^|[^ء-ي]){re.escape(k)}([^ء-ي]|$)", sn):
            tags.add("EXCEPTION")
            break

    if not tags:
        tags.add("OTHER")
    return tags


def tag_gazelle(predictions_path: Path | str) -> List[Set[str]]:
    """Tag each Gazelle sentence using one prediction JSONL as gold source."""
    rows = []
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    out = []
    for row in rows:
        sent = row.get("sentence") or ""
        gold_concat = "\n".join((g.get("irab") or "") for g in (row.get("gold") or []))
        out.append(classify_sentence(sent, gold_concat))
    return out


# ---------------------------------------------------------------------------
# Per-category breakdown
# ---------------------------------------------------------------------------
@dataclass
class CategoryBreakdown:
    tag: str
    n_sentences: int
    n_words: int
    per_system: Dict[str, Dict[str, float]]  # name -> {metric: rate}


def per_category_table(
    judgments: Sequence[WordJudgment],
    sentence_tags: Sequence[Set[str]],
    systems: Sequence[str],
    metrics: Sequence[str] = ("case", "marker", "fully"),
    B: int = 1000,
) -> List[CategoryBreakdown]:
    """Group judgments by sentence tag, compute per-system metric per group.

    A judgment with sent_idx i contributes to every tag in sentence_tags[i].
    """
    all_tags: Set[str] = set()
    for ts in sentence_tags:
        all_tags |= ts
    rows: List[CategoryBreakdown] = []
    for tag in sorted(all_tags):
        idxs = [i for i, ts in enumerate(sentence_tags) if tag in ts]
        sub = [j for j in judgments if j.sent_idx in set(idxs)]
        if not sub:
            continue
        per_system: Dict[str, Dict[str, float]] = {}
        for sys in systems:
            scores = per_judgment_scores(sub, sys)
            entry = {}
            for m in metrics:
                pt, lo, hi = bootstrap_proportion([s[m] for s in scores], B=B)
                entry[m] = pt
                entry[f"{m}_ci_low"] = lo
                entry[f"{m}_ci_high"] = hi
            per_system[sys] = entry
        rows.append(CategoryBreakdown(
            tag=tag, n_sentences=len(idxs), n_words=len(sub),
            per_system=per_system,
        ))
    return rows


def pretty_per_category_table(
    rows: Sequence[CategoryBreakdown],
    systems: Sequence[str],
    metric: str = "fully",
) -> str:
    head = f"\n  {'tag':18} {'#sent':>5} {'#word':>5}  " + "  ".join(f"{s:>22}" for s in systems)
    lines = [head]
    for row in rows:
        cells = []
        for s in systems:
            stats = row.per_system.get(s, {})
            pt = stats.get(metric, 0)
            lo = stats.get(f"{metric}_ci_low", 0)
            hi = stats.get(f"{metric}_ci_high", 0)
            cells.append(f"{pt*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}]")
        lines.append(f"  {row.tag:18} {row.n_sentences:>5} {row.n_words:>5}  " + "  ".join(f"{c:>22}" for c in cells))
    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Per-construction error analysis")
    p.add_argument("--system", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--metric", choices=["case", "marker", "role", "fully", "well_formed"],
                   default="fully")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    system_paths = {}
    for spec in args.system:
        name, path = spec.split("=", 1)
        system_paths[name] = Path(path)
    judgments = build_judgments(system_paths)

    # Use the first system's predictions JSONL as the gold/sentence source
    gold_path = next(iter(system_paths.values()))
    sentence_tags = tag_gazelle(gold_path)

    # Sentence-level tag counts (overlap allowed)
    tag_counts = Counter()
    for ts in sentence_tags:
        for t in ts:
            tag_counts[t] += 1
    print("Sentence-level tag distribution (sentences may have multiple tags):")
    for t, c in tag_counts.most_common():
        print(f"  {t:18} {c:>3}/{len(sentence_tags)}")

    rows = per_category_table(judgments, sentence_tags, list(system_paths.keys()))

    print(f"\nPer-category {args.metric} (95% bootstrap CI, B=1000):")
    print(pretty_per_category_table(rows, list(system_paths.keys()), metric=args.metric))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "tag_distribution": dict(tag_counts),
                "per_category": [
                    {"tag": r.tag, "n_sentences": r.n_sentences, "n_words": r.n_words,
                     "per_system": r.per_system}
                    for r in rows
                ],
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  wrote → {args.out}")


if __name__ == "__main__":
    main()
