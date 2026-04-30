"""Progress reporter for the manual gold-benchmark review.

Counts verified rows + flags items where the structural extractor cannot
recover all three of {case, role, marker}, which usually means a formatting
drift you'd want to fix.

    python -m irab_tashkeel.evaluation.gold_progress
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .structural import extract


def main():
    p = argparse.ArgumentParser(description="Gold benchmark review progress")
    p.add_argument("--path", type=Path, default=Path("data/gold_seed.jsonl"))
    p.add_argument("--show_flagged", type=int, default=10,
                   help="Show up to N flagged items (extractor couldn't recover case+role+marker)")
    args = p.parse_args()

    if not args.path.exists():
        print(f"❌ {args.path} not found")
        return

    rows = []
    with open(args.path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    n_sents = len(rows)
    n_words = sum(len(r.get("items") or []) for r in rows)
    n_verified = sum(
        1 for r in rows for it in (r.get("items") or []) if it.get("verified")
    )

    pct_sent = sum(
        1 for r in rows
        if r.get("items") and all(it.get("verified") for it in r["items"])
    )

    print(f"--- {args.path} ---")
    print(f"  sentences:           {n_sents}")
    print(f"  word judgments:      {n_words}")
    print(f"  verified words:      {n_verified} / {n_words}  ({n_verified*100/max(1,n_words):.1f}%)")
    print(f"  fully-verified sent: {pct_sent} / {n_sents}  ({pct_sent*100/max(1,n_sents):.1f}%)")

    # Flag items where extractor can't recover all three structured fields.
    flagged = []
    field_misses = Counter()
    for r in rows:
        for it in r.get("items") or []:
            irab = (it.get("irab") or "").strip()
            if not irab:
                continue
            ana = extract(irab)
            missing = []
            if ana.case is None:
                missing.append("case")
            if ana.role is None:
                # Skip role-less mabniyy particles — those genuinely have no role
                if not (ana.case == "mabni" and "حرف" in (ana.pos or "")):
                    missing.append("role")
            if ana.marker is None and ana.case in {"marfu", "mansub", "majrur", "majzum"}:
                missing.append("marker")
            if missing:
                for m in missing:
                    field_misses[m] += 1
                flagged.append((r.get("sentence", "")[:60], it.get("word", ""), irab, missing))

    print()
    print(f"  flagged items (extractor missed a field):  {len(flagged)} / {n_words}")
    if field_misses:
        for f_, c in field_misses.most_common():
            print(f"    {f_:8} missing in {c} items")

    if args.show_flagged and flagged:
        print(f"\n  examples (up to {args.show_flagged}):")
        for sent, word, irab, missing in flagged[: args.show_flagged]:
            tag = ",".join(missing)
            print(f"    [{tag:18}] {word:14} | {irab[:80]}")
            print(f"                       sent: {sent}")


if __name__ == "__main__":
    main()
