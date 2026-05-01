#!/usr/bin/env python
"""Compare two Sonnet RAG runs to estimate inference-time variance.

Computes:
  - per-run headline metrics (well, case, role-F1, marker, fully)
  - paired bootstrap deltas + McNemar p
  - per-word agreement on each structural field
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    a = Path("runs/baseline_eval_sonnet/claude_rag.predictions.jsonl")
    b = Path("runs/baseline_eval_sonnet_repro/claude_rag.predictions.jsonl")
    if not a.exists() or not b.exists():
        print(f"missing one of {a} / {b}", file=sys.stderr)
        sys.exit(1)

    # Reuse the stats CLI
    import subprocess
    print("--- Variance: Sonnet RAG run #2 vs run #1 ---\n")
    cmd = [
        ".venv/bin/python", "-m", "irab_tashkeel.evaluation.stats",
        "--system", f"sonnet_rag_run1={a}",
        "--system", f"sonnet_rag_run2={b}",
        "--reference", "sonnet_rag_run1", "--B", "1000",
        "--out", "runs/variance/sonnet_repro_paired.json",
    ]
    Path("runs/variance").mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)

    # Per-word agreement
    from irab_tashkeel.evaluation.discrimination import load_predictions, _norm_word
    from irab_tashkeel.evaluation.structural import extract
    p1 = load_predictions(a)
    p2 = load_predictions(b)
    common = set(p1) & set(p2)
    n = len(common)
    case_agree = role_agree = marker_agree = fully_agree = 0
    for k in common:
        e1, e2 = extract(p1[k]), extract(p2[k])
        if e1.case == e2.case:   case_agree += 1
        if e1.role == e2.role:   role_agree += 1
        if e1.marker == e2.marker: marker_agree += 1
        if e1.case == e2.case and e1.role == e2.role and e1.marker == e2.marker:
            fully_agree += 1
    print(f"\n--- Per-word agreement across the two runs (n={n} aligned words) ---")
    for name, k in (("case", case_agree), ("role", role_agree),
                    ("marker", marker_agree), ("fully", fully_agree)):
        pct = (k / n * 100) if n else 0
        print(f"  {name:6}: {k}/{n}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
