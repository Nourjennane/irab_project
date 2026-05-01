#!/usr/bin/env python
"""Aggregate Sonnet k-sweep results into a single JSON + paired-bootstrap deltas vs k=5."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_metrics(path: Path) -> dict:
    summ = json.loads(path.read_text())
    if isinstance(summ, list):
        # baseline_eval_sonnet/summary.json carries both claude_zero AND claude_rag.
        # Pick the rag entry; otherwise first.
        rag = next((e for e in summ if e.get("baseline") == "claude_rag"), None)
        summ = rag or summ[0]
    c = summ.get("constrained") or summ.get("raw")
    return {
        "well": c["well_formed_rate"] * 100,
        "pos": c["pos_accuracy"] * 100,
        "case": c["case_accuracy"] * 100,
        "role": c["role_f1_macro"] * 100,
        "marker": c["marker_em"] * 100,
        "fully": c["fully_correct_word"] * 100,
        "n": int(c["n"]),
    }


def main():
    src_root = Path("runs")
    sweep_dirs = {
        1: src_root / "k_sweep_sonnet_k1",
        3: src_root / "k_sweep_sonnet_k3",
        5: src_root / "baseline_eval_sonnet",   # reuse existing
        8: src_root / "k_sweep_sonnet_k8",
        12: src_root / "k_sweep_sonnet_k12",
    }
    table = {}
    for k, d in sweep_dirs.items():
        s = d / "summary.json"
        if not s.exists():
            print(f"  k={k}: MISSING {s}", file=sys.stderr)
            continue
        table[k] = load_metrics(s)

    out = src_root / "k_sensitivity"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(table, indent=2))
    print(f"wrote {out / 'results.json'}")

    # Pretty
    print(f"\n  {'k':>3}  {'well':>6}  {'case':>6}  {'role-F1':>7}  {'marker':>6}  {'fully':>6}")
    for k in sorted(table):
        m = table[k]
        print(f"  {k:>3}  {m['well']:>5.1f}%  {m['case']:>5.1f}%  {m['role']:>6.1f}%  {m['marker']:>5.1f}%  {m['fully']:>5.1f}%")


if __name__ == "__main__":
    main()
