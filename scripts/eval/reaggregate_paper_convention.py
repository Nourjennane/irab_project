"""Re-aggregate every raw eval JSON shard under the *paper convention*:
denominator = n_words for every axis (case, role, marker, fully).

Existing eval_v2 JSON divides each axis by its own observable count
(n_observable_case, n_observable_role, etc.), which inflates the
percentages. The published paper anchors on n_words (the full token
count). This script re-derives the paper-convention numbers from the
existing shards without rerunning the model.

Output:
  docs/eval_unified/unified_metrics.json   side-by-side both conventions
  docs/eval_unified/unified_report.md      human-readable

The numerators (count of correct tokens on each axis) are unchanged —
only the denominator is re-anchored.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]


def numerators_from_shard(d: Dict) -> Dict[str, int]:
    """Reconstruct numerators from an eval_v2 shard (case_correct,
    role_correct, marker_correct, fully_correct counts)."""
    m = d.get("metrics_full", {}) or {}
    n_words = m.get("n_words", d.get("n_tokens", 0)) or 0
    n_obs_case   = m.get("n_observable_case",   0)
    n_obs_role   = m.get("n_observable_role",   0)
    n_obs_marker = m.get("n_observable_marker", 0)
    n_obs_fully  = m.get("n_observable_fully",  0)
    case_corr   = round(m.get("case_acc",  0.0) * n_obs_case)
    role_corr   = round(m.get("role_acc",  0.0) * n_obs_role)
    marker_corr = round(m.get("marker_em", 0.0) * n_obs_marker)
    fully_corr  = round(m.get("fully",     0.0) * n_obs_fully)
    return {
        "n_words":       int(n_words),
        "n_obs_case":    int(n_obs_case),
        "n_obs_role":    int(n_obs_role),
        "n_obs_marker":  int(n_obs_marker),
        "n_obs_fully":   int(n_obs_fully),
        "case_corr":     int(case_corr),
        "role_corr":     int(role_corr),
        "marker_corr":   int(marker_corr),
        "fully_corr":    int(fully_corr),
    }


def both_conventions(num: Dict[str, int]) -> Dict[str, Any]:
    n = num["n_words"]
    return {
        "paper": {
            "n":         n,
            "case":      round(num["case_corr"]   / max(n, 1), 4),
            "role":      round(num["role_corr"]   / max(n, 1), 4),
            "marker":    round(num["marker_corr"] / max(n, 1), 4),
            "fully":     round(num["fully_corr"]  / max(n, 1), 4),
        },
        "observable": {
            "n":         num["n_obs_fully"],
            "case_n":    num["n_obs_case"],
            "role_n":    num["n_obs_role"],
            "marker_n":  num["n_obs_marker"],
            "fully_n":   num["n_obs_fully"],
            "case":      round(num["case_corr"]   / max(num["n_obs_case"],   1), 4),
            "role":      round(num["role_corr"]   / max(num["n_obs_role"],   1), 4),
            "marker":    round(num["marker_corr"] / max(num["n_obs_marker"], 1), 4),
            "fully":     round(num["fully_corr"]  / max(num["n_obs_fully"],  1), 4),
        },
        "numerators": num,
    }


def collect_shards() -> Dict[str, Dict[str, Any]]:
    """Walk all docs/final_eval*/raw/<ckpt>__<ds>.json and gather them
    keyed by (eval_dir, checkpoint, dataset).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for eval_dir in sorted((ROOT / "docs").glob("final_eval*")):
        raw = eval_dir / "raw"
        if not raw.is_dir():
            continue
        for f in sorted(raw.glob("*.json")):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            ckpt = d.get("checkpoint")
            ds = d.get("dataset")
            if not ckpt or not ds:
                continue
            num = numerators_from_shard(d)
            out.setdefault(eval_dir.name, {}).setdefault(ckpt, {})[ds] = both_conventions(num)
    return out


def main():
    out_dir = ROOT / "docs" / "eval_unified"
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = collect_shards()
    (out_dir / "unified_metrics.json").write_text(
        json.dumps(shards, indent=2, ensure_ascii=False)
    )

    # Markdown — unified table per dataset across all checkpoints
    lines: List[str] = [
        "# Unified evaluation — both conventions, side-by-side",
        "",
        "**Primary metric:** *paper convention* — denominator = `n_words`",
        "(every Gazelle/MASAQ word judgment, including those where gold is",
        "missing on a given axis — those count as wrong on that axis).",
        "",
        "**Secondary diagnostic:** *fully-observable subset* — denominator =",
        "`n_observable_fully` (tokens where all 3 gold fields are populated).",
        "",
        "These are the same model on the same data; only the denominator",
        "differs. The numerators (tokens correct on each axis) are unchanged.",
        "",
    ]

    for eval_name, by_ckpt in shards.items():
        lines.append(f"## {eval_name}")
        lines.append("")
        # Datasets in canonical order
        datasets = sorted({d for ck in by_ckpt.values() for d in ck.keys()})
        for ds in datasets:
            lines.append(f"### {ds}")
            lines.append("")
            ckpts = sorted(by_ckpt.keys())
            # Paper convention
            lines.append("**Paper convention (denominator = n_words):**")
            lines.append("")
            lines.append("| checkpoint | n_words | case | role | marker | fully |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for ck in ckpts:
                if ds not in by_ckpt[ck]:
                    continue
                p = by_ckpt[ck][ds]["paper"]
                lines.append(f"| {ck} | {p['n']} | {p['case']:.4f} | "
                              f"{p['role']:.4f} | {p['marker']:.4f} | "
                              f"**{p['fully']:.4f}** |")
            lines.append("")
            # Observable subset
            lines.append("**Fully-observable subset (denominator = n_observable_fully):**")
            lines.append("")
            lines.append("| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for ck in ckpts:
                if ds not in by_ckpt[ck]:
                    continue
                o = by_ckpt[ck][ds]["observable"]
                lines.append(f"| {ck} | {o['fully_n']} | {o['case']:.4f} | "
                              f"{o['role']:.4f} | {o['marker']:.4f} | "
                              f"**{o['fully']:.4f}** |")
            lines.append("")

    (out_dir / "unified_report.md").write_text("\n".join(lines))
    print(f"✓ wrote {out_dir / 'unified_report.md'}")
    print(f"✓ wrote {out_dir / 'unified_metrics.json'}")


if __name__ == "__main__":
    main()
