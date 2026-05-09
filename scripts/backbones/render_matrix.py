"""Render the backbone comparison matrix from per-backbone outputs.

Reads ``runs/backbone_benchmark/<backbone_id>/<config>/`` for each
backbone × config combination, gathers the headline metrics, and
emits a single comparison-matrix markdown table.

Decision rule (from ``docs/roadmap/backbone_upgrade.md``):

  A backbone replaces AraT5v2-base as the next-gen production base
  ONLY if:

    1. Phase 3-A overall fully on Gazelle ≥ 25.2 (no regression), AND
    2. fully on MASAQ ≥ 14.9 + 1.0, OR
    3. construction-probing on the catastrophic Gazelle subsets
       (kana / istithnāʾ / quranic_proxy) shows ≥ +5 pp on at
       least two of the three.

If no candidate clears the rule, the comparison matrix itself is
the contribution: a publishable Arabic-backbone benchmark on
iʿrāb generation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


# Frozen-baseline reference numbers (Phase 3-A on FIXED evaluator)
FROZEN_BASELINE = {
    "gazelle_overall_fully":  25.2,
    "gazelle_overall_role":   40.2,
    "gazelle_overall_case":   72.0,
    "gazelle_kana_fully":     14.3,
    "gazelle_kana_role":      57.1,
    "masaq_overall_fully":    14.9,
    "masaq_kana_fully":       11.0,
    "masaq_kana_calib_gap":   -0.063,
}


def _load_metrics(backbone_dir: Path, config: str) -> Optional[Dict[str, Any]]:
    """Load per-construction summary JSONs for a backbone × config.

    Looks for: ``<backbone_dir>/<config>/eval_per_construction/{gazelle,masaq}/per_construction_summary.json``
    Returns None when the run hasn't produced eval outputs yet.
    """
    base = backbone_dir / config / "eval_per_construction"
    if not base.exists():
        return None

    results: Dict[str, Any] = {}
    for surface in ("gazelle", "masaq"):
        summary_path = base / surface / "per_construction_summary.json"
        if summary_path.exists():
            try:
                results[surface] = json.loads(summary_path.read_text())
            except Exception as e:
                print(f"  [warn] failed to parse {summary_path}: {e}")
    return results if results else None


def _extract_headline(results: Dict[str, Any]) -> Dict[str, float]:
    """Pull out the headline metrics for the matrix table."""
    g = results.get("gazelle", {}).get("overall", {})
    m = results.get("masaq", {}).get("overall", {})
    g_kana = results.get("gazelle", {}).get("per_construction", {}).get("kana_sisters", {})
    m_kana = results.get("masaq", {}).get("per_construction", {}).get("kana_sisters", {})
    return {
        "gazelle_overall_fully":  100 * g.get("fully", 0.0),
        "gazelle_overall_role":   100 * g.get("role_acc", 0.0),
        "gazelle_overall_case":   100 * g.get("case_acc", 0.0),
        "gazelle_kana_fully":     100 * g_kana.get("fully", 0.0),
        "gazelle_kana_role":      100 * g_kana.get("role_acc", 0.0),
        "masaq_overall_fully":    100 * m.get("fully", 0.0),
        "masaq_kana_fully":       100 * m_kana.get("fully", 0.0),
        "masaq_kana_calib_gap":   m_kana.get("calib_gap", 0.0),
    }


def _decision_rule(headline: Dict[str, float]) -> str:
    """Apply the production-replacement rule."""
    gz_fully = headline.get("gazelle_overall_fully", 0)
    gz_kana_fully = headline.get("gazelle_kana_fully", 0)
    masaq_fully = headline.get("masaq_overall_fully", 0)
    if gz_fully >= 25.2 and masaq_fully >= 15.9:
        return "✓ ship as production"
    if gz_fully >= 25.2 and gz_kana_fully >= 19.3:
        return "✓ ships on construction-probing rule"
    if gz_fully < 25.2 - 1.0:
        return "✗ Gazelle regression"
    return "○ within noise"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "runs" / "backbone_benchmark"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "ablations_v2" / "backbone_matrix.md"))
    args = ap.parse_args()

    from irab_tashkeel.backbones import all_backbones

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = ["# Backbone Comparison Matrix\n"]
    md.append("Source: `runs/backbone_benchmark/`. Decision rule per "
              "`docs/roadmap/backbone_upgrade.md`.\n")
    md.append("Frozen-baseline reference (Phase 3-A on fixed evaluator):\n")
    md.append("```")
    md.append(f"  Gazelle overall fully: 25.2  | role: 40.2  | case: 72.0")
    md.append(f"  Gazelle kana fully:    14.3  | role: 57.1")
    md.append(f"  MASAQ overall fully:   14.9  | kana: 11.0  | kana calib_gap: -0.063")
    md.append("```\n")

    md.append("## Phase 3-A full retrain comparison\n")
    md.append("| backbone | n_params | pretraining | "
              "Gaz fully | Gaz role | Gaz case | "
              "Gaz kana fully | MASAQ fully | MASAQ kana fully | "
              "decision |")
    md.append("|---|:---:|:---:|"
              "---:|---:|---:|"
              "---:|---:|---:|"
              "---|")

    n_completed = 0
    for spec in all_backbones():
        bb_dir = root / spec.backbone_id
        results = _load_metrics(bb_dir, "phase3a_full")
        if results is None:
            md.append(f"| {spec.backbone_id} | {spec.n_params_est} | "
                      f"{spec.arabic_pretraining} | — | — | — | — | — | — | "
                      f"⏳ pending |")
            continue
        n_completed += 1
        h = _extract_headline(results)
        decision = _decision_rule(h)
        md.append(f"| {spec.backbone_id} | {spec.n_params_est} | "
                  f"{spec.arabic_pretraining} | "
                  f"{h['gazelle_overall_fully']:.1f} | "
                  f"{h['gazelle_overall_role']:.1f} | "
                  f"{h['gazelle_overall_case']:.1f} | "
                  f"{h['gazelle_kana_fully']:.1f} | "
                  f"{h['masaq_overall_fully']:.1f} | "
                  f"{h['masaq_kana_fully']:.1f} | "
                  f"{decision} |")

    md.append("")
    md.append(f"## Status\n")
    md.append(f"- Backbones registered: {len(all_backbones())}")
    md.append(f"- Backbones with completed `phase3a_full` runs: {n_completed}")

    out_path.write_text("\n".join(md))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
