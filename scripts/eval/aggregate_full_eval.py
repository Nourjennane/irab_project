"""Phase A — aggregate raw eval JSONs into final report + tables.

Reads ``docs/final_eval/raw/<checkpoint>__<dataset>.json`` produced
by ``run_full_eval_v2.py`` and emits:

  - ``docs/final_eval/final_eval_tables.json`` — flat machine-readable
    table for downstream rendering / paper tables
  - ``docs/final_eval/final_eval_report.md`` — human-readable
    diff between Phase 3-A baseline and stage_7 final, with all
    metrics side-by-side, fully_observable_only vs full noisy,
    per-source breakdown

Does **not** touch the model or rerun any inference. Pure
post-processing over the raw JSON shards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]


def _load_raw(raw_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(raw_dir.glob("*.json")):
        out.append(json.loads(p.read_text()))
    return out


def _row(rec: Dict[str, Any], scope: str) -> Dict[str, Any]:
    src = rec.get(f"metrics_{scope}", {}) or {}
    return {
        "checkpoint": rec.get("checkpoint"),
        "dataset":    rec.get("dataset"),
        "scope":      scope,
        "n_sentences": rec.get("n_sentences"),
        "n_words":    src.get("n_words"),
        "case_acc":   src.get("case_acc"),
        "role_acc":   src.get("role_acc"),
        "marker_em":  src.get("marker_em"),
        "fully":      src.get("fully"),
        "calib_gap":  src.get("calib_gap"),
        "calib_correct": src.get("calib_correct"),
        "calib_wrong":   src.get("calib_wrong"),
    }


def _delta(a: float, b: float) -> str:
    """Format b-a as a signed delta with 4-digit precision."""
    if a is None or b is None:
        return "—"
    d = b - a
    return f"{d:+.4f}"


def _md_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(ROOT / "docs" / "final_eval" / "raw"))
    ap.add_argument("--out_dir", default=str(ROOT / "docs" / "final_eval"))
    ap.add_argument("--baseline", default="phase3a")
    ap.add_argument("--candidate", default="stage7")
    ap.add_argument("--candidate_src", default="",
                    help="Optional source path of the candidate checkpoint to "
                         "cite in the generated report (e.g. "
                         "runs/validated_nextgen_recovery/). If omitted no "
                         "path is cited.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_raw(raw_dir)
    if not raw:
        raise SystemExit(f"no raw eval JSONs in {raw_dir}")

    all_rows = []
    for rec in raw:
        for scope in ("full", "observable"):
            all_rows.append(_row(rec, scope))

    tables = {
        "raw_records": raw,
        "rows": all_rows,
    }
    (out_dir / "final_eval_tables.json").write_text(
        json.dumps(tables, indent=2, ensure_ascii=False, default=str)
    )
    print(f"wrote {out_dir / 'final_eval_tables.json'}")

    # =================================================================
    # Build markdown report
    # =================================================================
    by_ck_ds_scope = {(r["checkpoint"], r["dataset"], r["scope"]): r
                      for r in all_rows}
    datasets = sorted({r["dataset"] for r in all_rows})

    lines: List[str] = [
        "# Phase A — Full Independent Evaluation",
        "",
        f"Comparing **{args.baseline}** (baseline) vs **{args.candidate}** "
        f"(curriculum-trained final).",
        "",
        "All metrics computed by `eval_v2` from scratch on the *full*, "
        "uncapped test sets with deterministic seed.",
        "",
        "## Headline metrics — full noisy evaluation",
        "",
    ]

    headers = ["dataset", "n_sent", "n_words",
               f"{args.baseline}.case", f"{args.candidate}.case", "Δ case",
               f"{args.baseline}.role", f"{args.candidate}.role", "Δ role",
               f"{args.baseline}.marker", f"{args.candidate}.marker", "Δ marker",
               f"{args.baseline}.fully", f"{args.candidate}.fully", "Δ fully",
               f"{args.baseline}.cal_gap", f"{args.candidate}.cal_gap"]

    rows_full: List[Dict[str, Any]] = []
    rows_obs: List[Dict[str, Any]]  = []
    for ds in datasets:
        for scope, dst in (("full", rows_full), ("observable", rows_obs)):
            b = by_ck_ds_scope.get((args.baseline, ds, scope)) or {}
            c = by_ck_ds_scope.get((args.candidate, ds, scope)) or {}
            dst.append({
                "dataset": ds,
                "n_sent": b.get("n_sentences", c.get("n_sentences")),
                "n_words": b.get("n_words", c.get("n_words")),
                f"{args.baseline}.case":   b.get("case_acc"),
                f"{args.candidate}.case":  c.get("case_acc"),
                "Δ case":                  _delta(b.get("case_acc"), c.get("case_acc")),
                f"{args.baseline}.role":   b.get("role_acc"),
                f"{args.candidate}.role":  c.get("role_acc"),
                "Δ role":                  _delta(b.get("role_acc"), c.get("role_acc")),
                f"{args.baseline}.marker": b.get("marker_em"),
                f"{args.candidate}.marker": c.get("marker_em"),
                "Δ marker":                _delta(b.get("marker_em"), c.get("marker_em")),
                f"{args.baseline}.fully":  b.get("fully"),
                f"{args.candidate}.fully": c.get("fully"),
                "Δ fully":                 _delta(b.get("fully"), c.get("fully")),
                f"{args.baseline}.cal_gap": b.get("calib_gap"),
                f"{args.candidate}.cal_gap": c.get("calib_gap"),
            })

    lines.append(_md_table(rows_full, headers))
    lines += [
        "",
        "## Headline metrics — fully-observable subset only",
        "",
        "Only tokens where all three gold fields (case, role, marker) "
        "are populated. Strict apples-to-apples accuracy.",
        "",
        _md_table(rows_obs, headers),
        "",
        "## Notes",
        "",
        f"- Baseline checkpoint: `{args.baseline}` "
        "(Phase 3-A, frozen prod model — see `runs/phase3a_491240/final/`).",
        f"- Candidate checkpoint: `{args.candidate}`"
        + (f" (source: `{args.candidate_src}`)." if args.candidate_src else "."),
        "- Metric module: `src/irab_tashkeel/eval_v2/` (single source of truth).",
        "- Evaluator entrypoint: `scripts/eval/run_full_eval_v2.py`.",
        "- This report is regenerated by `scripts/eval/aggregate_full_eval.py`.",
        "- Per-construction breakdown, calibration bins, and ambiguity-robust "
        "metrics are in the raw JSON under `docs/final_eval/raw/`.",
        "",
    ]

    (out_dir / "final_eval_report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir / 'final_eval_report.md'}")


if __name__ == "__main__":
    main()
