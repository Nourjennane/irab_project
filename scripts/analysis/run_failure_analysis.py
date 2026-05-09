"""Run the failure-analysis suite over a checkpoint × held-out set.

Loads the validated checkpoint, runs predictions on the full
(uncapped) eval sets, and writes::

  docs/failure_analysis/
      top_failures.md
      hardest_sentences.md
      role_confusions.md
      marker_confusions.md
      case_confusions.md
      long_range_failures.md
      nested_clause_failures.md
      overlap_failures.md
      calibration_failures.md
      structural_breakdown.md
      summary.json

This is data-quality infrastructure, not a model change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _load_model(ckpt_dir: Path, encoder_name: str, device):
    import torch
    from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel

    sd = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu",
                    weights_only=True)
    has_graph = any(k.startswith("graph_refiner.") or k == "graph_gate"
                    for k in sd)
    model = DepAwareStructuredModel(
        encoder_name=encoder_name,
        enable_morph_heads=True, morph_heads_enabled=None,
        enable_dep_features=True, enable_graph_refiner=has_graph,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device); model.eval()
    return model


def _md_top_failures(records, n: int = 50) -> str:
    lines = ["# Top failures (highest role confidence wrong)", ""]
    lines.append("| sid | tok | surface | gold (case/role/marker) "
                 "| pred (case/role/marker) | role_conf |")
    lines.append("|---|---|---|---|---|---|")
    for r in records[:n]:
        gold = f"{r.gold_case}/{r.gold_role}/{r.gold_marker}"
        pred = f"{r.pred_case}/{r.pred_role}/{r.pred_marker}"
        lines.append(f"| {r.sentence_id[:8]} | {r.token_index} | "
                      f"`{r.surface}` | {gold} | {pred} | "
                      f"{r.role_conf or 0:.3f} |")
    return "\n".join(lines)


def _md_confusions(top: List, axis: str, n: int = 20) -> str:
    lines = [f"# {axis.title()} confusions (gold → pred)", ""]
    lines.append(f"| gold | pred | count |")
    lines.append("|---|---|---|")
    for gold, pred, count in top[:n]:
        lines.append(f"| {gold} | {pred} | {count} |")
    return "\n".join(lines)


def _md_bucket(records, title: str, n: int = 30) -> str:
    lines = [f"# {title}", "",
             f"Total: **{len(records)}** failures.\n"]
    if not records:
        lines.append("No failures in this bucket.")
        return "\n".join(lines)
    lines.append("| sid | tok | surface | gold | pred | role_conf | "
                 "dep_d | clause_d | overlap |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in records[:n]:
        gold = f"{r.gold_case}/{r.gold_role}/{r.gold_marker}"
        pred = f"{r.pred_case}/{r.pred_role}/{r.pred_marker}"
        lines.append(f"| {r.sentence_id[:8]} | {r.token_index} | "
                      f"`{r.surface}` | {gold} | {pred} | "
                      f"{r.role_conf or 0:.3f} | "
                      f"{r.dependency_depth} | {r.clause_depth} | "
                      f"{'✓' if r.overlap else ''} |")
    return "\n".join(lines)


def _md_hardest_sentences(records, n: int = 30) -> str:
    """Sentences with the most failures."""
    from collections import Counter
    by_sid = Counter()
    for r in records:
        by_sid[r.sentence_id] += 1
    lines = ["# Hardest sentences (most failed tokens)", "",
             "| sid | n_failures |", "|---|---|"]
    for sid, n_fail in by_sid.most_common(n):
        lines.append(f"| {sid} | {n_fail} |")
    return "\n".join(lines)


def _md_structural(b: Dict[str, Dict]) -> str:
    lines = ["# Structural breakdown — `fully` accuracy by axis", ""]
    for axis_name, slots in b.items():
        lines.append(f"## {axis_name}")
        lines.append("")
        lines.append("| key | n | n_correct | fully |")
        lines.append("|---|---|---|---|")
        for k, slot in sorted(slots.items(), key=lambda x: str(x[0])):
            lines.append(f"| {k} | {slot['n']} | {slot['n_correct']} | "
                          f"{slot['fully']} |")
        lines.append("")
    return "\n".join(lines)


def _md_calibration(summary: Dict) -> str:
    lines = ["# Calibration analysis", ""]
    for axis, info in summary.items():
        lines.append(f"## {axis}")
        lines.append(f"- ECE: **{info['ece']}**")
        lines.append(f"- High-confidence wrong (≥0.95): "
                      f"**{info['high_conf_wrong_count']}**")
        rb = info["reliability"]
        lines.append("")
        lines.append("| bin | n | accuracy | mean_conf |")
        lines.append("|---|---|---|---|")
        for b in range(len(rb["n"])):
            lines.append(f"| [{b/10:.1f}-{(b+1)/10:.1f}) | {rb['n'][b]} | "
                          f"{rb['accuracy'][b]:.3f} | {rb['mean_conf'][b]:.3f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs" / "validated_nextgen_recovery"))
    ap.add_argument("--out_dir", default=str(ROOT / "docs" / "failure_analysis"))
    ap.add_argument("--datasets", nargs="+",
                    default=["gazelle_test", "masaq_quranic"])
    ap.add_argument("--encoder_name", default="UBC-NLP/AraT5v2-base-1024")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer
    from irab_tashkeel.data_v2.schema_v2 import read_jsonl
    from irab_tashkeel.training_v2.eval_hook import predict_for_eval
    from irab_tashkeel.eval_v2 import extract_outcomes
    from irab_tashkeel.analysis import (
        build_failure_records, bucket_failures,
        confusion_summary, structural_breakdown, calibration_summary,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint if (Path(args.checkpoint) / "tokenizer.json").exists()
        else args.encoder_name,
    )
    model = _load_model(Path(args.checkpoint), args.encoder_name, device)

    sentences = []
    for ds in args.datasets:
        p = ROOT / "data_v2" / "annotated" / ds / "all.jsonl"
        if p.exists():
            sentences.extend(list(read_jsonl(str(p))))
    print(f"loaded {len(sentences)} sentences")

    print("running predictions...")
    preds = predict_for_eval(model, tokenizer, sentences,
                              batch_size=args.batch_size, device=device)

    print("building failure records (only_failures=True)...")
    records = build_failure_records(sentences, preds, only_failures=True)
    print(f"  {len(records)} failure records")

    outcomes = extract_outcomes(sentences, preds)

    buckets = bucket_failures(records)
    confusions = confusion_summary(records)
    structural = structural_breakdown(outcomes)
    calib = calibration_summary(records)

    # ---- Write reports ----
    (out_dir / "top_failures.md").write_text(_md_top_failures(records, n=50))
    (out_dir / "hardest_sentences.md").write_text(_md_hardest_sentences(records))
    (out_dir / "role_confusions.md").write_text(
        _md_confusions(confusions["role"]["top"], "role"))
    (out_dir / "case_confusions.md").write_text(
        _md_confusions(confusions["case"]["top"], "case"))
    (out_dir / "marker_confusions.md").write_text(
        _md_confusions(confusions["marker"]["top"], "marker"))
    (out_dir / "long_range_failures.md").write_text(
        _md_bucket(buckets.get("long_range_failures", []),
                   "Long-range failures (head_distance ≥ 5)"))
    (out_dir / "nested_clause_failures.md").write_text(
        _md_bucket(buckets.get("nested_clause_failures", []),
                   "Nested-clause failures (clause_depth ≥ 2)"))
    (out_dir / "overlap_failures.md").write_text(
        _md_bucket(buckets.get("overlap_failures", []),
                   "Overlap failures (token in ≥ 2 constructions)"))
    (out_dir / "calibration_failures.md").write_text(
        _md_bucket(buckets.get("calibration_failures", []),
                   "Calibration failures (very high conf, still wrong)"))
    (out_dir / "structural_breakdown.md").write_text(_md_structural(structural))
    (out_dir / "calibration.md").write_text(_md_calibration(calib))

    # Machine-readable summary
    summary = {
        "checkpoint": args.checkpoint,
        "n_sentences": len(sentences),
        "n_failure_records": len(records),
        "buckets": {k: len(v) for k, v in buckets.items()},
        "confusions": confusions,
        "structural_breakdown": structural,
        "calibration": calib,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )
    print(f"\n✓ reports written to {out_dir}")


if __name__ == "__main__":
    main()
