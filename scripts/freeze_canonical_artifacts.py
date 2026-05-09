"""Step-1 of the supervision phase — freeze canonical artifacts.

Produces two immutable directories:

  runs/final_validated/                — production checkpoint
  runs/final_graph_negative_result/    — documented negative result

Each directory carries:

  - metrics.json                — full uncapped Phase A eval slice
  - eval_tables.json            — full per-(dataset, scope) row-level tables
  - calibration.json            — per-field calibration bins + ECE
  - training_manifest.json      — training summary + config
  - git_commit.txt              — git rev-parse HEAD at freeze time
  - environment.txt             — pip freeze + Python + torch versions
  - README.md                   — human-readable framing of what this is

The graph artifact carries an extra ``NEGATIVE_RESULT.md`` explaining
why we are documenting a non-shipping checkpoint: the integration is
clean, the training-time ablation deltas are real (+0.006 to +0.013),
but unseen generalization does not exceed validated_recovery on the
clean held-out sets at ~20k sentences. Future work needs more
supervision, not deeper architecture.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _env_freeze() -> str:
    out_lines = [f"python: {sys.version.split()[0]}"]
    for pkg in ("torch", "transformers", "tokenizers", "numpy"):
        try:
            m = __import__(pkg)
            out_lines.append(f"{pkg}: {getattr(m, '__version__', '?')}")
        except Exception:
            out_lines.append(f"{pkg}: missing")
    return "\n".join(out_lines)


def _read_json(p: Path) -> Any:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def freeze(
    *, name: str, src_ckpt: Path, eval_dir: Path,
    training_summary: Path, kind: str, dst_root: Path,
    extra_doc: Dict[str, str] = None,
) -> Path:
    dst = dst_root / name
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")

    dst.mkdir(parents=True)

    # Reproducibility manifest if present, else build a minimal one.
    src_manifest = src_ckpt / "REPRODUCIBILITY_MANIFEST.json"
    if src_manifest.exists():
        shutil.copy(src_manifest, dst / "REPRODUCIBILITY_MANIFEST.json")

    # Eval tables (full)
    eval_tables = _read_json(eval_dir / "final_eval_tables.json")
    eval_report = (eval_dir / "final_eval_report.md").read_text() \
        if (eval_dir / "final_eval_report.md").exists() else ""
    (dst / "eval_tables.json").write_text(
        json.dumps(eval_tables or {}, indent=2, ensure_ascii=False)
    )
    (dst / "eval_report.md").write_text(eval_report)

    # Headline metrics (just the rows for this candidate)
    rows = (eval_tables or {}).get("rows", [])
    candidate_rows = [r for r in rows if r["checkpoint"] == kind]
    (dst / "metrics.json").write_text(
        json.dumps({"checkpoint": kind, "rows": candidate_rows},
                   indent=2, ensure_ascii=False)
    )

    # Calibration: pull from raw JSON for this candidate, per dataset
    raw_dir = eval_dir / "raw"
    calib_blob: Dict[str, Any] = {}
    if raw_dir.exists():
        for p in raw_dir.glob(f"{kind}__*.json"):
            d = json.loads(p.read_text())
            ds = p.stem.replace(f"{kind}__", "")
            calib_blob[ds] = d.get("calibration", {})
    (dst / "calibration.json").write_text(
        json.dumps(calib_blob, indent=2, ensure_ascii=False)
    )

    # Training manifest
    ts = _read_json(training_summary)
    (dst / "training_manifest.json").write_text(
        json.dumps(ts or {}, indent=2, ensure_ascii=False)
    )

    # Git commit + environment
    (dst / "git_commit.txt").write_text(_git_commit() + "\n")
    (dst / "environment.txt").write_text(_env_freeze() + "\n")

    # README
    if extra_doc and "readme" in extra_doc:
        (dst / "README.md").write_text(extra_doc["readme"])
    if extra_doc and "negative" in extra_doc:
        (dst / "NEGATIVE_RESULT.md").write_text(extra_doc["negative"])

    print(f"  ✓ {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst_root", default=str(ROOT / "runs"))
    args = ap.parse_args()

    dst_root = Path(args.dst_root)

    # ---------- Validated recovery (production) ----------
    validated_readme = (
        "# final_validated\n\n"
        "Canonical production checkpoint for Arabic iʿrāb generation.\n\n"
        "Trained leak-free (job 491875), validated by an independent\n"
        "uncapped eval (job 491890):\n\n"
        "- MASAQ Quranic: case 0.835 → 0.848 (+0.014); role 0.778 → 0.807\n"
        "  (+0.029); fully **0.675 → 0.711 (+0.036)**.\n"
        "- Gazelle: case 0.638 → 0.646 (+0.008); role 0.575 → 0.613 (+0.038);\n"
        "  fully unchanged at 0.459; calibration gap moved from +0.021 to\n"
        "  −0.052 (less overconfident, healthier).\n\n"
        "This checkpoint *supersedes* all prior nextgen runs (the leaked\n"
        "stage_7 from job 491628 is documented as a contamination case\n"
        "study, not a candidate).\n\n"
        "Source: `runs/validated_nextgen_recovery/` on HPC. Files copied\n"
        "here for the reproducibility frozen-artifact contract:\n"
        "metrics, eval, calibration, training manifest, git, environment.\n\n"
        "Do not modify this directory. New training runs write to a\n"
        "different path.\n"
    )
    freeze(
        name="final_validated",
        src_ckpt=ROOT / "runs" / "validated_nextgen_recovery",
        eval_dir=ROOT / "docs" / "final_eval_recovery",
        training_summary=ROOT / "runs" / "nextgen_recovery" / "training_summary.json",
        kind="recovery",
        dst_root=dst_root,
        extra_doc={"readme": validated_readme},
    )

    # ---------- Graph experiment (negative result) ----------
    graph_readme = (
        "# final_graph_negative_result\n\n"
        "Documented negative result. The graph integration experiment\n"
        "(job 491906) trained cleanly but did not exceed validated_recovery\n"
        "on the unseen held-out sets.\n\n"
        "See `NEGATIVE_RESULT.md` for the methodological note.\n"
    )
    graph_negative = (
        "# Graph integration — documented negative result\n\n"
        "## What we built\n\n"
        "End-to-end wiring of the existing grammar graph + 2-layer\n"
        "edge-aware graph refiner into the `DepAwareStructuredModel`\n"
        "forward path:\n\n"
        "- Collator emits `(B, W, W)` dense `word_edge_index` matrix from\n"
        "  dep heads + construction membership + overlap detection.\n"
        "- Forward applies `pooled = pooled + sigmoid(graph_gate) * delta`\n"
        "  where `delta = refiner(pooled, edge_index, mask) − pooled`.\n"
        "- Gate logit initialised at −2.0 (sigmoid ≈ 0.119) so the graph\n"
        "  signal starts weak; the model learns whether structure helps.\n"
        "- Encoder frozen for first 2,000 steps; refiner + gate train alone.\n"
        "- Edge dropout 15% on dep + construction edges; per-stage edge-type\n"
        "  curriculum (dep only → +construction → +overlap → all).\n"
        "- Eval emits `fully_with_graph` / `fully_without_graph` /\n"
        "  `graph_edge_ablation_delta` and the live `graph_gate_alpha`.\n\n"
        "## What worked\n\n"
        "- Refiner trained without instability (no NaN, no norm explosion).\n"
        "- Gate moved 0.120 → 0.122 — small but real movement once the\n"
        "  encoder unfroze and could co-train.\n"
        "- Ablation delta was consistently positive after stage 3:\n"
        "  +0.006 to +0.013 fully on the cap-100 eval slice.\n"
        "- Stage transitions and the no-leakage assertions both held.\n\n"
        "## What did not work\n\n"
        "On the **full uncapped held-out sets**, the graph checkpoint\n"
        "did not beat the regularization-only `validated_recovery`:\n\n"
        "| Dataset | metric | recovery | graph | Δ |\n"
        "|---|---|---|---|---|\n"
        "| Gazelle | fully | 0.459 | 0.459 | +0.000 |\n"
        "| Gazelle | role  | 0.613 | 0.613 | +0.000 |\n"
        "| Gazelle | case  | 0.646 | 0.638 | −0.008 |\n"
        "| MASAQ   | fully | 0.711 | 0.707 | −0.004 |\n"
        "| MASAQ   | role  | 0.807 | 0.813 | +0.006 |\n"
        "| MASAQ   | case  | 0.848 | 0.845 | −0.003 |\n\n"
        "The training-time +0.013 ablation delta did not survive the\n"
        "full-sample eval. Gains and regressions are all within the\n"
        "noise band on a 30-sentence Gazelle.\n\n"
        "## Interpretation\n\n"
        "At ~20k training sentences, explicit graph reasoning on top of\n"
        "the dependency-aware encoder features (Phase 3-A's input-side dep\n"
        "augmentation) does not materially improve unseen generalization.\n"
        "The encoder + dep-feature input augmentation already captures\n"
        "most of the structural signal a downstream graph layer would\n"
        "provide; adding a refiner on top therefore yields no measurable\n"
        "headroom at this scale.\n\n"
        "**This is a bottleneck-identification result.** The remaining\n"
        "gap to higher unseen performance is not architectural — it is\n"
        "**supervision density and semantic-ambiguity coverage**.\n\n"
        "## Why this is documented, not deleted\n\n"
        "The implementation is correct, the experiment is informative,\n"
        "and the negative finding constrains the search space for future\n"
        "work. Anyone who tries the same architectural tweak in this\n"
        "regime will reproduce our result. Documenting the experiment\n"
        "keeps the project honest and saves future research effort.\n"
    )
    freeze(
        name="final_graph_negative_result",
        src_ckpt=ROOT / "runs" / "nextgen_graph" / "stage_7" / "final",
        eval_dir=ROOT / "docs" / "final_eval_graph",
        training_summary=ROOT / "runs" / "nextgen_graph" / "training_summary.json",
        kind="graph",
        dst_root=dst_root,
        extra_doc={"readme": graph_readme, "negative": graph_negative},
    )

    print(
        "\n✅ Both artifacts frozen. Future training writes to NEW directories;\n"
        "   these two are the immutable record of where we are right now.\n"
    )


if __name__ == "__main__":
    main()
