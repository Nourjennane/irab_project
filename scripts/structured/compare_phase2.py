"""Phase 2 — mechanism comparison aggregator.

Reads the per-variant ``runs/phase2_eval_<variant>_<jobid>/`` directories
and produces a single mechanism-comparison report covering:

* **Gate metrics** (case / role-F1 / marker / fully) on Gazelle, both
  heads-only and heads+constraints.
* **Cross-register stability**: same metrics on MASAQ; the delta is
  reported as "gazelle-MASAQ retention ratio".
* **Rare-role behaviour**: macro-F1 over the 9 lowest-support v3 classes.
* **Head-role behaviour**: macro-F1 over the 8 highest-support v3 classes.
* **Calibration gap**: mean role conf on correct − mean role conf on
  wrong. A larger gap means the head's confidence is informative; a
  shrinking gap means the head is confidently wrong on rare classes.
* **Long-tail collapse count**: how many v3 classes have F1 < 50%.
* **Conditioning activity**: for FiLM, the L2 norm of (γ − 1) and the
  L2 norm of β at the saved checkpoint. Tells us whether FiLM actually
  *learned* to deviate from identity vs sat at no-op. Same idea for
  additive (norm of W_b) and concat-embed (norm of MLP final layer).
* **Anomaly flags** (per the user's brief): rows where role-F1 moves
  but ``fully`` doesn't, or vice versa, or where MASAQ retention is
  unusually high/low. Surfaces patterns that may matter for the
  staged-reasoning architecture.

Usage:
    python scripts/structured/compare_phase2.py \\
        --runs runs/phase2_eval_phase2_v3_film_NNN \\
                runs/phase2_eval_phase2_v3_additive_MMM \\
                ...

Writes a markdown report to ``runs/phase2_comparison.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _label_from_dir(run_dir: Path) -> str:
    """Extract a human-readable variant label from ``runs/phase2_eval_phase2_v3_film_491116``."""
    name = run_dir.name
    name = name.replace("phase2_eval_", "").replace("phase2_", "")
    return name


def _gate_row(headline: dict, baseline_key: str) -> Dict[str, float]:
    """Pull (case_acc, role_f1, marker_acc, fully) from run_baselines summary.json."""
    entries = headline.get("baselines", []) or headline.get("results", [])
    for entry in entries:
        if entry.get("name") == baseline_key:
            return {
                "case": entry.get("case_accuracy", 0.0) * 100,
                "role_f1": entry.get("role_macro_f1", entry.get("role_f1", 0.0)) * 100,
                "marker": entry.get("marker_accuracy", 0.0) * 100,
                "fully": entry.get("fully_correct", 0.0) * 100,
                "n": entry.get("n_judgments", entry.get("n", 0)),
            }
    return {"case": 0.0, "role_f1": 0.0, "marker": 0.0, "fully": 0.0, "n": 0}


def _stress_row(stream4: dict) -> Dict[str, float]:
    if not stream4:
        return {}
    s = stream4.get("stress_table", {})
    return {
        "rare_macro_f1": s.get("rare_role_macro_f1", 0.0) * 100,
        "head_macro_f1": s.get("head_role_macro_f1", 0.0) * 100,
        "long_tail_collapse": s.get("long_tail_collapse_count", 0),
        "calib_correct": s.get("calibration_correct", 0.0),
        "calib_wrong": s.get("calibration_wrong", 0.0),
        "calib_gap": s.get("calibration_gap", 0.0),
    }


def _conditioning_activity(model_dir: Path) -> Dict[str, float]:
    """Inspect the saved state-dict for conditioning-module weight norms.

    For FiLM:  gamma_proj.bias - 1 (deviation from identity) and beta_proj norm.
    For additive: bias_proj weight norm.
    For concat-embed: mlp final-layer weight norm.
    """
    if not model_dir.exists():
        return {}
    sd_path = model_dir / "pytorch_model.bin"
    if not sd_path.exists():
        return {}
    try:
        import torch
        sd = torch.load(sd_path, map_location="cpu", weights_only=True)
    except Exception as e:
        return {"_load_error": str(e)}
    out = {}
    # FiLM
    if "conditioning.gamma_proj.weight" in sd:
        gw = sd["conditioning.gamma_proj.weight"]
        gb = sd["conditioning.gamma_proj.bias"]
        bw = sd["conditioning.beta_proj.weight"]
        bb = sd["conditioning.beta_proj.bias"]
        out["mechanism"] = "film"
        out["gamma_weight_norm"] = float(gw.norm().item())
        out["gamma_bias_dev_from_1"] = float((gb - 1.0).norm().item())
        out["beta_weight_norm"] = float(bw.norm().item())
        out["beta_bias_norm"] = float(bb.norm().item())
    elif "conditioning.bias_proj.weight" in sd:
        out["mechanism"] = "additive"
        out["bias_proj_weight_norm"] = float(sd["conditioning.bias_proj.weight"].norm().item())
    elif "conditioning.mlp.0.weight" in sd:
        out["mechanism"] = "concat_embed"
        out["mlp_final_weight_norm"] = float(sd["conditioning.mlp.2.weight"].norm().item())
        out["mlp_final_bias_norm"] = float(sd["conditioning.mlp.2.bias"].norm().item())
    else:
        out["mechanism"] = "none"
    return out


def _morph_macro(morph_summary: dict) -> Optional[float]:
    if not morph_summary:
        return None
    # Common shape: {"per_feature": {f: {"acc": ..., "f1": ...}}, "macro_acc": ...}
    if "macro_acc" in morph_summary:
        return float(morph_summary["macro_acc"]) * 100
    if "macro_f1" in morph_summary:
        return float(morph_summary["macro_f1"]) * 100
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="paths to runs/phase2_eval_<variant>_<jobid>/ directories")
    ap.add_argument("--out", default="runs/phase2_comparison.md")
    ap.add_argument("--phase1_baseline", default=None,
                    help="optional path to runs/phase1_morph_eval_*/ for the baseline row")
    args = ap.parse_args()

    rows: List[dict] = []
    if args.phase1_baseline:
        p1 = Path(args.phase1_baseline)
        if p1.exists():
            gh = _load_json(p1 / "gazelle" / "summary.json") or {}
            mh = _load_json(p1 / "masaq" / "summary.json") or {}
            sg = _load_json(p1 / "gazelle" / "phase4a_summary.json") or {}
            rows.append({
                "variant": "phase1 (baseline)",
                "model_dir": "",
                "gazelle_heads": _gate_row(gh, "structured_v1"),
                "gazelle_constr": _gate_row(gh, "structured_v1_constrained"),
                "masaq_heads": _gate_row(mh, "structured_v1"),
                "stress_gz": _stress_row(sg),
                "morph_macro": _morph_macro(_load_json(p1 / "morph" / "morphology_summary.json")),
                "conditioning": {"mechanism": "none"},
            })

    for run_path in args.runs:
        run = Path(run_path)
        gh = _load_json(run / "gazelle" / "summary.json") or {}
        mh = _load_json(run / "masaq" / "summary.json") or {}
        sg = _load_json(run / "gazelle" / "phase4a_summary.json") or {}
        sm = _load_json(run / "masaq" / "phase4a_summary.json") or {}
        morph = _load_json(run / "morph" / "morphology_summary.json")

        # Locate the model dir from provenance, falling back to a sibling guess
        variant = _label_from_dir(run)
        model_dir = None
        for sib in run.parent.iterdir():
            if not sib.is_dir():
                continue
            n = sib.name
            if "eval" in n:
                continue
            if variant in n or n.replace("phase2_", "") == variant:
                final = sib / "final"
                if final.exists():
                    model_dir = final
                    break

        rows.append({
            "variant": variant,
            "model_dir": str(model_dir) if model_dir else "",
            "gazelle_heads": _gate_row(gh, "structured_v1"),
            "gazelle_constr": _gate_row(gh, "structured_v1_constrained"),
            "masaq_heads": _gate_row(mh, "structured_v1"),
            "stress_gz": _stress_row(sg),
            "stress_masaq": _stress_row(sm),
            "morph_macro": _morph_macro(morph),
            "conditioning": _conditioning_activity(model_dir) if model_dir else {},
        })

    # ---- Build report ----
    lines = []
    lines.append("# Phase 2 — mechanism comparison\n")
    lines.append("Decision gate (vs Phase 1 baseline `53.7 / 42.3 / 41.0 / 19.4`):")
    lines.append("- case ≥ 53.0")
    lines.append("- role-F1 ≥ 43.0")
    lines.append("- fully ≥ 19.4\n")

    # Headline gate table
    lines.append("## 1. Gazelle gate metrics (heads only)\n")
    lines.append("| Variant | case | role-F1 | marker | fully | n |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        g = r["gazelle_heads"]
        lines.append(f"| {r['variant']} | {g['case']:.1f} | {g['role_f1']:.1f} | "
                     f"{g['marker']:.1f} | {g['fully']:.1f} | {g['n']} |")

    lines.append("\n## 2. Gazelle gate metrics (heads + 4 constraints)\n")
    lines.append("| Variant | case | role-F1 | marker | fully |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        g = r["gazelle_constr"]
        lines.append(f"| {r['variant']} | {g['case']:.1f} | {g['role_f1']:.1f} | "
                     f"{g['marker']:.1f} | {g['fully']:.1f} |")

    lines.append("\n## 3. MASAQ cross-register (heads only)\n")
    lines.append("| Variant | case | role-F1 | marker | fully |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        m = r["masaq_heads"]
        lines.append(f"| {r['variant']} | {m['case']:.1f} | {m['role_f1']:.1f} | "
                     f"{m['marker']:.1f} | {m['fully']:.1f} |")

    lines.append("\n## 4. Stress table (Gazelle, heads only)\n")
    lines.append("| Variant | rare-F1 | head-F1 | long-tail | calib correct | calib wrong | calib gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        s = r["stress_gz"]
        if not s:
            lines.append(f"| {r['variant']} | — | — | — | — | — | — |")
            continue
        lines.append(f"| {r['variant']} | {s['rare_macro_f1']:.1f} | {s['head_macro_f1']:.1f} | "
                     f"{s['long_tail_collapse']} | {s['calib_correct']:.3f} | "
                     f"{s['calib_wrong']:.3f} | {s['calib_gap']:+.3f} |")

    lines.append("\n## 5. Conditioning activity (did the module move off identity?)\n")
    lines.append("| Variant | mechanism | activity |")
    lines.append("|---|---|---|")
    for r in rows:
        c = r.get("conditioning", {})
        mech = c.get("mechanism", "none")
        activity = "n/a"
        if mech == "film":
            activity = (
                f"|γ−1|={c.get('gamma_bias_dev_from_1', 0):.2f} "
                f"|β|={c.get('beta_bias_norm', 0):.2f} "
                f"|W_γ|={c.get('gamma_weight_norm', 0):.2f}"
            )
        elif mech == "additive":
            activity = f"|W_b|={c.get('bias_proj_weight_norm', 0):.2f}"
        elif mech == "concat_embed":
            activity = (
                f"|MLP_final_W|={c.get('mlp_final_weight_norm', 0):.2f} "
                f"|MLP_final_b|={c.get('mlp_final_bias_norm', 0):.2f}"
            )
        lines.append(f"| {r['variant']} | {mech} | {activity} |")

    lines.append("\n## 6. Morphology accuracy (UD-PADT test) — has Phase 2 hurt morph heads?\n")
    lines.append("| Variant | morph macro | Δ vs Phase 1 |")
    lines.append("|---|---:|---:|")
    p1_macro = next((r["morph_macro"] for r in rows if r["variant"].startswith("phase1") and r["morph_macro"] is not None), None)
    for r in rows:
        m = r["morph_macro"]
        delta = (f"{m - p1_macro:+.2f}" if (m is not None and p1_macro is not None) else "—")
        lines.append(f"| {r['variant']} | {m if m is not None else '—'} | {delta} |")

    # Anomaly flags
    lines.append("\n## 7. Anomaly flags — patterns that may matter\n")
    flags = []
    p1_g = next((r["gazelle_heads"] for r in rows if r["variant"].startswith("phase1")), None)
    for r in rows:
        if r["variant"].startswith("phase1"):
            continue
        g = r["gazelle_heads"]
        if p1_g:
            d_role = g["role_f1"] - p1_g["role_f1"]
            d_fully = g["fully"] - p1_g["fully"]
            d_case = g["case"] - p1_g["case"]
            if abs(d_role) >= 1.0 and abs(d_fully) < 0.5:
                flags.append(f"- **{r['variant']}**: role-F1 moved {d_role:+.1f} pp but "
                             f"fully unchanged ({d_fully:+.1f}) — role discrimination "
                             f"improves without composing into the structured tuple. "
                             f"May indicate role wins are on cells where case or marker "
                             f"is already wrong, so the joint gain is hidden.")
            if abs(d_fully) >= 1.0 and abs(d_role) < 0.5:
                flags.append(f"- **{r['variant']}**: fully moved {d_fully:+.1f} pp but "
                             f"role-F1 unchanged ({d_role:+.1f}) — fully recovery is "
                             f"happening downstream of the role head. Likely a "
                             f"case + marker compose-without-role-shift pattern.")
            if abs(d_case) >= 2.0 and d_role < 0:
                flags.append(f"- **{r['variant']}**: case {d_case:+.1f}, role-F1 {d_role:+.1f} — "
                             f"case wins traded for role losses. The conditioning "
                             f"module is reallocating encoder capacity, not adding it.")
        # MASAQ stability
        m_row = r["masaq_heads"]
        if g["role_f1"] > 0 and m_row["role_f1"] > 0:
            retention = m_row["role_f1"] / g["role_f1"]
            if retention < 0.30:
                flags.append(f"- **{r['variant']}**: MASAQ role-F1 retention only {retention*100:.0f}% — "
                             f"register transfer broken. The conditioning module may have "
                             f"learned MSA-specific morph→role couplings.")
            elif retention > 0.45:
                flags.append(f"- **{r['variant']}**: MASAQ role-F1 retention {retention*100:.0f}% — "
                             f"unusually high. Conditioning may be acting more register-"
                             f"agnostic than parallel multi-task heads.")

    if flags:
        lines.extend(flags)
    else:
        lines.append("(no patterns flagged automatically — check tables manually)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print()
    # Echo gate summary to stdout for quick CLI inspection
    for r in rows:
        g = r["gazelle_heads"]
        print(f"{r['variant']:>40s}  Gazelle: case={g['case']:5.1f}  role-F1={g['role_f1']:5.1f}  "
              f"marker={g['marker']:5.1f}  fully={g['fully']:5.1f}")


if __name__ == "__main__":
    main()
