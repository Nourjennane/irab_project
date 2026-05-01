"""System discrimination eval — score systems against perturbed gold.

For each system × Gazelle word × gold variant (original, case_flip, role_flip,
marker_mangle), we re-run structural extraction and check whether the
prediction matches the perturbed-gold's case / role / marker fields.

Why: a system that is genuinely doing case discrimination should ALMOST NEVER
match the case-flipped gold on case (because the system's prediction follows
the original case). Conversely, a system that is just emitting a generic
"اسم" string would happen to match either gold variant the same way. The
delta between original-gold-score and perturbed-gold-score, averaged across
the 30 sentences, is therefore a per-dimension discrimination signal.

Outputs:
  runs/discrimination/per_system.json  — { system: {field × variant: score} }
  RESULTS.md addendum                  — summary table

Note: this does NOT add independent statistical-power gains the way an extra
hand-annotated test set would. The same model output is being scored against
multiple gold variants, so observations are not independent. We report this
as a metric-validation / model-discrimination axis, not as a CI tightener.

Usage:
    python -m irab_tashkeel.evaluation.discrimination \\
        --system "stanza=runs/baseline_eval_stanza/stanza.predictions.jsonl" \\
        --system "haiku_rag=runs/baseline_eval_v2/claude_rag.predictions.jsonl" \\
        --system "sonnet_rag=runs/baseline_eval_sonnet/claude_rag.predictions.jsonl" \\
        --perturbed data/perturbed_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .structural import extract


def _norm_word(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[ً-ْٰ]", "", s)
    return re.sub(r"[^ء-ي]+", "", s)


def load_predictions(path: Path) -> Dict[Tuple[str, str], str]:
    """Map (sentence_norm, word_norm) → predicted i'rāb."""
    out: Dict[Tuple[str, str], str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sent = row.get("sentence", "")
            preds = row.get("pred") or []
            for p in preds:
                w = p.get("word", "") if isinstance(p, dict) else ""
                irab = p.get("irab", "") if isinstance(p, dict) else ""
                if w:
                    out[(sent, _norm_word(w))] = irab
    return out


def load_perturbed(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def score(systems: Dict[str, Path], perturbed: Path) -> Dict[str, dict]:
    """Score each system on each perturbation variant; return summary dict."""
    sys_preds = {name: load_predictions(p) for name, p in systems.items()}
    pert_rows = load_perturbed(perturbed)

    # Group perturbed rows by (sentence, word) to align variants
    by_word: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in pert_rows:
        by_word[(r["sentence"], _norm_word(r["word"]))].append(r)

    out: Dict[str, dict] = {}
    for sys_name, preds in sys_preds.items():
        # for each variant, accumulate {field: (n, n_match)}
        per_variant: Dict[str, Dict[str, Tuple[int, int]]] = {}
        for variant in ("none", "case", "role", "marker"):
            per_variant[variant] = {f: [0, 0] for f in ("case", "role", "marker", "fully")}
        for (sent, w_norm), variants in by_word.items():
            pred_irab = preds.get((sent, w_norm), "")
            if not pred_irab:
                continue   # word not predicted by this system — skip
            p_ext = extract(pred_irab)
            pc, pr, pm = p_ext.case, p_ext.role, p_ext.marker
            for v in variants:
                # NOTE: in perturb.py records, `extracted_pred_*` is the extraction
                # from the (possibly-perturbed) gold variant — which is what we want
                # to score the system's prediction against. `extracted_gold_*` holds
                # the original-pre-perturbation gold extraction.
                gc, gr, gm = v["extracted_pred_case"], v["extracted_pred_role"], v["extracted_pred_marker"]
                # When gold field is None (extractor couldn't extract from gold either),
                # the comparison is undefined — skip that field for that record.
                bucket = per_variant[v["corrupted_field"]]
                for fld_name, gold_val, pred_val in (
                    ("case", gc, pc), ("role", gr, pr), ("marker", gm, pm),
                ):
                    if gold_val is None:
                        continue
                    bucket[fld_name][0] += 1
                    if gold_val == pred_val:
                        bucket[fld_name][1] += 1
                # Fully = case ∧ role ∧ marker all defined and matching
                if gc is not None and gr is not None and gm is not None:
                    bucket["fully"][0] += 1
                    if gc == pc and gr == pr and gm == pm:
                        bucket["fully"][1] += 1
        # Convert to rates
        rates: Dict[str, Dict[str, dict]] = {}
        for variant, fields in per_variant.items():
            rates[variant] = {}
            for fld, (n, k) in fields.items():
                rates[variant][fld] = {
                    "n": n,
                    "match": k,
                    "rate_pct": (k / n * 100) if n else 0.0,
                }
        out[sys_name] = rates
    return out


def pretty(table: Dict[str, dict], systems: List[str]) -> str:
    lines = []
    lines.append("\n  System discrimination on perturbed gold (rate of match against the variant):")
    lines.append("    'none' = original gold (matches the headline number)")
    lines.append("    'case'/'role'/'marker' = perturbed gold; lower-is-better here means "
                 "the system disagrees with the wrong-on-purpose variant.\n")
    head = f"  {'system':18}  {'variant':8}  {'case':>8}  {'role':>8}  {'marker':>8}  {'fully':>8}"
    lines.append(head)
    for s in systems:
        for v in ("none", "case", "role", "marker"):
            row = table[s][v]
            cells = " ".join(
                f"{row[f]['rate_pct']:>6.1f}%(n={row[f]['n']})"
                for f in ("case", "role", "marker", "fully")
            )
            lines.append(f"  {s:18}  {v:8}  {cells}")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="System discrimination eval on perturbed gold")
    p.add_argument("--system", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--perturbed", type=Path, default=Path("data/perturbed_eval.jsonl"))
    p.add_argument("--out", type=Path, default=Path("runs/discrimination/per_system.json"))
    args = p.parse_args()

    systems: Dict[str, Path] = {}
    for spec in args.system:
        name, path = spec.split("=", 1)
        systems[name] = Path(path)

    table = score(systems, args.perturbed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    print(pretty(table, list(systems.keys())))
    print(f"\n  wrote → {args.out}")


if __name__ == "__main__":
    main()
