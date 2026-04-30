"""Statistical helpers for the i'rāb evaluation.

With n=134 word judgments on Gazelle, every metric difference under ~7-8
points is plausibly noise. This module provides:

  - bootstrap CIs (1000 resamples, BCa-percentile) on per-system metrics
  - paired bootstrap CIs on system-vs-system deltas (resample matched items)
  - McNemar's exact test for paired binary outcomes (e.g. fully_correct_word)

All functions consume the prediction JSONL files written by
`evaluation/run_baselines.py` and align word-level judgments by sentence.

Why bootstrap and not just Wilson intervals: role_f1_macro and other corpus-
level metrics are not simple binomial proportions; bootstrap is the only
honest CI for them. We use percentile bootstrap (BCa is overkill at this
sample size).
"""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .structural import IrabAnalysis, extract


_DIAC_RE = re.compile(r"[ً-ْٰ]")


def _normalize_word(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = _DIAC_RE.sub("", s)
    return re.sub(r"[^ء-ي]+", "", s)


# ---------------------------------------------------------------------------
# 1. Load + align prediction JSONLs into per-word judgment rows
# ---------------------------------------------------------------------------
@dataclass
class WordJudgment:
    """One per-word evaluation point, aligned across systems."""
    sent_idx: int             # index of the source Gazelle sentence
    word_idx: int             # word index within that sentence
    gold_word: str
    gold_irab: str
    sys_irab: Dict[str, str]  # system_name -> predicted irab string ("" if missing)


def load_predictions(jsonl_path: Path | str) -> List[Dict]:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_judgments(
    system_paths: Dict[str, Path | str],
) -> List[WordJudgment]:
    """Load every system's predictions JSONL and align by (sentence, word).

    Args:
        system_paths: {system_name: predictions.jsonl_path}

    Returns:
        List of WordJudgment, one per gold word, with predicted irab from
        every system (empty string if a system has no prediction for that word).
    """
    # Pick any one file as the gold source; gold should be identical across.
    first = next(iter(system_paths.values()))
    gold_rows = load_predictions(first)

    # Build per-system word lookups: (sent_idx, normalized_gold_word) -> pred_irab
    sys_lookups: Dict[str, Dict[Tuple[int, str], str]] = {}
    for name, path in system_paths.items():
        rows = load_predictions(path)
        lk: Dict[Tuple[int, str], str] = {}
        for sent_idx, row in enumerate(rows):
            preds = row.get("pred") or []
            for p in preds:
                w = _normalize_word(p.get("word") or "")
                if w:
                    lk[(sent_idx, w)] = (p.get("irab") or "").strip()
        sys_lookups[name] = lk

    out: List[WordJudgment] = []
    for sent_idx, row in enumerate(gold_rows):
        for word_idx, g in enumerate(row.get("gold") or []):
            gw_raw = g.get("word") or ""
            gw_norm = _normalize_word(gw_raw)
            if not gw_norm:
                continue
            sys_irab = {
                name: lk.get((sent_idx, gw_norm), "") for name, lk in sys_lookups.items()
            }
            out.append(WordJudgment(
                sent_idx=sent_idx,
                word_idx=word_idx,
                gold_word=gw_raw,
                gold_irab=(g.get("irab") or "").strip(),
                sys_irab=sys_irab,
            ))
    return out


# ---------------------------------------------------------------------------
# 2. Per-word scoring functions (mapping a judgment to a binary metric)
# ---------------------------------------------------------------------------
def _score_pair(gold_text: str, pred_text: str) -> Dict[str, bool]:
    g = extract(gold_text)
    p = extract(pred_text)
    return {
        "well_formed":   bool(p.well_formed),
        "case":          (g.case is not None and g.case == p.case),
        "pos":           (g.pos is not None and g.pos == p.pos),
        "marker":        (g.marker is not None and g.marker == p.marker),
        "role":          (g.role is not None and g.role == p.role),
        "fully":         (g.case is not None and g.case == p.case
                          and g.role is not None and g.role == p.role
                          and g.marker is not None and g.marker == p.marker),
    }


def per_judgment_scores(
    judgments: Sequence[WordJudgment],
    system: str,
) -> List[Dict[str, bool]]:
    return [_score_pair(j.gold_irab, j.sys_irab.get(system, "")) for j in judgments]


def per_judgment_role_pairs(
    judgments: Sequence[WordJudgment],
    system: str,
) -> List[Tuple[Optional[str], Optional[str]]]:
    """Return (gold_role, pred_role) pairs for macro-F1 computation."""
    out = []
    for j in judgments:
        g = extract(j.gold_irab).role
        p = extract(j.sys_irab.get(system, "")).role
        out.append((g, p))
    return out


def role_f1_macro(role_pairs: Sequence[Tuple[Optional[str], Optional[str]]]) -> float:
    counts = Counter()
    gold_n = Counter()
    pred_n = Counter()
    for g, p in role_pairs:
        gold = g or "<none>"
        pred = p or "<none>"
        counts[(gold, pred)] += 1
        gold_n[gold] += 1
        pred_n[pred] += 1
    f1s = []
    for r in set(gold_n) | set(pred_n):
        if r == "<none>":
            continue
        tp = counts.get((r, r), 0)
        fn = gold_n.get(r, 0) - tp
        fp = pred_n.get(r, 0) - tp
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1s.append(2 * tp / denom)
    return sum(f1s) / len(f1s) if f1s else 0.0


# ---------------------------------------------------------------------------
# 3. Bootstrap CIs
# ---------------------------------------------------------------------------
def bootstrap_proportion(
    flags: Sequence[bool], B: int = 1000, alpha: float = 0.05, seed: int = 42,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for a binary-flag proportion.

    Returns (point_estimate, ci_low, ci_high).
    """
    n = len(flags)
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = sum(flags) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        sample = [flags[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(B * alpha / 2)]
    hi = boots[int(B * (1 - alpha / 2))]
    return (point, lo, hi)


def bootstrap_role_f1(
    role_pairs: Sequence[Tuple[Optional[str], Optional[str]]],
    B: int = 1000, alpha: float = 0.05, seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for macro role-F1 — resample (gold, pred) pairs."""
    n = len(role_pairs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    point = role_f1_macro(role_pairs)
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        sample = [role_pairs[rng.randrange(n)] for _ in range(n)]
        boots.append(role_f1_macro(sample))
    boots.sort()
    lo = boots[int(B * alpha / 2)]
    hi = boots[int(B * (1 - alpha / 2))]
    return (point, lo, hi)


# ---------------------------------------------------------------------------
# 4. Paired comparisons
# ---------------------------------------------------------------------------
def paired_bootstrap_proportion_delta(
    flags_a: Sequence[bool], flags_b: Sequence[bool],
    B: int = 1000, alpha: float = 0.05, seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Paired bootstrap on Δ = mean(B) − mean(A) for matched binary scores.

    Returns (delta, ci_low, ci_high, p_value_approx).

    `p_value_approx` is the fraction of resamples with sign-flip vs the
    point estimate — i.e., a two-sided estimate of P(Δ_resample crosses 0).
    Treat as a soft significance gauge; for hard inference use McNemar.
    """
    if len(flags_a) != len(flags_b):
        raise ValueError("paired comparison requires equal-length flag lists")
    n = len(flags_a)
    if n == 0:
        return (0.0, 0.0, 0.0, 1.0)
    point_a = sum(flags_a) / n
    point_b = sum(flags_b) / n
    delta = point_b - point_a
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        sa = sb = 0
        for _ in range(n):
            i = rng.randrange(n)
            if flags_a[i]: sa += 1
            if flags_b[i]: sb += 1
        boots.append(sb / n - sa / n)
    boots.sort()
    lo = boots[int(B * alpha / 2)]
    hi = boots[int(B * (1 - alpha / 2))]
    if delta >= 0:
        p = sum(1 for d in boots if d <= 0) / B
    else:
        p = sum(1 for d in boots if d >= 0) / B
    return (delta, lo, hi, p * 2)  # two-sided


def mcnemar_exact(flags_a: Sequence[bool], flags_b: Sequence[bool]) -> Tuple[int, int, float]:
    """McNemar's exact test on paired binary outcomes.

    Returns (n_a_only, n_b_only, p_value).

    For each item, B can be (0,0), (1,0), (0,1), (1,1).
    Discordant pairs are (1,0) "a-only" and (0,1) "b-only".
    Under H0 the b/a counts are exchangeable; exact binomial test on the
    discordants gives a small-n-friendly p-value.
    """
    if len(flags_a) != len(flags_b):
        raise ValueError("mcnemar requires equal-length lists")
    n10 = sum(1 for a, b in zip(flags_a, flags_b) if a and not b)
    n01 = sum(1 for a, b in zip(flags_a, flags_b) if b and not a)
    n_disc = n10 + n01
    if n_disc == 0:
        return (n10, n01, 1.0)
    # Two-sided exact binomial: prob of seeing as extreme a split as observed
    k = min(n10, n01)
    # P(K <= k) under Binom(n_disc, 0.5)
    cum = 0.0
    for i in range(k + 1):
        cum += math.comb(n_disc, i) * (0.5 ** n_disc)
    p_two = min(1.0, 2 * cum)
    return (n10, n01, p_two)


# ---------------------------------------------------------------------------
# 5. Convenience: full report for a set of systems + metrics
# ---------------------------------------------------------------------------
@dataclass
class SystemReport:
    name: str
    n: int
    well_formed: Tuple[float, float, float]
    case:        Tuple[float, float, float]
    pos:         Tuple[float, float, float]
    marker:      Tuple[float, float, float]
    role_f1:     Tuple[float, float, float]
    fully:       Tuple[float, float, float]


def system_report(judgments: Sequence[WordJudgment], system: str, B: int = 1000) -> SystemReport:
    scores = per_judgment_scores(judgments, system)
    role_pairs = per_judgment_role_pairs(judgments, system)
    return SystemReport(
        name=system,
        n=len(judgments),
        well_formed=bootstrap_proportion([s["well_formed"] for s in scores], B=B),
        case       =bootstrap_proportion([s["case"] for s in scores], B=B),
        pos        =bootstrap_proportion([s["pos"] for s in scores], B=B),
        marker     =bootstrap_proportion([s["marker"] for s in scores], B=B),
        role_f1    =bootstrap_role_f1(role_pairs, B=B),
        fully      =bootstrap_proportion([s["fully"] for s in scores], B=B),
    )


def pairwise_deltas(
    judgments: Sequence[WordJudgment], a: str, b: str, B: int = 1000,
) -> Dict[str, Dict]:
    """Compute paired bootstrap deltas + McNemar p-values for binary metrics."""
    sa = per_judgment_scores(judgments, a)
    sb = per_judgment_scores(judgments, b)
    out: Dict[str, Dict] = {}
    for metric in ("well_formed", "case", "pos", "marker", "fully"):
        flags_a = [s[metric] for s in sa]
        flags_b = [s[metric] for s in sb]
        delta, lo, hi, p_boot = paired_bootstrap_proportion_delta(flags_a, flags_b, B=B)
        n10, n01, p_mc = mcnemar_exact(flags_a, flags_b)
        out[metric] = {
            "delta": delta, "ci_low": lo, "ci_high": hi,
            "p_paired_boot": p_boot,
            "n_a_only": n10, "n_b_only": n01,
            "p_mcnemar": p_mc,
            "significant_005": (lo > 0 or hi < 0) and p_mc < 0.05,
        }
    return out


def pretty_report(report: SystemReport) -> str:
    def fmt(t):
        return f"{t[0]*100:5.1f}% [{t[1]*100:4.1f}, {t[2]*100:4.1f}]"
    return (
        f"{report.name:30}  n={report.n}  "
        f"well={fmt(report.well_formed)}  "
        f"case={fmt(report.case)}  "
        f"role-F1={fmt(report.role_f1)}  "
        f"marker={fmt(report.marker)}  "
        f"fully={fmt(report.fully)}"
    )


def main():
    """CLI: compute CIs + paired deltas across N system prediction files."""
    import argparse
    p = argparse.ArgumentParser(description="Bootstrap CIs + paired tests")
    p.add_argument("--system", action="append", required=True, metavar="NAME=PATH",
                   help="system label and prediction JSONL, e.g. claude_rag=runs/.../claude_rag.predictions.jsonl")
    p.add_argument("--reference", default=None,
                   help="reference system name; pairwise deltas computed against it")
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--out", type=Path, default=None,
                   help="optional JSON path to dump full report")
    args = p.parse_args()

    system_paths: Dict[str, Path] = {}
    for spec in args.system:
        if "=" not in spec:
            raise SystemExit(f"--system expects NAME=PATH, got {spec}")
        name, path = spec.split("=", 1)
        system_paths[name] = Path(path)

    judgments = build_judgments(system_paths)
    print(f"loaded {len(judgments)} word-level judgments across {len(system_paths)} systems\n")

    reports = []
    for name in system_paths:
        rep = system_report(judgments, name, B=args.B)
        print(pretty_report(rep))
        reports.append(rep)

    if args.reference and args.reference in system_paths:
        print(f"\n=== Paired deltas vs {args.reference} ===")
        ref = args.reference
        for name in system_paths:
            if name == ref:
                continue
            d = pairwise_deltas(judgments, ref, name, B=args.B)
            print(f"\n  {name} − {ref}:")
            for metric in ("well_formed", "case", "marker", "fully"):
                e = d[metric]
                sig = "★" if e["significant_005"] else " "
                print(f"    {metric:14} Δ={e['delta']*100:+5.1f}% "
                      f"[{e['ci_low']*100:+4.1f},{e['ci_high']*100:+4.1f}]  "
                      f"McNemar p={e['p_mcnemar']:.3f}  {sig}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_judgments": len(judgments),
            "systems": {r.name: {
                "n": r.n,
                "well_formed": r.well_formed, "case": r.case, "pos": r.pos,
                "marker": r.marker, "role_f1": r.role_f1, "fully": r.fully,
            } for r in reports},
        }
        if args.reference and args.reference in system_paths:
            payload["pairwise_vs_" + args.reference] = {
                name: pairwise_deltas(judgments, args.reference, name, B=args.B)
                for name in system_paths if name != args.reference
            }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n  wrote full report → {args.out}")


if __name__ == "__main__":
    main()
