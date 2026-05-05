"""Compute role-F1 ONLY on the subset of words where gold has an extractable role.

Avoids the verb/sub-component bug where the extractor scrapes role terms from
non-head morphemes. Reports macro-F1 with bootstrap CIs.

Usage:
    python scripts/role_subset_scoring.py
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

from irab_tashkeel.evaluation.structural import extract


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[ً-ْٰ]", "", s)
    return re.sub(r"[^ء-ي]+", "", s)


def collect_pairs(path: Path) -> List[Tuple[str | None, str | None]]:
    """Return list of (gold_role, pred_role) for every aligned word."""
    out: List[Tuple[str | None, str | None]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pred_by_word = {_norm(p.get("word", "")): p.get("irab", "")
                            for p in (row.get("pred") or [])
                            if isinstance(p, dict) and p.get("word")}
            for g in (row.get("gold") or []):
                ge = extract(g.get("irab", ""))
                pe = extract(pred_by_word.get(_norm(g.get("word", "")), ""))
                out.append((ge.role, pe.role))
    return out


def macro_f1(pairs: List[Tuple[str | None, str | None]]) -> float:
    """Macro-F1 over ALL gold-role classes, ignoring None vs None."""
    classes = set(g for g, p in pairs if g is not None) | set(p for g, p in pairs if p is not None)
    if not classes:
        return 0.0
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def bootstrap_macro_f1(pairs: List[Tuple[str | None, str | None]],
                       B: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap macro-F1 → (point, lo95, hi95)."""
    point = macro_f1(pairs)
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        boots.append(macro_f1(sample))
    boots.sort()
    return (point, boots[int(0.025 * B)], boots[int(0.975 * B) - 1])


def subset_score(path: Path) -> dict:
    """Subset role-F1: only count words where gold has an extractable role."""
    pairs = collect_pairs(path)
    n_total = len(pairs)
    subset = [(g, p) for g, p in pairs if g is not None]
    n_subset = len(subset)
    pt, lo, hi = bootstrap_macro_f1(subset, B=1000)
    pt_full, lo_full, hi_full = bootstrap_macro_f1(pairs, B=1000)
    return {
        "n_total": n_total,
        "n_subset": n_subset,
        "subset_pct": n_subset / n_total * 100 if n_total else 0,
        "role_f1_subset": pt * 100,
        "role_f1_subset_ci": (lo * 100, hi * 100),
        "role_f1_full": pt_full * 100,
        "role_f1_full_ci": (lo_full * 100, hi_full * 100),
    }


GAZELLE_PATHS = {
    "stanza":        "runs/baseline_eval_stanza/stanza.predictions.jsonl",
    "qwen_rag":      "runs/baseline_eval_openweight/openweight.predictions.jsonl",
    "haiku_zero":    "runs/baseline_eval/claude_zero.predictions.jsonl",
    "haiku_rag":     "runs/baseline_eval_v2/claude_rag.predictions.jsonl",
    "sonnet_zero":   "runs/baseline_eval_sonnet/claude_zero.predictions.jsonl",
    "sonnet_rag":    "runs/baseline_eval_sonnet/claude_rag.predictions.jsonl",
    "arat5_base":    "runs/baseline_eval_arat5_irab/arat5_irab.predictions.jsonl",
    "mt5_base":      "runs/baseline_eval_mt5_gazelle/arat5_irab.predictions.jsonl",
    "aragpt2_large": "runs/baseline_eval_aragpt2_gazelle/aragpt2_irab.predictions.jsonl",
    "acegpt_13b":    "runs/baseline_eval_acegpt13b_gazelle/acegpt_irab.predictions.jsonl",
}

MASAQ_PATHS = {
    "stanza":        "runs/baseline_eval_masaq_stanza/stanza.predictions.jsonl",
    "arat5_base":    "runs/baseline_eval_masaq_arat5/arat5_irab.predictions.jsonl",
    "mt5_base":      "runs/baseline_eval_masaq_mt5/arat5_irab.predictions.jsonl",
    "aragpt2_large": "runs/baseline_eval_masaq_aragpt2/aragpt2_irab.predictions.jsonl",
    "acegpt_13b":    "runs/baseline_eval_masaq_acegpt13b/acegpt_irab.predictions.jsonl",
    "sonnet_rag":    "runs/baseline_eval_masaq_sonnet/claude_rag.predictions.jsonl",
}


def main():
    print("=" * 78)
    print("GAZELLE — role-F1 subset (gold has role) vs full (all words)")
    print("=" * 78)
    print(f"{'system':14}  {'n_subset/total':>16}  {'subset role-F1':>22}  {'full role-F1':>22}")
    for sys, path in GAZELLE_PATHS.items():
        if not Path(path).exists():
            print(f"  {sys:14}  MISSING")
            continue
        r = subset_score(Path(path))
        print(f"{sys:14}  {r['n_subset']:>4}/{r['n_total']:<4} ({r['subset_pct']:4.1f}%)  "
              f"{r['role_f1_subset']:5.1f}% [{r['role_f1_subset_ci'][0]:4.1f}, {r['role_f1_subset_ci'][1]:5.1f}]  "
              f"{r['role_f1_full']:5.1f}% [{r['role_f1_full_ci'][0]:4.1f}, {r['role_f1_full_ci'][1]:5.1f}]")
    print()
    print("=" * 78)
    print("MASAQ — role-F1 subset (gold has role) vs full (all words)")
    print("=" * 78)
    print(f"{'system':14}  {'n_subset/total':>16}  {'subset role-F1':>22}  {'full role-F1':>22}")
    for sys, path in MASAQ_PATHS.items():
        if not Path(path).exists():
            print(f"  {sys:14}  MISSING")
            continue
        # Only score if file is reasonably complete (>50 verses)
        n_lines = sum(1 for _ in open(path, encoding="utf-8"))
        if n_lines < 50:
            print(f"  {sys:14}  SKIPPED (only {n_lines} verses, eval still running)")
            continue
        r = subset_score(Path(path))
        print(f"{sys:14}  {r['n_subset']:>4}/{r['n_total']:<5} ({r['subset_pct']:4.1f}%)  "
              f"{r['role_f1_subset']:5.1f}% [{r['role_f1_subset_ci'][0]:4.1f}, {r['role_f1_subset_ci'][1]:5.1f}]  "
              f"{r['role_f1_full']:5.1f}% [{r['role_f1_full_ci'][0]:4.1f}, {r['role_f1_full_ci'][1]:5.1f}]")


if __name__ == "__main__":
    main()
