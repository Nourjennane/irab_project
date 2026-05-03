"""Two-sample bootstrap for the cross-register role-F1 drop.

The Gazelle and MASAQ subsets are distinct items (different word distributions,
different registers, different annotators) — paired bootstrap doesn't apply.
Instead we use a two-sample bootstrap that resamples each side independently
B times and reports the 95% CI of the difference.
"""
from __future__ import annotations

import random
from pathlib import Path
from scripts.role_subset_scoring import collect_pairs, macro_f1


def two_sample_bootstrap_diff(pairs_a, pairs_b, B: int = 1000, seed: int = 42):
    """Δ macro-F1 = F1(a) − F1(b), with 95% bootstrap CI.

    For each of B iterations: resample with replacement from both samples
    independently, compute F1 on each, take difference. Return percentile
    CI on the differences.
    """
    rng = random.Random(seed)
    f1_a_point = macro_f1(pairs_a)
    f1_b_point = macro_f1(pairs_b)
    diff_point = (f1_a_point - f1_b_point) * 100
    diffs = []
    na, nb = len(pairs_a), len(pairs_b)
    for _ in range(B):
        sa = [pairs_a[rng.randrange(na)] for _ in range(na)]
        sb = [pairs_b[rng.randrange(nb)] for _ in range(nb)]
        diffs.append((macro_f1(sa) - macro_f1(sb)) * 100)
    diffs.sort()
    return diff_point, diffs[int(0.025 * B)], diffs[int(0.975 * B) - 1]


SYSTEMS = {
    "stanza":     ("runs/baseline_eval_stanza/stanza.predictions.jsonl",
                   "runs/baseline_eval_masaq_stanza/stanza.predictions.jsonl"),
    "mt5_base":   ("runs/baseline_eval_mt5_gazelle/arat5_irab.predictions.jsonl",
                   "runs/baseline_eval_masaq_mt5/arat5_irab.predictions.jsonl"),
    "arat5_base": ("runs/baseline_eval_arat5_irab/arat5_irab.predictions.jsonl",
                   "runs/baseline_eval_masaq_arat5/arat5_irab.predictions.jsonl"),
    "sonnet_rag": ("runs/baseline_eval_sonnet/claude_rag.predictions.jsonl",
                   "runs/baseline_eval_masaq_sonnet/claude_rag.predictions.jsonl"),
}


def subset_pairs(path: Path):
    return [(g, p) for g, p in collect_pairs(path) if g is not None]


def main():
    print(f"{'system':14}  {'F1 Gazelle':>14}  {'F1 MASAQ':>13}  {'Δ (Gazelle − MASAQ)':>28}")
    for sys, (gz_path, ms_path) in SYSTEMS.items():
        if not (Path(gz_path).exists() and Path(ms_path).exists()):
            print(f"{sys:14}  MISSING (skip)")
            continue
        n_lines = sum(1 for _ in open(ms_path))
        if n_lines < 50:
            print(f"{sys:14}  MASAQ eval still running ({n_lines} verses); skip")
            continue
        gz = subset_pairs(Path(gz_path))
        ms = subset_pairs(Path(ms_path))
        f1_gz = macro_f1(gz) * 100
        f1_ms = macro_f1(ms) * 100
        d, lo, hi = two_sample_bootstrap_diff(gz, ms, B=1000)
        sig = "★" if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else " "
        print(f"{sys:14}  {f1_gz:5.1f} (n={len(gz):>3})  {f1_ms:5.1f} (n={len(ms):>4})  "
              f"{d:+6.1f} pp [{lo:+5.1f}, {hi:+6.1f}] {sig}")


if __name__ == "__main__":
    main()
