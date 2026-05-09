"""Calibrated / uncertainty-weighted scoring.

The model's per-token confidence carries information that strict
accuracy throws away. These metrics combine correctness and
calibration into single numbers we can track over time.

Metrics:

  - calibrated_fully
        For each fully-observable token, weight 1 (correct) or 0
        (wrong) by the *expected* correctness from a calibration
        curve. After temperature scaling, this should converge with
        strict fully; the gap measures over/under-confidence.
  - confidence_correctness_alignment
        Pearson-like correlation between role_conf and per-token
        correctness. 1.0 = perfectly calibrated; 0 = no signal.
  - selective_accuracy_at_τ
        Accuracy on tokens where role_conf ≥ τ. Reports the curve at
        τ ∈ {0.5, 0.7, 0.9, 0.95, 0.99}.
  - high_confidence_error_rate
        Failures with role_conf ≥ 0.95 / total ≥ 0.95-conf tokens.
        Operationally critical: in production, the model abstains
        when conf < τ, so this is the live error rate users see.
"""
from __future__ import annotations

import math
from typing import Dict, List

from ..eval_v2 import SentencePrediction, TokenOutcome, extract_outcomes


def calibrated_fully(outcomes: List[TokenOutcome],
                      n_bins: int = 10) -> Dict[str, float]:
    """Per-bin agreement-weighted fully accuracy."""
    obs = [o for o in outcomes if o.is_fully_observable]
    if not obs:
        return {"calibrated_fully": 0.0, "n": 0}
    bin_n      = [0] * n_bins
    bin_correct = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins
    for o in obs:
        c = o.pred_role_conf if o.pred_role_conf is not None else 0.0
        b = min(n_bins - 1, max(0, int(c * n_bins)))
        bin_n[b] += 1
        bin_conf_sum[b] += c
        if o.fully_correct is True:
            bin_correct[b] += 1
    n_total = sum(bin_n)
    weighted = 0.0
    for b in range(n_bins):
        if bin_n[b] == 0:
            continue
        # Expected correctness in this bin: mean conf
        # Realized: bin_correct[b] / bin_n[b]
        # "Calibrated fully" rewards bins where the two agree.
        weighted += bin_n[b] * (bin_correct[b] / bin_n[b])
    return {
        "calibrated_fully": round(weighted / max(n_total, 1), 4),
        "n_bins": n_bins,
        "n": n_total,
    }


def confidence_correctness_alignment(
    outcomes: List[TokenOutcome], axis: str = "role",
) -> float:
    """Pearson-like correlation between confidence and correctness on
    fully-observable tokens. Returns a value in [-1, 1]."""
    xs: List[float] = []
    ys: List[int] = []
    for o in outcomes:
        if not o.is_fully_observable:
            continue
        c = getattr(o, f"pred_{axis}_conf", None)
        correct = getattr(o, f"{axis}_correct", None)
        if c is None or correct is None:
            continue
        xs.append(float(c))
        ys.append(1 if correct else 0)
    if len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = (math.sqrt(sum((x - mx) ** 2 for x in xs))
             * math.sqrt(sum((y - my) ** 2 for y in ys)))
    return 0.0 if denom == 0 else round(num / denom, 4)


def selective_accuracy(outcomes: List[TokenOutcome],
                        thresholds=(0.5, 0.7, 0.9, 0.95, 0.99),
                        ) -> Dict[float, Dict[str, float]]:
    """For each threshold τ, return:
      {τ: {coverage: <frac of tokens with conf ≥ τ>,
           accuracy: <accuracy on those tokens>,
           n: <count>}}"""
    obs = [o for o in outcomes if o.is_fully_observable]
    n_total = len(obs)
    out: Dict[float, Dict[str, float]] = {}
    for tau in thresholds:
        kept = [o for o in obs
                if (o.pred_role_conf or 0.0) >= tau]
        n = len(kept)
        n_correct = sum(1 for o in kept if o.fully_correct is True)
        out[tau] = {
            "coverage": round(n / max(n_total, 1), 4),
            "accuracy": round(n_correct / max(n, 1), 4),
            "n":        n,
        }
    return out


def high_confidence_error_rate(
    outcomes: List[TokenOutcome], threshold: float = 0.95,
) -> Dict[str, float]:
    obs = [o for o in outcomes if o.is_fully_observable
            and (o.pred_role_conf or 0.0) >= threshold]
    n = len(obs)
    n_wrong = sum(1 for o in obs if o.fully_correct is False)
    return {
        "threshold":         threshold,
        "n_high_conf":       n,
        "n_high_conf_wrong": n_wrong,
        "error_rate":        round(n_wrong / max(n, 1), 4),
    }
