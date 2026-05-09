"""Stage transition gates.

A stage advances to the next stage only when:

  1. The training has run at least ``cfg.target_steps`` steps, AND
  2. The stage's ``gate_metric`` on the dev split meets ``gate_threshold``,
     OR
  3. The training has run ``cfg.max_steps`` (timeout — advance anyway).

Gates are decoupled from training to keep the orchestration testable.
The trainer hands the gate a metric snapshot; the gate returns
"advance" / "continue" / "fail" plus a reason string.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from .config import StageConfig


class GateDecision(str, Enum):
    ADVANCE   = "advance"
    CONTINUE  = "continue"
    TIMEOUT_ADVANCE = "timeout_advance"
    FAIL      = "fail"


@dataclass
class GateResult:
    decision: GateDecision
    reason:   str
    measured: Optional[float] = None
    threshold: Optional[float] = None
    steps:    int = 0


def evaluate_gate(
    cfg: StageConfig, metrics: Dict[str, float], steps: int,
) -> GateResult:
    """Decide whether to advance from this stage.

    metrics — metric snapshot keyed by ``cfg.gate_metric``
    steps   — training steps taken in this stage so far
    """
    measured = metrics.get(cfg.gate_metric)

    # Hard timeout
    if steps >= cfg.max_steps:
        return GateResult(
            decision=GateDecision.TIMEOUT_ADVANCE,
            reason=f"max_steps={cfg.max_steps} reached",
            measured=measured, threshold=cfg.gate_threshold, steps=steps,
        )

    if steps < cfg.target_steps:
        return GateResult(
            decision=GateDecision.CONTINUE,
            reason=f"under target_steps ({steps}/{cfg.target_steps})",
            measured=measured, threshold=cfg.gate_threshold, steps=steps,
        )

    if measured is None:
        return GateResult(
            decision=GateDecision.CONTINUE,
            reason=f"gate_metric {cfg.gate_metric!r} not in metrics yet",
            steps=steps,
        )

    if measured >= cfg.gate_threshold:
        return GateResult(
            decision=GateDecision.ADVANCE,
            reason=f"{cfg.gate_metric}={measured:.4f} ≥ {cfg.gate_threshold:.4f}",
            measured=measured, threshold=cfg.gate_threshold, steps=steps,
        )

    return GateResult(
        decision=GateDecision.CONTINUE,
        reason=f"{cfg.gate_metric}={measured:.4f} < {cfg.gate_threshold:.4f}; "
               f"keep training",
        measured=measured, threshold=cfg.gate_threshold, steps=steps,
    )
