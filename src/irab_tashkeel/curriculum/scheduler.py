"""Curriculum scheduler — orchestrates 7-stage training across all stages.

Reads schema_v2 sentences once, builds per-stage pools, returns a
:class:`StratifiedSampler` for the active stage, and advances
stages based on :class:`GateResult` from the trainer.

Typical flow::

    sched = CurriculumScheduler.from_corpus(sentences)
    for batch in sched.iter_batches(batch_size=32):
        loss = train_step(batch)
        if step % eval_interval == 0:
            metrics = run_eval(model, dev_set)
            gate = sched.advance_or_continue(metrics, current_stage_steps)
            if gate.decision is GateDecision.ADVANCE:
                checkpoint(model, sched.current_stage_id)
                if sched.is_done(): break

The trainer never directly looks at the stage configs — it asks
the scheduler for the next batch and the scheduler handles the
mixing + filtering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from ..data_v2.schema_v2 import Sentence
from .config import DEFAULT_STAGES, StageConfig, all_stages
from .gates import GateDecision, GateResult, evaluate_gate
from .sampler import StagePool, StratifiedSampler, build_stage_pool


@dataclass
class ScheduleState:
    """Persistent state of the schedule (checkpointable)."""
    active_stage_id:     int = 1
    steps_in_stage:      int = 0
    history:             List[Dict] = field(default_factory=list)


class CurriculumScheduler:
    """Orchestrate training across the 7-stage curriculum."""

    def __init__(
        self,
        pools: Dict[int, StagePool],
        configs: Dict[int, StageConfig] = None,
        state: Optional[ScheduleState] = None,
        seed: int = 0,
    ):
        self.pools = pools
        self.configs = configs or DEFAULT_STAGES
        self.state = state or ScheduleState()
        self.seed = seed

    # -- factory methods -----------------------------------------------------

    @classmethod
    def from_corpus(
        cls, sentences: List[Sentence],
        configs: Dict[int, StageConfig] = None, seed: int = 0,
    ) -> "CurriculumScheduler":
        """Build a scheduler from a flat list of schema_v2 sentences."""
        configs = configs or DEFAULT_STAGES
        pools = {sid: build_stage_pool(sentences, cfg)
                 for sid, cfg in configs.items()}
        return cls(pools=pools, configs=configs, seed=seed)

    # -- queries -------------------------------------------------------------

    @property
    def current_stage_id(self) -> int:
        return self.state.active_stage_id

    @property
    def current_config(self) -> StageConfig:
        return self.configs[self.current_stage_id]

    @property
    def current_pool(self) -> StagePool:
        return self.pools[self.current_stage_id]

    def earlier_pools(self) -> List[StagePool]:
        return [self.pools[sid] for sid in sorted(self.configs.keys())
                if sid < self.current_stage_id]

    def is_done(self) -> bool:
        return self.current_stage_id > max(self.configs.keys())

    def stage_summary(self) -> Dict:
        out = {}
        for sid, pool in self.pools.items():
            cfg = self.configs[sid]
            out[sid] = {
                "name": cfg.name,
                "n_eligible": pool.n,
                "by_source": {src: len(v) for src, v in pool.by_source.items()},
                "active": sid == self.current_stage_id,
            }
        return out

    # -- batch iteration -----------------------------------------------------

    def make_sampler(self, batch_size: int = 32) -> StratifiedSampler:
        """Return a sampler for the current stage."""
        return StratifiedSampler(
            current_pool=self.current_pool,
            earlier_pools=self.earlier_pools(),
            config=self.current_config,
            batch_size=batch_size,
            seed=self.seed + self.current_stage_id,
        )

    def iter_batches(self, batch_size: int = 32) -> Iterator[List[Sentence]]:
        """Yield batches indefinitely from the current stage.

        The trainer should call :func:`advance_or_continue` periodically
        and break out of this loop when ``is_done()``.
        """
        sampler = self.make_sampler(batch_size=batch_size)
        while not self.is_done():
            batch = sampler.sample_batch()
            self.state.steps_in_stage += 1
            yield batch

    # -- gate decision -------------------------------------------------------

    def advance_or_continue(self, metrics: Dict[str, float]) -> GateResult:
        """Evaluate the current stage's gate and possibly advance."""
        cfg = self.current_config
        result = evaluate_gate(cfg, metrics, self.state.steps_in_stage)
        # log
        self.state.history.append({
            "stage_id": cfg.stage_id, "stage_name": cfg.name,
            "decision": result.decision.value, "reason": result.reason,
            "measured": result.measured, "threshold": result.threshold,
            "steps": result.steps,
        })
        if result.decision in (GateDecision.ADVANCE,
                                GateDecision.TIMEOUT_ADVANCE):
            self.state.active_stage_id += 1
            self.state.steps_in_stage = 0
        return result

    # -- (de)serialisation ---------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "active_stage_id": self.state.active_stage_id,
            "steps_in_stage": self.state.steps_in_stage,
            "history": list(self.state.history),
        }

    def load_state_dict(self, d: Dict) -> None:
        self.state = ScheduleState(
            active_stage_id=int(d.get("active_stage_id", 1)),
            steps_in_stage=int(d.get("steps_in_stage", 0)),
            history=list(d.get("history", [])),
        )
