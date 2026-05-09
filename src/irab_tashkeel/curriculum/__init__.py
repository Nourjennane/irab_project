"""curriculum — Step 7 staged-learning framework.

Public API:

    config.StageConfig
    config.DEFAULT_STAGES, all_stages(), get_stage(stage_id)

    sampler.StagePool
    sampler.StratifiedSampler
    sampler.build_stage_pool(sentences, cfg)
    sampler.stage_eligibility(sentence, cfg)

    gates.GateDecision, GateResult, evaluate_gate

    scheduler.CurriculumScheduler
    scheduler.ScheduleState

See ``src/irab_tashkeel/curriculum/README.md`` for the conceptual
mapping of stage 1..7 to what the model learns.
"""
from .config import (
    DEFAULT_STAGES, StageConfig, all_stages, get_stage,
)
from .gates import GateDecision, GateResult, evaluate_gate
from .sampler import (
    StagePool, StratifiedSampler, build_stage_pool, stage_eligibility,
)
from .scheduler import CurriculumScheduler, ScheduleState

__all__ = [
    "DEFAULT_STAGES", "StageConfig", "all_stages", "get_stage",
    "GateDecision", "GateResult", "evaluate_gate",
    "StagePool", "StratifiedSampler", "build_stage_pool", "stage_eligibility",
    "CurriculumScheduler", "ScheduleState",
]
