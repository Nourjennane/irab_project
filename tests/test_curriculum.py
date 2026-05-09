"""Tests for the Step 7 curriculum framework."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from irab_tashkeel.curriculum import (
    CurriculumScheduler, DEFAULT_STAGES, GateDecision, GateResult,
    StageConfig, StagePool, StratifiedSampler, all_stages,
    build_stage_pool, evaluate_gate, get_stage, stage_eligibility,
)
from irab_tashkeel.data_v2.schema_v2 import (
    AnnotationCompleteness, CurriculumMetadata, Domain, Sentence,
    SentenceMetadata, Token, AnnotationQuality,
)


# ---------------------------------------------------------------------------
# StageConfig
# ---------------------------------------------------------------------------

def test_default_stages_are_seven():
    assert len(DEFAULT_STAGES) == 7
    assert sorted(DEFAULT_STAGES.keys()) == [1, 2, 3, 4, 5, 6, 7]


def test_each_stage_has_gate_threshold():
    for cfg in all_stages():
        assert cfg.gate_metric
        assert cfg.gate_threshold > 0


def test_stage_difficulty_range_increases():
    for cfg in all_stages():
        lo, hi = cfg.difficulty_range
        assert lo == 1, f"stage {cfg.stage_id} should allow stage 1 samples"
        # Higher stages should allow up to higher difficulty
        if cfg.stage_id > 1:
            assert hi >= cfg.stage_id - 1


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def _make_sent(*, difficulty=1, semantic=0, completeness=1.0,
               source="distill_v2", domain="msa_news",
               n_constructions=0, ambiguity=0.0) -> Sentence:
    s = Sentence(
        raw_text="x",
        tokens=[Token(index=0, surface="x")],
        metadata=SentenceMetadata(
            source=source, domain=domain,
            annotation_quality=AnnotationQuality.SILVER_LLM_DISTILL.value,
        ),
        completeness=AnnotationCompleteness(fields_complete_pct=completeness),
        curriculum=CurriculumMetadata(
            difficulty_level=difficulty, semantic_pressure_score=semantic,
            ambiguity_score=ambiguity, sentence_length_tokens=1,
        ),
    )
    return s


def test_eligibility_difficulty_filter():
    cfg = get_stage(1)
    s_low = _make_sent(difficulty=1)
    s_high = _make_sent(difficulty=5)
    assert stage_eligibility(s_low, cfg)
    assert not stage_eligibility(s_high, cfg)


def test_eligibility_source_filter():
    cfg = get_stage(1)
    s_ok = _make_sent(source="distill_v2")
    s_bad = _make_sent(source="some_other_source")
    assert stage_eligibility(s_ok, cfg)
    assert not stage_eligibility(s_bad, cfg)


def test_eligibility_max_semantic_pressure():
    cfg = get_stage(2)              # max_semantic_pressure=1
    s_ok = _make_sent(difficulty=2, semantic=1, source="distill_v2")
    s_bad = _make_sent(difficulty=2, semantic=2, source="distill_v2")
    assert stage_eligibility(s_ok, cfg)
    assert not stage_eligibility(s_bad, cfg)


def test_eligibility_min_semantic_pressure_for_stage_5():
    cfg = get_stage(5)              # min_semantic_pressure=2
    s_ok = _make_sent(difficulty=5, semantic=2)
    s_bad = _make_sent(difficulty=5, semantic=1)
    assert stage_eligibility(s_ok, cfg)
    assert not stage_eligibility(s_bad, cfg)


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

def test_build_stage_pool_buckets_by_source():
    sents = [
        _make_sent(source="distill_v2"),
        _make_sent(source="distill_v2"),
        _make_sent(source="ud_padt_train"),
    ]
    pool = build_stage_pool(sents, get_stage(1))
    assert pool.n == 3
    assert "distill_v2" in pool.by_source
    assert len(pool.by_source["distill_v2"]) == 2


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def test_sampler_stratified_draw():
    cfg = get_stage(1)
    pool = build_stage_pool(
        [_make_sent(source="distill_v2") for _ in range(10)] +
        [_make_sent(source="ud_padt_train") for _ in range(5)],
        cfg,
    )
    sampler = StratifiedSampler(pool, [], cfg, batch_size=4, seed=0)
    batch = sampler.sample_batch()
    assert len(batch) == 4


def test_sampler_rehearsal_proportions():
    cfg2 = get_stage(2)
    cur_pool = build_stage_pool(
        [_make_sent(difficulty=2, semantic=1, source="distill_v2") for _ in range(20)],
        cfg2,
    )
    earlier = [build_stage_pool(
        [_make_sent(source="distill_v2") for _ in range(10)],
        get_stage(1),
    )]
    sampler = StratifiedSampler(cur_pool, earlier, cfg2, batch_size=10, seed=0)
    batch = sampler.sample_batch()
    assert len(batch) == 10


def test_sampler_deterministic():
    cfg = get_stage(1)
    pool = build_stage_pool(
        [_make_sent(source="distill_v2") for _ in range(20)], cfg,
    )
    s1 = StratifiedSampler(pool, [], cfg, batch_size=4, seed=42).sample_batch()
    s2 = StratifiedSampler(pool, [], cfg, batch_size=4, seed=42).sample_batch()
    assert [s.sentence_id for s in s1] == [s.sentence_id for s in s2]


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def test_gate_continue_under_target_steps():
    cfg = get_stage(1)
    res = evaluate_gate(cfg, {"morph_macro_f1": 0.95}, steps=100)
    assert res.decision is GateDecision.CONTINUE


def test_gate_advance_when_threshold_met():
    cfg = get_stage(1)
    res = evaluate_gate(cfg, {"morph_macro_f1": 0.95}, steps=cfg.target_steps)
    assert res.decision is GateDecision.ADVANCE


def test_gate_continue_when_threshold_not_met():
    cfg = get_stage(1)
    res = evaluate_gate(cfg, {"morph_macro_f1": 0.50}, steps=cfg.target_steps)
    assert res.decision is GateDecision.CONTINUE


def test_gate_timeout_advance_at_max_steps():
    cfg = get_stage(1)
    res = evaluate_gate(cfg, {"morph_macro_f1": 0.20}, steps=cfg.max_steps + 1)
    assert res.decision is GateDecision.TIMEOUT_ADVANCE


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def test_scheduler_starts_at_stage_1():
    sched = CurriculumScheduler.from_corpus(
        [_make_sent(source="distill_v2") for _ in range(10)]
    )
    assert sched.current_stage_id == 1


def test_scheduler_advances_on_gate_pass():
    sents = [_make_sent(source="distill_v2") for _ in range(10)]
    sched = CurriculumScheduler.from_corpus(sents)
    # Advance forcibly via a gate that meets threshold + steps
    sched.state.steps_in_stage = sched.current_config.target_steps
    res = sched.advance_or_continue({"morph_macro_f1": 1.0})
    assert res.decision is GateDecision.ADVANCE
    assert sched.current_stage_id == 2


def test_scheduler_history_logged():
    sents = [_make_sent(source="distill_v2") for _ in range(10)]
    sched = CurriculumScheduler.from_corpus(sents)
    sched.state.steps_in_stage = sched.current_config.target_steps
    sched.advance_or_continue({"morph_macro_f1": 1.0})
    assert len(sched.state.history) == 1
    assert sched.state.history[0]["decision"] == "advance"


def test_scheduler_is_done_after_all_stages():
    sents = [_make_sent(source="distill_v2") for _ in range(10)]
    sched = CurriculumScheduler.from_corpus(sents)
    for sid in range(1, 8):
        cfg = sched.current_config
        sched.state.steps_in_stage = cfg.target_steps
        sched.advance_or_continue({cfg.gate_metric: 1.0})
    assert sched.is_done()


def test_scheduler_serialization():
    sents = [_make_sent(source="distill_v2") for _ in range(10)]
    sched = CurriculumScheduler.from_corpus(sents)
    sched.state.steps_in_stage = 100
    d = sched.to_dict()

    sched2 = CurriculumScheduler.from_corpus(sents)
    sched2.load_state_dict(d)
    assert sched2.state.active_stage_id == sched.state.active_stage_id
    assert sched2.state.steps_in_stage == 100
