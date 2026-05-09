"""Integration tests for the full data_v2 → eval_v2 pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from irab_tashkeel.data_v2.constructions.detector import (
    detect_constructions, detect_constructions_pass,
    overlap_summary, clause_consistency_check, FAMILIES,
)
from irab_tashkeel.data_v2.loaders import distill2, gazelle, ud_padt, masaq  # noqa: F401
from irab_tashkeel.data_v2.loaders.gazelle import GazelleLoader
from irab_tashkeel.data_v2.loaders.ud_padt import UdPadtLoader
from irab_tashkeel.data_v2.loaders.masaq import MasaqLoader
from irab_tashkeel.data_v2.metadata import difficulty
from irab_tashkeel.data_v2.splitter import (
    SplitConfig, stratified_split, write_split, _stratum_key,
)
from irab_tashkeel.data_v2.schema_v2 import Sentence, Token, LabelTag, Construction
from irab_tashkeel.eval_v2 import (
    SentencePrediction, TokenPrediction, aggregate_outcomes,
    construction_detection_metrics, extract_outcomes, overall_metrics,
    stratified_metrics,
)


# ---------------------------------------------------------------------------
# Construction-detector
# ---------------------------------------------------------------------------

def test_detector_kana_simple():
    s = Sentence(
        raw_text="كان الطالب مجتهداً",
        tokens=[
            Token(index=0, surface="كان"),
            Token(index=1, surface="الطالب"),
            Token(index=2, surface="مجتهداً"),
        ],
    )
    cs = detect_constructions(s)
    fams = {c.family for c in cs}
    assert "kana_sisters" in fams
    kana = [c for c in cs if c.family == "kana_sisters"][0]
    assert kana.subgroup == "kana_completion"
    assert kana.head_idx == 0
    assert kana.particle_surface == "كان"


def test_detector_idafa_via_role():
    s = Sentence(
        raw_text="باب البيت",
        tokens=[
            Token(index=0, surface="باب"),
            Token(index=1, surface="البيت",
                  role=LabelTag(value="mudaaf_ilayh", source="gold_human")),
        ],
    )
    cs = detect_constructions(s)
    fams = {c.family for c in cs}
    assert "idafa" in fams


def test_detector_idafa_multi():
    s = Sentence(
        raw_text="باب بيت زيد",
        tokens=[
            Token(index=0, surface="باب"),
            Token(index=1, surface="بيت",
                  role=LabelTag(value="mudaaf_ilayh", source="gold_human")),
            Token(index=2, surface="زيد",
                  role=LabelTag(value="mudaaf_ilayh", source="gold_human")),
        ],
    )
    cs = detect_constructions(s)
    fams = {c.family for c in cs}
    assert "idafa" in fams
    assert "idafa_multi" in fams


def test_detector_overlap_marks_ambiguity():
    """When two constructions cover the same token, both should be
    flagged with ambiguity_score and listed as alternative analyses
    of each other.
    """
    s = Sentence(
        raw_text="إن الطالب مجتهد",
        tokens=[
            Token(index=0, surface="إن"),       # inna
            Token(index=1, surface="الطالب",
                  role=LabelTag(value="mudaaf_ilayh", source="gold_human")),
            Token(index=2, surface="مجتهد"),
        ],
    )
    cs = detect_constructions(s)
    # inna span [0..3] + idafa span [0..2] overlap at token 1
    # Both should have ambiguity_score > 0
    assert any(c.ambiguity_score > 0 for c in cs), \
        "expected at least one construction to flag ambiguity from overlap"


def test_detector_consistency_check_clean():
    s = Sentence(
        raw_text="كان الطالب مجتهداً",
        tokens=[
            Token(index=0, surface="كان"),
            Token(index=1, surface="الطالب"),
            Token(index=2, surface="مجتهداً"),
        ],
    )
    s.constructions = detect_constructions(s)
    issues = clause_consistency_check(s)
    assert issues == []


# ---------------------------------------------------------------------------
# UD-PADT loader
# ---------------------------------------------------------------------------

def test_ud_padt_loads_dev():
    loader = UdPadtLoader(root=str(ROOT), split="dev")
    sents = loader.load_all()
    assert len(sents) > 0
    s = sents[0]
    assert s.metadata.source == "ud_padt"
    assert s.metadata.annotation_quality == "gold_treebank"
    assert s.completeness.has_morph
    assert s.completeness.has_dep
    # UD-PADT has no iʿrāb role/marker
    assert not s.completeness.has_role
    assert not s.completeness.has_marker
    # First-token gold-quality dep label
    assert any(t.dep_label.is_present and t.dep_label.confidence == 1.0
               for t in s.tokens)


# ---------------------------------------------------------------------------
# MASAQ loader
# ---------------------------------------------------------------------------

def test_masaq_loads_with_quranic_metadata():
    loader = MasaqLoader(root=str(ROOT))
    sents = loader.load_all()
    assert len(sents) > 0
    s = sents[0]
    assert s.metadata.domain == "quranic"
    assert s.metadata.annotation_quality == "gold_human"
    # source_id should look like "<sura>:<verse>"
    assert ":" in s.metadata.source_id


# ---------------------------------------------------------------------------
# Stratified splitter
# ---------------------------------------------------------------------------

def test_splitter_deterministic(tmp_path):
    loader = GazelleLoader(root=str(ROOT))
    sents = loader.load_all()
    detect_constructions_pass(sents)
    difficulty.populate_all(sents)

    cfg = SplitConfig(train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15,
                      min_stratum_for_eval=2)
    r1 = stratified_split(sents, cfg)
    r2 = stratified_split(sents, cfg)
    assert [s.sentence_id for s in r1.train] == [s.sentence_id for s in r2.train]
    assert [s.sentence_id for s in r1.dev]   == [s.sentence_id for s in r2.dev]
    assert [s.sentence_id for s in r1.test]  == [s.sentence_id for s in r2.test]


def test_splitter_writes_diagnostics(tmp_path):
    loader = GazelleLoader(root=str(ROOT))
    sents = loader.load_all()
    detect_constructions_pass(sents)
    difficulty.populate_all(sents)

    cfg = SplitConfig(min_stratum_for_eval=2)
    result = stratified_split(sents, cfg)
    counts = write_split(tmp_path, result)
    assert counts["train"] + counts["dev"] + counts["test"] == len(sents)
    assert (tmp_path / "split_report.md").exists()
    assert (tmp_path / "coverage_report.md").exists()
    assert (tmp_path / "stratum_assignments.json").exists()


# ---------------------------------------------------------------------------
# eval_v2
# ---------------------------------------------------------------------------

def _make_sentence_with_predictions():
    """Tiny sentence + perfect predictions for sanity tests."""
    s = Sentence(
        raw_text="كان الطالب مجتهداً",
        tokens=[
            Token(index=0, surface="كان",
                  case=LabelTag(value="mabni", source="gold_human"),
                  role=LabelTag(value="fil", source="gold_human"),
                  marker=LabelTag(value="fath_short", source="gold_human")),
            Token(index=1, surface="الطالب",
                  case=LabelTag(value="raf", source="gold_human"),
                  role=LabelTag(value="ism_kana", source="gold_human"),
                  marker=LabelTag(value="damma_visible", source="gold_human")),
            Token(index=2, surface="مجتهداً",
                  case=LabelTag(value="nasb", source="gold_human"),
                  role=LabelTag(value="khabar_kana", source="gold_human"),
                  marker=LabelTag(value="tanween_fath", source="gold_human")),
        ],
    )
    s.completeness.fields_complete_pct = 1.0
    pred = SentencePrediction(
        sentence_id=s.sentence_id,
        tokens=[
            TokenPrediction(sentence_id=s.sentence_id, token_index=0,
                            case="mabni", role="fil", marker="fath_short",
                            case_conf=0.95, role_conf=0.95, marker_conf=0.9),
            TokenPrediction(sentence_id=s.sentence_id, token_index=1,
                            case="raf", role="ism_kana", marker="damma_visible",
                            case_conf=0.92, role_conf=0.88, marker_conf=0.91),
            TokenPrediction(sentence_id=s.sentence_id, token_index=2,
                            case="nasb", role="khabar_kana", marker="tanween_fath",
                            case_conf=0.85, role_conf=0.83, marker_conf=0.82),
        ],
    )
    return s, pred


def test_eval_v2_perfect_predictions():
    s, pred = _make_sentence_with_predictions()
    out = overall_metrics([s], [pred], fully_observable_only=True)
    o = out["overall"]
    assert o["case_acc"] == pytest.approx(1.0)
    assert o["role_acc"] == pytest.approx(1.0)
    assert o["marker_em"] == pytest.approx(1.0)
    assert o["fully"] == pytest.approx(1.0)


def test_eval_v2_one_wrong():
    s, pred = _make_sentence_with_predictions()
    pred.tokens[2].role = "khabar"   # wrong: should be khabar_kana
    pred.tokens[2].role_conf = 0.4
    out = overall_metrics([s], [pred])
    o = out["overall"]
    # 2/3 role correct (metrics rounded to 4 dp in aggregator)
    assert o["role_acc"] == pytest.approx(2/3, abs=1e-3)
    # case still 100%
    assert o["case_acc"] == pytest.approx(1.0)
    # fully drops to 2/3
    assert o["fully"] == pytest.approx(2/3, abs=1e-3)


def test_eval_v2_stratified_no_crash():
    s, pred = _make_sentence_with_predictions()
    outcomes = extract_outcomes([s], [pred])
    out = stratified_metrics(outcomes, axes=("domain", "difficulty"))
    assert "domain" in out
    assert "difficulty" in out
