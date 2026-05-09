"""Tests for training_v2 — dataset, collator, loss (no torch needed for some tests)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from irab_tashkeel.data_v2.schema_v2 import (
    AnnotationCompleteness, CurriculumMetadata, LabelTag, Sentence,
    SentenceMetadata, Token,
)
from irab_tashkeel.training_v2 import (
    HeadLossWeights, SchemaV2Dataset, TrainerConfig, MORPH_VOCABS,
    MORPH_TO_ID,
)


def _make_sent(*, n_tokens: int = 3) -> Sentence:
    s = Sentence(
        raw_text=" ".join(["w"] * n_tokens),
        tokens=[
            Token(
                index=i, surface="w",
                case=LabelTag(value="raf", source="gold_human"),
                role=LabelTag(value="mubtada", source="gold_human"),
                marker=LabelTag(value="damma_visible", source="gold_human"),
                pos=LabelTag(value="noun", source="gold_human"),
            )
            for i in range(n_tokens)
        ],
        metadata=SentenceMetadata(
            source="test", domain="msa_news",
        ),
        completeness=AnnotationCompleteness(fields_complete_pct=1.0),
        curriculum=CurriculumMetadata(difficulty_level=1, sentence_length_tokens=n_tokens),
    )
    return s


# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------

def test_trainer_config_default_head_weights():
    cfg = TrainerConfig()
    assert cfg.head_loss_weights.case == 1.0
    assert cfg.head_loss_weights.morph == 0.5


def test_trainer_config_stage_overrides():
    cfg = TrainerConfig()
    s1 = cfg.head_weights_for_stage(1)
    s5 = cfg.head_weights_for_stage(5)
    # stage 1 emphasises morph, stage 5 emphasises role
    assert s1.morph > s1.case
    assert s5.role > s5.morph


def test_trainer_config_unknown_stage_falls_back_to_default():
    cfg = TrainerConfig()
    s_unknown = cfg.head_weights_for_stage(99)
    assert s_unknown.case == cfg.head_loss_weights.case


# ---------------------------------------------------------------------------
# SchemaV2Dataset
# ---------------------------------------------------------------------------

def test_dataset_length():
    ds = SchemaV2Dataset([_make_sent() for _ in range(5)])
    assert len(ds) == 5


def test_dataset_item_shape():
    ds = SchemaV2Dataset([_make_sent(n_tokens=4)])
    item = ds[0]
    assert item["raw_text"]
    assert len(item["words"]) == 4
    assert len(item["case"]) == 4
    assert len(item["role"]) == 4
    assert "morph" in item
    assert all(axis in item["morph"] for axis in MORPH_VOCABS)


def test_dataset_metadata_fields():
    ds = SchemaV2Dataset([_make_sent()])
    item = ds[0]
    assert item["metadata"]["domain"] == "msa_news"
    assert item["completeness"]["fields_complete_pct"] == 1.0
    assert item["curriculum"]["difficulty_level"] == 1


# ---------------------------------------------------------------------------
# HeadLossWeights
# ---------------------------------------------------------------------------

def test_head_loss_weights_as_dict():
    w = HeadLossWeights(case=1.0, role=2.0, marker=0.5, pos=0.5,
                        morph=1.0, dep=0.0)
    d = w.as_dict()
    assert d["case"] == 1.0
    assert d["role"] == 2.0
    assert d["dep"] == 0.0


# ---------------------------------------------------------------------------
# Morph vocab consistency
# ---------------------------------------------------------------------------

def test_morph_vocab_round_trip():
    for axis, vocab in MORPH_VOCABS.items():
        for v in vocab:
            assert MORPH_TO_ID[axis][v] >= 0


def test_morph_unknown_value_via_label_id():
    from irab_tashkeel.training_v2.collator import _label_id, IGNORE
    for axis, vocab in MORPH_VOCABS.items():
        # known value
        assert _label_id(vocab[0], MORPH_TO_ID[axis]) == 0
        # unknown value
        assert _label_id("nonexistent_value", MORPH_TO_ID[axis]) == IGNORE
        # None
        assert _label_id(None, MORPH_TO_ID[axis]) == IGNORE
