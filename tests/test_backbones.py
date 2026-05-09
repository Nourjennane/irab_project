"""Tests for the backbones registry."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from irab_tashkeel.backbones import (
    BackboneSpec, REGISTRY, all_backbones,
    by_arch_family, by_pretraining, get_backbone,
)


def test_registry_has_at_least_ten():
    assert len(REGISTRY) >= 10


def test_registry_includes_frozen_baseline_reference():
    spec = get_backbone("arat5v2-base")
    assert spec.hf_model_id == "UBC-NLP/AraT5v2-base-1024"
    assert spec.arch_family == "t5_enc_dec"


def test_unknown_backbone_raises():
    with pytest.raises(KeyError):
        get_backbone("nonexistent_backbone_xyz")


def test_all_backbones_have_unique_ids():
    ids = [b.backbone_id for b in all_backbones()]
    assert len(ids) == len(set(ids))


def test_all_backbones_have_hf_model_ids():
    for b in all_backbones():
        assert b.hf_model_id, f"{b.backbone_id} missing hf_model_id"


def test_by_arch_family_t5():
    t5_models = by_arch_family("t5_enc_dec")
    assert any(b.backbone_id == "arat5v2-base" for b in t5_models)
    assert any(b.backbone_id == "mt5-base" for b in t5_models)


def test_by_pretraining_classical():
    cls = by_pretraining("classical")
    assert any(b.backbone_id == "camelbert-ca" for b in cls)


def test_arch_diversity():
    """Registry should span at least 4 architecture families."""
    families = {b.arch_family for b in all_backbones()}
    assert len(families) >= 4


def test_pretraining_diversity():
    """Registry should span at least 4 pretraining types."""
    types = {b.arabic_pretraining for b in all_backbones()}
    assert len(types) >= 4
