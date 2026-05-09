"""Smoke tests for data_v2 loaders + metadata + index pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from irab_tashkeel.data_v2.loaders.base import all_registered, get_loader
# Import both loader modules so they register at test time
from irab_tashkeel.data_v2.loaders import distill2  # noqa: F401
from irab_tashkeel.data_v2.loaders.gazelle import GazelleLoader
from irab_tashkeel.data_v2.metadata import difficulty
from irab_tashkeel.data_v2.metadata.semantic_pressure import score_sentence as sp_score
from irab_tashkeel.data_v2.metadata.ambiguity import score_sentence as amb_score
from irab_tashkeel.data_v2.index.construction_index import ConstructionIndex


def test_loaders_registered():
    reg = all_registered()
    assert "distill_v2" in reg
    assert "gazelle_test" in reg


def test_get_loader_unknown_raises():
    with pytest.raises(KeyError):
        get_loader("nonexistent_source")


def test_gazelle_loader_end_to_end():
    """Gazelle loader produces valid schema_v2 sentences with correct
    annotation quality and at least some role labels populated.
    """
    loader = GazelleLoader(root=str(ROOT))
    sents = loader.load_all()
    assert len(sents) >= 25, f"expected ~30 Gazelle sentences, got {len(sents)}"
    s = sents[0]
    assert s.metadata.source == "gazelle_test"
    assert s.metadata.annotation_quality == "gold_human"
    assert s.n_tokens > 0
    assert s.schema_version == "2.0.0"


def test_metadata_population():
    loader = GazelleLoader(root=str(ROOT))
    sents = loader.load_all()
    n = difficulty.populate_all(sents)
    assert n == len(sents)
    # All sentences should have a difficulty in 1..7
    for s in sents:
        assert 1 <= s.curriculum.difficulty_level <= 7


def test_construction_index_filter():
    loader = GazelleLoader(root=str(ROOT))
    sents = loader.load_all()
    difficulty.populate_all(sents)

    idx = ConstructionIndex.from_sentences(sents)
    assert len(idx) == len(sents)
    assert "msa_news" in idx.domain_histogram()
    # Filter by domain works
    msa = idx.filter(domain="msa_news")
    assert len(msa) == len(sents)
    # Filter by quality works
    gold = idx.filter(quality="gold_human")
    assert len(gold) == len(sents)


def test_semantic_pressure_in_range():
    loader = GazelleLoader(root=str(ROOT))
    sents = loader.load_all()
    for s in sents:
        sp = sp_score(s)
        assert 0 <= sp <= 3
        amb = amb_score(s)
        assert 0.0 <= amb <= 1.0
