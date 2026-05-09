"""Phase R2 — unit tests for structural reasoners.

Tests the consensus-voting + confidence machinery against synthetic
:class:`RetrievalHit` payloads (no model load, no FAISS index).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from irab_tashkeel.grammar_memory.memory import RetrievalHit
from irab_tashkeel.grammar_memory.signature import ConstructionInstance
from irab_tashkeel.grammar_memory.structural_reasoner import (
    KanaReasoner, IstithnaReasoner, MawsoolReasoner,
    InnaReasoner, QuranicProxyReasoner,
    REASONER_REGISTRY, get_reasoner, supported_families,
)


def _mk_instance(items, particle="كان", construction="kana_sisters",
                  particle_group="kana_completion"):
    sentence = " ".join(it.get("word", "?") for it in items)
    return ConstructionInstance(
        instance_id=f"_test_{hash(sentence)}",
        sentence=sentence,
        sentence_idx=0,
        construction=construction,
        particle_group=particle_group,
        span=(0, len(items)),
        particle_surface=particle,
        head_morph={"gender": "masc", "number": "sg", "definite": "definite"},
        head_deprel="root",
        head_governor_upos="ROOT",
        sentence_length=len(items),
        items=items,
    )


def _mk_hit(items, *, particle="كان", construction="kana_sisters",
            particle_group="kana_completion", cosine=0.85, sym=0.7):
    inst = _mk_instance(items, particle=particle, construction=construction,
                        particle_group=particle_group)
    return RetrievalHit(
        instance=inst, cosine=cosine, sym_overlap=sym,
        score=0.7 * cosine + 0.3 * sym, rank=0,
    )


# ---------------------------------------------------------------------------
# 1. Registry sanity
# ---------------------------------------------------------------------------

def test_registry_has_all_target_families():
    families = set(supported_families())
    assert {"kana_sisters", "istithna", "mawsool",
            "inna_sisters", "quranic_proxy"}.issubset(families)


def test_get_reasoner_returns_correct_class():
    assert isinstance(get_reasoner("kana_sisters"), KanaReasoner)
    assert isinstance(get_reasoner("istithna"), IstithnaReasoner)
    assert get_reasoner("idafa") is None   # no reasoner for iḍāfa


# ---------------------------------------------------------------------------
# 2. Insufficient retrievals → invalid output
# ---------------------------------------------------------------------------

def test_insufficient_retrievals_invalid():
    reasoner = KanaReasoner()
    hits = [_mk_hit([
        {"word": "كان", "case": "mabni", "role": "fil", "marker": "fath_short"},
        {"word": "الطالبُ", "case": "raf", "role": "ism_kana", "marker": "damma_visible"},
        {"word": "مجتهداً", "case": "nasb", "role": "khabar_kana", "marker": "tanween_fath"},
    ])]   # only 1 hit
    out = reasoner.reason(
        query_span=[], retrieved=hits, query_words=["كان", "الطالبُ", "مجتهداً"],
        span=(0, 3), particle_group="kana_completion", particle_surface="كان",
    )
    assert not out.valid
    assert out.confidence == 0.0


# ---------------------------------------------------------------------------
# 3. KanaReasoner full consensus
# ---------------------------------------------------------------------------

def test_kana_reasoner_full_consensus():
    """5 retrievals with identical labels → consensus 1.0 → high confidence."""
    items = [
        {"word": "كان", "case": "mabni", "role": "fil", "marker": "fath_short"},
        {"word": "الطالبُ", "case": "raf", "role": "ism_kana", "marker": "damma_visible"},
        {"word": "مجتهداً", "case": "nasb", "role": "khabar_kana", "marker": "tanween_fath"},
    ]
    hits = [_mk_hit(items) for _ in range(5)]
    out = KanaReasoner().reason(
        query_span=[], retrieved=hits, query_words=["كان", "أحمد", "مجدّ"],
        span=(0, 3), particle_group="kana_completion", particle_surface="كان",
    )
    assert out.valid
    assert out.span_len == 3
    assert out.consensus_rate == pytest.approx(1.0)
    # Confidence = 0.8 * 1.0 + 0.2 * 0.85 = 0.97
    assert out.confidence == pytest.approx(0.8 + 0.2 * 0.85, abs=1e-6)
    # Position 0 (particle): mabni / fil
    assert out.predicted[0]["case"] == "mabni"
    assert out.predicted[0]["role"] == "fil"
    # Position 1 (ism): raf / ism_kana
    assert out.predicted[1]["case"] == "raf"
    assert out.predicted[1]["role"] == "ism_kana"
    # Position 2 (khabar): nasb / khabar_kana
    assert out.predicted[2]["case"] == "nasb"
    assert out.predicted[2]["role"] == "khabar_kana"
    # Rule string mentions kana_completion
    assert "kana_completion" in out.rule


def test_kana_reasoner_partial_consensus():
    """3/5 agree, 2/5 disagree → consensus 0.6."""
    base = [
        {"word": "كان", "case": "mabni", "role": "fil", "marker": "fath_short"},
        {"word": "الطالبُ", "case": "raf", "role": "ism_kana", "marker": "damma_visible"},
        {"word": "مجتهداً", "case": "nasb", "role": "khabar_kana", "marker": "tanween_fath"},
    ]
    alt = [
        {"word": "كان", "case": "mabni", "role": "fil", "marker": "fath_short"},
        {"word": "الكتابُ", "case": "raf", "role": "ism_kana", "marker": "damma_visible"},
        # noisy retrieval: wrong khabar role
        {"word": "ثمين", "case": "nasb", "role": "naat", "marker": "tanween_fath"},
    ]
    hits = [_mk_hit(base) for _ in range(3)] + [_mk_hit(alt) for _ in range(2)]
    out = KanaReasoner().reason(
        query_span=[], retrieved=hits, query_words=["كان", "الكتابُ", "مفيداً"],
        span=(0, 3), particle_group="kana_completion", particle_surface="كان",
    )
    assert out.valid
    # Position 2 role: 3 of 5 vote khabar_kana → consensus_rate at pos 2 role = 0.6
    assert out.consensus_per_pos[2]["role_rate"] == pytest.approx(0.6, abs=1e-6)
    # Top vote is still khabar_kana (3 > 2)
    assert out.predicted[2]["role"] == "khabar_kana"
    # Overall consensus rate: only pos 2's role is split (3 vs 2).
    # pos 0: (1 + 1 + 1) / 3 = 1.0
    # pos 1: (1 + 1 + 1) / 3 = 1.0
    # pos 2: (1 + 0.6 + 1) / 3 ≈ 0.867
    # mean = (1 + 1 + 0.867) / 3 ≈ 0.956
    assert out.consensus_rate == pytest.approx((1 + 1 + (1 + 0.6 + 1) / 3) / 3, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. IstithnaReasoner sub-pattern rules
# ---------------------------------------------------------------------------

def test_istithna_illa_rule():
    items = [
        {"word": "إلا", "case": "mabni", "role": "harf_other", "marker": "sukun"},
        {"word": "خالداً", "case": "nasb", "role": "mafoul_other", "marker": "tanween_fath"},
        {"word": "كذا", "case": "nasb", "role": "other", "marker": "tanween_fath"},
    ]
    hits = [_mk_hit(items, particle="إلا", construction="istithna", particle_group="illa") for _ in range(4)]
    out = IstithnaReasoner().reason(
        query_span=[], retrieved=hits, query_words=["إلا", "زيداً", "."],
        span=(0, 3), particle_group="illa", particle_surface="إلا",
    )
    assert out.valid
    assert "illa" in out.rule
    assert out.predicted[0]["case"] == "mabni"
    assert out.predicted[1]["case"] == "nasb"
    assert out.predicted[1]["role"] == "mafoul_other"


def test_istithna_noun_rule():
    items = [
        {"word": "غير", "case": "nasb", "role": "mafoul_other", "marker": "fatha_visible"},
        {"word": "زيد", "case": "jarr", "role": "mudaaf_ilayh", "marker": "kasra_visible"},
        {"word": "كذا", "case": "raf", "role": "other", "marker": "damma_visible"},
    ]
    hits = [_mk_hit(items, particle="غير", construction="istithna", particle_group="istithna_noun") for _ in range(4)]
    out = IstithnaReasoner().reason(
        query_span=[], retrieved=hits, query_words=["غير", "أحمد", "."],
        span=(0, 3), particle_group="istithna_noun", particle_surface="غير",
    )
    assert out.valid
    assert "noun" in out.rule
    assert out.predicted[0]["case"] == "nasb"
    assert out.predicted[1]["case"] == "jarr"
    assert out.predicted[1]["role"] == "mudaaf_ilayh"


# ---------------------------------------------------------------------------
# 5. InnaReasoner — case mirror of Kana
# ---------------------------------------------------------------------------

def test_inna_reasoner_assertion():
    items = [
        {"word": "إن", "case": "mabni", "role": "harf_other", "marker": "fath_short"},
        {"word": "الطالبَ", "case": "nasb", "role": "ism_inna", "marker": "fatha_visible"},
        {"word": "مجتهدٌ", "case": "raf", "role": "khabar_inna", "marker": "tanween_damm"},
    ]
    hits = [_mk_hit(items, particle="إن", construction="inna_sisters",
                     particle_group="inna_assertion") for _ in range(5)]
    out = InnaReasoner().reason(
        query_span=[], retrieved=hits, query_words=["إن", "أحمد", "مجدّ"],
        span=(0, 3), particle_group="inna_assertion", particle_surface="إن",
    )
    assert out.valid
    # Inna reverses kana: ism is nasb, khabar is raf
    assert out.predicted[1]["case"] == "nasb"
    assert out.predicted[1]["role"] == "ism_inna"
    assert out.predicted[2]["case"] == "raf"
    assert out.predicted[2]["role"] == "khabar_inna"
    assert "inna_assertion" in out.rule


# ---------------------------------------------------------------------------
# 6. MawsoolReasoner + QuranicProxyReasoner — smoke tests
# ---------------------------------------------------------------------------

def test_mawsool_reasoner_definite():
    items = [
        {"word": "الذي", "case": "mabni", "role": "other", "marker": "sukun"},
        {"word": "يعمل", "case": "raf", "role": "fil", "marker": "damma_visible"},
        {"word": "بصدق", "case": "jarr", "role": "ism_majrur", "marker": "kasra_visible"},
    ]
    hits = [_mk_hit(items, particle="الذي", construction="mawsool",
                     particle_group="definite_relative") for _ in range(4)]
    out = MawsoolReasoner().reason(
        query_span=[], retrieved=hits, query_words=["الذي", "يأتي", "."],
        span=(0, 3), particle_group="definite_relative", particle_surface="الذي",
    )
    assert out.valid
    assert "definite" in out.rule
    assert out.predicted[0]["case"] == "mabni"


def test_quranic_proxy_reasoner_smoke():
    items = [
        {"word": "قد", "case": "mabni", "role": "harf_other", "marker": "sukun"},
        {"word": "نزل", "case": "mabni", "role": "fil", "marker": "fath_short"},
        {"word": "الكتابُ", "case": "raf", "role": "fail", "marker": "damma_visible"},
    ]
    hits = [_mk_hit(items, particle="قد", construction="quranic_proxy",
                     particle_group="qad_idh") for _ in range(4)]
    out = QuranicProxyReasoner().reason(
        query_span=[], retrieved=hits, query_words=["قد", "كتب", "ذلك"],
        span=(0, 3), particle_group="qad_idh", particle_surface="قد",
    )
    assert out.valid
    assert out.predicted[0]["case"] == "mabni"
    assert "qad_idh" in out.rule


# ---------------------------------------------------------------------------
# 7. Confidence sensitivity
# ---------------------------------------------------------------------------

def test_low_cosine_lowers_confidence():
    items = [
        {"word": "كان", "case": "mabni", "role": "fil", "marker": "fath_short"},
        {"word": "الطالبُ", "case": "raf", "role": "ism_kana", "marker": "damma_visible"},
        {"word": "مجتهداً", "case": "nasb", "role": "khabar_kana", "marker": "tanween_fath"},
    ]
    high_cos = [_mk_hit(items, cosine=0.95) for _ in range(5)]
    low_cos = [_mk_hit(items, cosine=0.10) for _ in range(5)]
    out_hi = KanaReasoner().reason(
        query_span=[], retrieved=high_cos, query_words=["كان", "أ", "ب"],
        span=(0, 3), particle_group="kana_completion", particle_surface="كان",
    )
    out_lo = KanaReasoner().reason(
        query_span=[], retrieved=low_cos, query_words=["كان", "أ", "ب"],
        span=(0, 3), particle_group="kana_completion", particle_surface="كان",
    )
    assert out_hi.confidence > out_lo.confidence
