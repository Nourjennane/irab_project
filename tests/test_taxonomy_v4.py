"""Phase 4a taxonomy_v4 unit tests — bijective mapping + canonicalisation."""
from __future__ import annotations

import pytest

from irab_tashkeel.structured.schema import ROLE_LABELS as ROLE_LABELS_V3
from irab_tashkeel.structured.taxonomy_v4 import (
    ROLE_LABELS_V4, ROLE_TO_ID_V4, NEW_TO_OLD, OLD_TO_NEW_CHILDREN,
    canonicalize_role_v4, collapse_to_v3, AUTO_FALLBACK_AT_RISK, auto_fallback,
)


def test_v4_has_34_unique_labels():
    assert len(ROLE_LABELS_V4) == 34
    assert len(set(ROLE_LABELS_V4)) == 34


def test_new_to_old_covers_every_v4_label():
    assert set(NEW_TO_OLD.keys()) == set(ROLE_LABELS_V4)


def test_new_to_old_targets_only_v3_labels():
    assert set(NEW_TO_OLD.values()) <= set(ROLE_LABELS_V3)


def test_every_v3_label_is_reachable():
    """Every old (v3) label has at least one v4 child."""
    for v3_label in ROLE_LABELS_V3:
        assert v3_label in OLD_TO_NEW_CHILDREN
        assert len(OLD_TO_NEW_CHILDREN[v3_label]) >= 1


def test_round_trip_identity_on_all_v3_labels():
    """For every v3 label x, there is some v4 child y such that NEW_TO_OLD[y] == x."""
    for v3_label in ROLE_LABELS_V3:
        children = OLD_TO_NEW_CHILDREN[v3_label]
        for v4_child in children:
            assert NEW_TO_OLD[v4_child] == v3_label


def test_split_buckets_have_residuals():
    """Buckets we split (dharf, fil, harf_other, mafoul_other) must keep a
    residual v4 label of the same name."""
    for parent in ("dharf", "fil", "harf_other", "mafoul_other"):
        assert parent in ROLE_LABELS_V4
        assert NEW_TO_OLD[parent] == parent


def test_split_buckets_have_multiple_children():
    """The 4 split buckets must have ≥ 2 v4 children each (residual + ≥1 split)."""
    expected_children = {
        "dharf":         ["dharf", "dharf_zaman", "dharf_makan"],
        "fil":           ["fil", "fil_madi", "fil_mudari", "fil_naqis"],
        "harf_other":    ["harf_other", "harf_nafy", "harf_nasb", "harf_tahqiq"],
        "mafoul_other":  ["mafoul_other", "mafoul_mutlaq"],
    }
    for parent, expected in expected_children.items():
        assert sorted(OLD_TO_NEW_CHILDREN[parent]) == sorted(expected)


def test_collapse_to_v3_is_pure():
    """collapse_to_v3 returns the same v3 label for the same v4 input every time."""
    for v4_label in ROLE_LABELS_V4:
        assert collapse_to_v3(v4_label) == NEW_TO_OLD[v4_label]
    assert collapse_to_v3(None) is None


def test_canonicalize_v4_recognises_split_patterns():
    """The 9 v4 splits must canonicalise from their primary raw forms."""
    cases = [
        # (raw, expected v4 label)
        ("ظرف زمان", "dharf_zaman"),
        ("ظرف مكان", "dharf_makan"),
        ("فعل ماضٍ", "fil_madi"),
        ("فعل ماض",  "fil_madi"),
        ("فعل مضارع", "fil_mudari"),
        ("فعل ناقص",  "fil_naqis"),
        ("حرف نفي",        "harf_nafy"),
        ("حرف نفي وجزم",   "harf_nafy"),
        ("حرف توكيد ونصب", "harf_nasb"),
        ("حرف ناصب",       "harf_nasb"),
        ("حرف نصب",        "harf_nasb"),
        ("حرف تحقيق",      "harf_tahqiq"),
        ("مفعول مطلق",     "mafoul_mutlaq"),
    ]
    for raw, expected in cases:
        got = canonicalize_role_v4(raw)
        assert got == expected, f"canonicalize_role_v4({raw!r}) = {got!r}, expected {expected!r}"


def test_canonicalize_v4_unchanged_v3_labels():
    """Unsplit v3 labels (mubtada, khabar, …) should canonicalise to themselves."""
    cases = [
        ("مبتدأ",        "mubtada"),
        ("خبر",          "khabar"),
        ("فاعل",         "fail"),
        ("نائب فاعل",    "naib_fail"),
        ("مفعول به",     "mafoul_bih"),
        ("نعت",          "naat"),
        ("بدل",          "badal"),
        ("حال",          "hal"),
        ("تمييز",        "tamyeez"),
        ("مضاف إليه",    "mudaaf_ilayh"),
        ("اسم مجرور",    "ism_majrur"),
        ("حرف جر",       "harf_jarr"),
        ("حرف عطف",      "harf_atf"),
        ("اسم إن",       "ism_inna"),
        ("خبر إن",       "khabar_inna"),
        ("اسم كان",      "ism_kana"),
        ("خبر كان",      "khabar_kana"),
        ("منادى",        "munada"),
        ("علامة ترقيم", "punctuation"),
    ]
    for raw, expected in cases:
        got = canonicalize_role_v4(raw)
        assert got == expected, f"canonicalize_role_v4({raw!r}) = {got!r}, expected {expected!r}"


def test_canonicalize_v4_residuals():
    """Generic patterns that don't match a v4 split should fall back to the
    v4 residual of the parent (e.g. 'فعل' alone → 'fil', 'ظرف' alone → 'dharf')."""
    cases = [
        ("فعل",        "fil"),         # generic, not فعل ماض/مضارع/ناقص
        ("ظرف",        "dharf"),       # generic adverbial
        ("حرف",        "harf_other"),  # generic particle
        ("مفعول فيه",  "mafoul_other"),# split candidate Phase 4b, currently residual
        ("مفعول لأجله", "mafoul_other"),
    ]
    for raw, expected in cases:
        got = canonicalize_role_v4(raw)
        assert got == expected, f"canonicalize_role_v4({raw!r}) = {got!r}, expected {expected!r}"


def test_canonicalize_v4_other_for_unknown():
    assert canonicalize_role_v4("اسم موصول") == "other"   # mawsool deferred to 4b
    assert canonicalize_role_v4("لا محل له من الإعراب") == "other"


def test_canonicalize_v4_handles_none_and_empty():
    assert canonicalize_role_v4(None) is None
    assert canonicalize_role_v4("") is None
    assert canonicalize_role_v4("   ") is None


def test_auto_fallback_only_applies_to_at_risk_labels():
    """auto_fallback should only collapse labels listed in AUTO_FALLBACK_AT_RISK."""
    # at-risk label with low support → fallback
    assert auto_fallback("harf_tahqiq", support=10, f1=0.40) == "harf_other"
    # at-risk label with adequate support → keep
    assert auto_fallback("harf_tahqiq", support=200, f1=0.80) == "harf_tahqiq"
    # not-at-risk label → never fallback
    assert auto_fallback("dharf_zaman", support=10, f1=0.40) == "dharf_zaman"
    assert auto_fallback("fil_madi", support=10, f1=0.40) == "fil_madi"


def test_at_risk_set_is_documented_subset():
    """AUTO_FALLBACK_AT_RISK should be a subset of ROLE_LABELS_V4."""
    assert set(AUTO_FALLBACK_AT_RISK) <= set(ROLE_LABELS_V4)
