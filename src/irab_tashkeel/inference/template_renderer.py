"""Deterministic template-based Arabic-prose renderer for structured i'rāb.

Given a per-word structured prediction (case / role / marker / POS) the
renderer composes a canonical i'rāb prose string. This is the inverse of the
extractor in :mod:`irab_tashkeel.evaluation.structural` — both can be applied
back-to-back with stable round-trip behaviour for the canonical schema.

Renderer pattern (formulaic Arabic grammar prose):

    {role_form} {case_form_adj} وعلامة {case_form_verb} {marker_form}

Examples:
    case=raf,  role=fail,        marker=damma_visible
        -> "فاعل مرفوع وعلامة رفعه الضمة الظاهرة على آخره"
    case=jarr, role=ism_majrur,  marker=kasra_visible
        -> "اسم مجرور وعلامة جره الكسرة الظاهرة على آخره"
    case=mabni, role=harf_jarr,  marker=sukun
        -> "حرف جر مبني على السكون لا محل له من الإعراب"

Special cases:

* When ``role == "punctuation"`` we emit the bare role form (no case morphology).
* When ``case == "mabni"`` we use the "مبني على X لا محل له" template instead
  of the رفعه/نصبه form.
* When ``marker == "none"`` (literal "لا يوجد") we drop the وعلامة … clause.
* When ``role == "harf_jarr" / "harf_atf" / "harf_other"`` we use the
  particle template "{role_form} مبني على {marker_form} لا محل له من الإعراب".
"""

from __future__ import annotations

from typing import Optional

from ..structured.schema import (
    ARABIC_CASE_FORMS, ARABIC_ROLE_FORMS, ARABIC_MARKER_FORMS,
)
from ..structured.word_irab import WordIrab


_NO_LOC = "لا محل له من الإعراب"
_VERB_BASE_TEMPLATE = "فعل مبني على {marker} {loc}"
_PARTICLE_TEMPLATE = "{role} مبني على {marker_short} {no_loc}"
_MABNI_NOUN_TEMPLATE = "{role} مبني على {marker_short} في محل {case_label}"


# Marker forms split into short ("السكون") vs long ("السكون المقدر") for
# templates where a short form is more idiomatic.
_MARKER_SHORT = {
    "damma_visible": "الضمة",
    "fatha_visible": "الفتحة",
    "kasra_visible": "الكسرة",
    "damma_hidden":  "الضمة المقدرة",
    "fatha_hidden":  "الفتحة المقدرة",
    "kasra_hidden":  "الكسرة المقدرة",
    "tanween_damm":  "تنوين الضم",
    "tanween_fath":  "تنوين الفتح",
    "tanween_kasr":  "تنوين الكسر",
    "sukun":         "السكون",
    "sukun_hidden":  "السكون المقدر",
    "ya":            "الياء",
    "waw":           "الواو",
    "alif":          "الألف",
    "nun":           "النون",
    "fath_short":    "الفتح",
    "none":          "",
    "other":         "",
}


def render_word(item: WordIrab) -> str:
    """Render a single :class:`WordIrab` into Arabic i'rāb prose.

    Defensive: returns an empty string for items that lack any of the four
    required fields. The qualitative renderer can flag these explicitly.
    """
    case = item.case
    role = item.role
    marker = item.marker
    if not (case and role):
        return ""

    role_form = ARABIC_ROLE_FORMS.get(role, "")
    marker_long = ARABIC_MARKER_FORMS.get(marker or "none", "")
    marker_short = _MARKER_SHORT.get(marker or "none", "")

    # ---- particles (harf jarr / atf / other + Phase 4a sub-types) ----
    if role in ("harf_jarr", "harf_atf", "harf_other",
                "harf_nafy", "harf_nasb", "harf_tahqiq"):
        sk = marker_short or "السكون"
        return f"{role_form} مبني على {sk} {_NO_LOC}".strip()

    # ---- punctuation ----
    if role == "punctuation":
        return "علامة ترقيم"

    # ---- mabni nouns / pronouns ----
    if case == "mabni":
        sk = marker_short or "السكون"
        # use a sensible في محل label
        loc_case = {"raf": "رفع", "nasb": "نصب", "jarr": "جر", "jazm": "جزم"}.get(role, "")
        if role in ("fil", "fil_madi", "fil_mudari", "fil_naqis"):
            # v4 sub-types render with their canonical Arabic surface form
            base = ARABIC_ROLE_FORMS.get(role, "فعل")
            return f"{base} مبني على {sk} {_NO_LOC}".strip() if marker_short == "الفتح" or role == "fil_madi" else f"{base} مبني على {sk} {_NO_LOC}".strip()
        if role_form:
            return f"{role_form} مبني على {sk} {_NO_LOC}".strip()
        return f"مبني على {sk} {_NO_LOC}".strip()

    # ---- verbs (non-mabni — rare in MSA news, but exists) ----
    if role in ("fil", "fil_madi", "fil_mudari", "fil_naqis"):
        case_data = ARABIC_CASE_FORMS.get(case, {})
        adj = case_data.get("adj", "")
        verb = case_data.get("verb", "")
        suffix = ""
        if marker_long:
            suffix = f" وعلامة {verb} {marker_long}"
        # Use the v4 surface form when one of the verb sub-types
        verb_base = ARABIC_ROLE_FORMS.get(role, "فعل مضارع")
        return f"{verb_base} {adj}{suffix}".strip()

    # ---- standard nominal frame ----
    case_data = ARABIC_CASE_FORMS.get(case, {})
    adj = case_data.get("adj", "")
    verb = case_data.get("verb", "")
    # Suppress redundant case adjective if the role form already contains it
    # (e.g. "اسم مجرور" already implies "مجرور"; emitting "اسم مجرور مجرور" is wrong).
    if adj and adj in role_form:
        adj = ""
    parts = [role_form, adj]
    if marker_long:
        parts.append(f"وعلامة {verb} {marker_long}")
    return " ".join(p for p in parts if p).strip()


def render_sentence(items, joiner: str = "\n") -> str:
    """Render a list of :class:`WordIrab` to per-line Arabic prose."""
    out = []
    for it in items:
        prose = render_word(it)
        out.append(f"{it.word}: {prose}" if prose else f"{it.word}:")
    return joiner.join(out)
