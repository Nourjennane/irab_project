"""Canonical i'rāb label schema for multi-head structured prediction.

Four classification axes — each closed-vocabulary, ASCII-keyed:

* ``CASE_LABELS``    (5)  raf / nasb / jarr / jazm / mabni
* ``ROLE_LABELS``    (25) syntactic-role taxonomy (mubtada, fail, mafoul_bih, ...)
* ``MARKER_LABELS``  (18) surface case-marker phrases (damma_visible, fatha_visible, ...)
* ``POS_LABELS``     (6)  noun / verb / particle / pronoun / adjective / punctuation

The label sets are derived from the distill_v2/word_level.jsonl distribution
(77,534 rows, 590 unique role strings, 109 unique marker phrases) by collapsing
the long tail into a small canonical set that covers ~95% of mass.

Two directions are exposed:

* **prose -> canonical** via ``canonicalize_*`` functions used at training-data
  prep time and at evaluation time (when scoring against gold prose).
* **canonical -> Arabic surface** via ``ARABIC_*`` lookup tables used by the
  template renderer at inference time.

This module is intentionally string-only, no torch dependency, so it can be
imported by data tooling that runs without a GPU.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Arabic normalization (alif/ya/hamza/tatweel/diacritics)
# ---------------------------------------------------------------------------
_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
# Match alif variants except the bare alif itself; the bare alif is preserved.
_ALIF_VARIANTS = str.maketrans({
    "آ": "ا",  # ALIF WITH MADDA ABOVE
    "أ": "ا",  # ALIF WITH HAMZA ABOVE
    "إ": "ا",  # ALIF WITH HAMZA BELOW
    "ٱ": "ا",  # ALIF WASLA
})
_YA_NORMALIZE = str.maketrans({
    "ى": "ي",  # ALIF MAQSURA -> YA
})
_HAMZA_FORMS = str.maketrans({
    "ؤ": "ء",  # WAW WITH HAMZA -> bare hamza
    "ئ": "ء",  # YEH WITH HAMZA -> bare hamza
})


def arabic_normalize(s: str, *, fold_alif: bool = True, fold_ya: bool = True,
                     fold_hamza: bool = False) -> str:
    """Normalize Arabic text for stable matching.

    Defaults: NFC, strip diacritics, drop tatweel, fold alif variants, fold ya/alif maqsura.
    Hamza-fold is OFF by default because it discards a real distinction;
    callers that match Haiku-style prose can flip it on.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = _DIACRITICS.sub("", s)
    s = s.replace(_TATWEEL, "")
    if fold_alif:
        s = s.translate(_ALIF_VARIANTS)
    if fold_ya:
        s = s.translate(_YA_NORMALIZE)
    if fold_hamza:
        s = s.translate(_HAMZA_FORMS)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Case (5)
# ---------------------------------------------------------------------------
CASE_LABELS = ["raf", "nasb", "jarr", "jazm", "mabni"]
N_CASE = len(CASE_LABELS)
CASE_TO_ID = {c: i for i, c in enumerate(CASE_LABELS)}
ID_TO_CASE = {i: c for c, i in CASE_TO_ID.items()}

# Map from raw distill_v2 case strings (and structural.py's labels) to canonical.
# Both Haiku's romanizations (rafʿ/naṣb) and the extractor's English keys
# (marfu/mansub/majrur/majzum) are accepted.
_CASE_ALIASES = {
    # Haiku/distill_v2 native
    "raf": "raf", "raf'": "raf", "rafʿ": "raf", "rafu": "raf",
    "nasb": "nasb", "naṣb": "nasb", "naSb": "nasb",
    "jarr": "jarr", "jar": "jarr", "jr": "jarr",
    "jazm": "jazm",
    "mabni": "mabni", "mabniyy": "mabni",
    # structural.py keys
    "marfu": "raf", "mansub": "nasb", "majrur": "jarr", "majzum": "jazm",
    # Arabic-script noise
    "مبني": "mabni", "جر": "jarr", "رفع": "raf", "نصب": "nasb", "جزم": "jazm",
}


def canonicalize_case(raw: Optional[str]) -> Optional[str]:
    """Map any seen case string to a canonical label, or None on no match."""
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    if s in _CASE_ALIASES:
        return _CASE_ALIASES[s]
    s_low = s.lower()
    if s_low in _CASE_ALIASES:
        return _CASE_ALIASES[s_low]
    s_norm = arabic_normalize(s, fold_alif=False, fold_ya=False)
    if s_norm in _CASE_ALIASES:
        return _CASE_ALIASES[s_norm]
    # Tolerant containment for compound noise like "naṣb/jarr"
    for k in ("raf", "nasb", "jarr", "jazm", "mabni", "marfu", "mansub", "majrur", "majzum"):
        if k in s_low:
            return _CASE_ALIASES[k]
    return None


# ---------------------------------------------------------------------------
# Role (25)
# ---------------------------------------------------------------------------
ROLE_LABELS = [
    # core nominal roles
    "mubtada", "khabar", "fail", "naib_fail",
    # complements / objects
    "mafoul_bih", "mafoul_other",  # other = lah / motlaq / fih / maah collapsed
    # inna sisters
    "ism_inna", "khabar_inna",
    # kana sisters
    "ism_kana", "khabar_kana",
    # modifiers
    "naat", "badal", "hal", "tamyeez",
    # idafa / prep
    "mudaaf_ilayh", "ism_majrur",
    # coordination & emphatics
    "matuf",
    # adverbials
    "dharf",
    # particles
    "harf_jarr", "harf_atf", "harf_other",
    # verbs
    "fil",
    # vocative
    "munada",
    # corpus housekeeping
    "punctuation",
    # catch-all
    "other",
]
N_ROLE = len(ROLE_LABELS)
ROLE_TO_ID = {r: i for i, r in enumerate(ROLE_LABELS)}
ID_TO_ROLE = {i: r for r, i in ROLE_TO_ID.items()}

# Arabic-prose -> canonical role.  Patterns checked in order, first match wins,
# so put longest/most-specific first.
#
# IMPORTANT: patterns are matched against the output of arabic_normalize() which
# folds alif variants (أإآٱ -> ا) and ya variants (ى -> ي).  The pattern strings
# below MUST therefore use bare ا and bare ي only — using أ/إ/ى in the pattern
# would never match because the input has already been folded.
_ROLE_PATTERNS: list[tuple[str, str]] = [
    # inna / kana sisters (must precede generic ism/khabar)
    (r"اسم\s*ان", "ism_inna"),       # اسم إن / اسم أن both -> اسم ان
    (r"خبر\s*ان", "khabar_inna"),
    (r"اسم\s*كان", "ism_kana"),
    (r"اسم\s*اصبح", "ism_kana"),     # اسم أصبح
    (r"اسم\s*ظل", "ism_kana"),
    (r"اسم\s*ليس", "ism_kana"),
    (r"اسم\s*صار", "ism_kana"),
    (r"اسم\s*بات", "ism_kana"),
    (r"اسم\s*امسى", "ism_kana"),
    (r"اسم\s*ما\s*زال", "ism_kana"),
    (r"خبر\s*كان", "khabar_kana"),
    (r"خبر\s*اصبح", "khabar_kana"),
    (r"خبر\s*ظل", "khabar_kana"),
    (r"خبر\s*ليس", "khabar_kana"),
    (r"خبر\s*صار", "khabar_kana"),
    (r"خبر\s*بات", "khabar_kana"),
    # idafa / prep complements
    (r"مضاف\s*اليه", "mudaaf_ilayh"),  # مضاف إليه -> مضاف اليه
    (r"^مضاف$", "mudaaf_ilayh"),       # bare "مضاف" alone treated as ilayh
    (r"اسم\s*مجرور", "ism_majrur"),
    (r"مجرور\s*بحرف\s*الجر", "ism_majrur"),
    (r"جار\s*ومجرور", "ism_majrur"),
    (r"^جار$", "ism_majrur"),
    (r"^مجرور$", "ism_majrur"),
    # objects
    (r"نائب\s*فاعل", "naib_fail"),
    (r"مفعول\s*به", "mafoul_bih"),
    (r"مفعول\s*مطلق", "mafoul_other"),
    (r"مفعول\s*لاجله", "mafoul_other"),  # مفعول لأجله -> مفعول لاجله
    (r"مفعول\s*فيه", "mafoul_other"),
    (r"مفعول\s*معه", "mafoul_other"),
    # core nominal
    (r"فاعل", "fail"),
    (r"مبتدا", "mubtada"),  # مبتدأ -> مبتدا (covers مبتدأ ثان too)
    (r"خبر", "khabar"),
    # modifiers / appositives
    (r"نعت", "naat"),
    (r"صفة", "naat"),
    (r"بدل", "badal"),
    (r"عطف\s*بيان", "matuf"),
    (r"معطوف", "matuf"),
    (r"^عطف$", "matuf"),
    (r"حال", "hal"),
    (r"تمييز", "tamyeez"),
    # vocative
    (r"منادى", "munada"),  # already uses ى -> ي, but بناء: منادى -> منادي after fold
    (r"منادي", "munada"),
    # adverbials
    (r"ظرف", "dharf"),
    # particles (must come after compound roles like "حرف جر" used as POS-like role)
    (r"حرف\s*جر", "harf_jarr"),
    (r"حرف\s*عطف", "harf_atf"),
    (r"عاطف", "harf_atf"),
    (r"حرف", "harf_other"),
    (r"اداة\s*تعريف", "harf_other"),  # أداة تعريف -> اداة تعريف
    # verbs
    (r"فعل\s*ماض", "fil"),
    (r"فعل\s*مضارع", "fil"),
    (r"فعل\s*امر", "fil"),  # فعل أمر -> فعل امر
    (r"فعل\s*ناقص", "fil"),
    (r"فعل", "fil"),
    # pronouns / relatives — let derive_pos handle these; map to closest role
    (r"اسم\s*موصول", "other"),
    # corpus housekeeping
    (r"علامة\s*ترقيم", "punctuation"),
]
_ROLE_RE = [(re.compile(p), c) for p, c in _ROLE_PATTERNS]


def canonicalize_role(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = arabic_normalize(raw)
    if not s:
        return None
    for rx, label in _ROLE_RE:
        if rx.search(s):
            return label
    return "other"


# ---------------------------------------------------------------------------
# Marker (18)
# ---------------------------------------------------------------------------
MARKER_LABELS = [
    "damma_visible", "fatha_visible", "kasra_visible",
    "damma_hidden",  "fatha_hidden",  "kasra_hidden",
    "tanween_damm", "tanween_fath", "tanween_kasr",
    "sukun", "sukun_hidden",
    "ya", "waw", "alif", "nun",
    "fath_short",   # bare الفتح / الفتح المقدر, used for verbs
    "none",         # the literal "لا يوجد" rows
    "other",
]
N_MARKER = len(MARKER_LABELS)
MARKER_TO_ID = {m: i for i, m in enumerate(MARKER_LABELS)}
ID_TO_MARKER = {i: m for m, i in MARKER_TO_ID.items()}

# Patterns are matched after arabic_normalize() — use bare ا and bare ي only.
_MARKER_PATTERNS: list[tuple[str, str]] = [
    # tanween (must precede plain damma/fatha/kasra)
    (r"تنوين\s*الضم", "tanween_damm"),
    (r"تنوين\s*الفتح", "tanween_fath"),
    (r"تنوين\s*الكسر", "tanween_kasr"),
    # hidden / muqaddara variants (must precede visible)
    (r"الضمة\s*المقدرة", "damma_hidden"),
    (r"الفتحة\s*المقدرة", "fatha_hidden"),
    (r"الكسرة\s*المقدرة", "kasra_hidden"),
    (r"السكون\s*المقدر", "sukun_hidden"),
    (r"الفتح\s*المقدر", "fath_short"),
    (r"الضم\s*المقدر", "damma_hidden"),
    # visible
    (r"الضمة\s*الظاهرة", "damma_visible"),
    (r"الفتحة\s*الظاهرة", "fatha_visible"),
    (r"الكسرة\s*الظاهرة", "kasra_visible"),
    # bare forms (assume visible — the most common short-form in distill_v2)
    (r"الضمة", "damma_visible"),
    (r"الفتحة", "fatha_visible"),
    (r"الكسرة", "kasra_visible"),
    (r"الضم\b", "damma_visible"),     # bare "الضم" -> visible damma
    (r"الفتح\b", "fath_short"),
    (r"الكسر\b", "kasra_visible"),
    # sukun
    (r"السكون", "sukun"),
    # special long-vowel and pluralization markers — ا and ي are bare here
    (r"الياء", "ya"),
    (r"الواو", "waw"),
    (r"الالف", "alif"),     # الألف -> الالف after fold
    (r"النون", "nun"),
    # explicit none / no-marker variants
    (r"لا\s*يوجد", "none"),
    (r"لا\s*علامة", "none"),
    (r"لا\s*محل", "none"),
    (r"^مبني", "none"),  # bare "mabni" as a marker is a corpus quirk
    (r"البناء", "none"),
]
_MARKER_RE = [(re.compile(p), c) for p, c in _MARKER_PATTERNS]


def canonicalize_marker(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = arabic_normalize(raw)
    if not s:
        return None
    for rx, label in _MARKER_RE:
        if rx.search(s):
            return label
    return "other"


# ---------------------------------------------------------------------------
# POS (6) — derived heuristically because distill_v2/word_level.jsonl has
# an empty pos field. We map from canonical role + raw role string.
# ---------------------------------------------------------------------------
POS_LABELS = ["noun", "verb", "particle", "pronoun", "adjective", "punctuation"]
N_POS = len(POS_LABELS)
POS_TO_ID = {p: i for i, p in enumerate(POS_LABELS)}
ID_TO_POS = {i: p for p, i in POS_TO_ID.items()}

_PRONOUN_RE = re.compile(r"ضمير")
_VERB_RE = re.compile(r"فعل")
_PARTICLE_RE = re.compile(r"حرف|أداة|عاطف")
_ADJECTIVE_RE = re.compile(r"نعت|صفة")
_PUNCT_RE = re.compile(r"علامة\s*ترقيم|ترقيم")


# Direct mapping from explicit Arabic POS tag (sentence-level distilled.jsonl).
_POS_DIRECT = {
    "اسم": "noun", "فعل": "verb", "حرف": "particle", "ضمير": "pronoun",
    "صفة": "adjective", "نعت": "adjective", "علامة ترقيم": "punctuation",
    "اداة": "particle", "أداة": "particle",
}


def canonicalize_pos(raw_pos: Optional[str]) -> Optional[str]:
    """Map an explicit Arabic POS tag (e.g. 'اسم') to a canonical POS label."""
    if raw_pos is None:
        return None
    s = raw_pos.strip()
    if not s:
        return None
    if s in _POS_DIRECT:
        return _POS_DIRECT[s]
    s_norm = arabic_normalize(s)
    return _POS_DIRECT.get(s_norm)


def derive_pos(raw_role: Optional[str], raw_irab: Optional[str] = None,
               raw_pos: Optional[str] = None) -> str:
    """POS assignment.  Prefer explicit raw_pos; fall back to heuristic.

    distill_v2/distilled.jsonl populates pos with Arabic tags (اسم/فعل/حرف/ضمير).
    distill_v2/word_level.jsonl leaves pos empty, so the heuristic kicks in.
    """
    if raw_pos:
        cp = canonicalize_pos(raw_pos)
        if cp:
            return cp
    text = " ".join(filter(None, [raw_role, raw_irab]))
    text_norm = arabic_normalize(text)
    if not text_norm:
        return "noun"
    if _PUNCT_RE.search(text_norm):
        return "punctuation"
    if _PRONOUN_RE.search(text_norm):
        return "pronoun"
    if _VERB_RE.search(text_norm):
        return "verb"
    if _PARTICLE_RE.search(text_norm):
        return "particle"
    if _ADJECTIVE_RE.search(text_norm):
        return "adjective"
    return "noun"


# ---------------------------------------------------------------------------
# Inverse: canonical -> Arabic surface (for the template renderer)
# ---------------------------------------------------------------------------
ARABIC_CASE_FORMS = {
    # (canonical_case): {"adj": agreeing adjective, "verb": verb form for prose}
    "raf":  {"adj": "مرفوع",  "verb": "رفعه"},
    "nasb": {"adj": "منصوب",  "verb": "نصبه"},
    "jarr": {"adj": "مجرور",  "verb": "جره"},
    "jazm": {"adj": "مجزوم",  "verb": "جزمه"},
    "mabni": {"adj": "مبني",  "verb": "بنائه"},
}

ARABIC_ROLE_FORMS = {
    "mubtada":     "مبتدأ",
    "khabar":      "خبر",
    "fail":        "فاعل",
    "naib_fail":   "نائب فاعل",
    "mafoul_bih":  "مفعول به",
    "mafoul_other": "مفعول",
    "ism_inna":    "اسم إن",
    "khabar_inna": "خبر إن",
    "ism_kana":    "اسم كان",
    "khabar_kana": "خبر كان",
    "naat":        "نعت",
    "badal":       "بدل",
    "hal":         "حال",
    "tamyeez":     "تمييز",
    "mudaaf_ilayh": "مضاف إليه",
    "ism_majrur":  "اسم مجرور",
    "matuf":       "معطوف",
    "dharf":       "ظرف",
    "harf_jarr":   "حرف جر",
    "harf_atf":    "حرف عطف",
    "harf_other":  "حرف",
    "fil":         "فعل",
    "munada":      "منادى",
    "punctuation": "علامة ترقيم",
    "other":       "",
    # ── Phase 4a (taxonomy v4) additions — additive; v3 entries unchanged ──
    "dharf_zaman":   "ظرف زمان",
    "dharf_makan":   "ظرف مكان",
    "fil_madi":      "فعل ماضٍ",
    "fil_mudari":    "فعل مضارع",
    "fil_naqis":     "فعل ناقص",
    "harf_nafy":     "حرف نفي",
    "harf_nasb":     "حرف ناصب",
    "harf_tahqiq":   "حرف تحقيق",
    "mafoul_mutlaq": "مفعول مطلق",
}

ARABIC_MARKER_FORMS = {
    "damma_visible":  "الضمة الظاهرة على آخره",
    "fatha_visible":  "الفتحة الظاهرة على آخره",
    "kasra_visible":  "الكسرة الظاهرة على آخره",
    "damma_hidden":   "الضمة المقدرة على آخره",
    "fatha_hidden":   "الفتحة المقدرة على آخره",
    "kasra_hidden":   "الكسرة المقدرة على آخره",
    "tanween_damm":   "تنوين الضم",
    "tanween_fath":   "تنوين الفتح",
    "tanween_kasr":   "تنوين الكسر",
    "sukun":          "السكون",
    "sukun_hidden":   "السكون المقدر",
    "ya":             "الياء",
    "waw":            "الواو",
    "alif":           "الألف",
    "nun":            "النون",
    "fath_short":     "الفتح",
    "none":           "",
    "other":          "",
}
