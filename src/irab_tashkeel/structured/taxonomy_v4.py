"""Phase 4a expanded role taxonomy (v4: 34 labels).

Layered on top of the v3 25-label canonical schema in :mod:`schema`. Phase 4a
splits 4 heterogeneous v3 buckets into linguistically-meaningful sub-types,
producing 9 new labels:

    v3 dharf       → dharf_zaman / dharf_makan / dharf
    v3 fil         → fil_madi / fil_mudari / fil_naqis / fil
    v3 harf_other  → harf_nafy / harf_nasb / harf_tahqiq / harf_other
    v3 mafoul_other → mafoul_mutlaq / mafoul_other

Net: 25 v3 labels - 4 (parents that get split, kept as residual) - 0
(parents fully consumed) + 4 (parent residuals) + 9 (new splits)
   = 25 + 9 new labels = **34 labels total**

`mawsool` (split candidate from `other`) is deferred to Phase 4b.

Frozen design rules:
* **Bijective NEW_TO_OLD on the 25 v3 labels.** Every v4 label maps to
  exactly one v3 label; the round-trip ``OLD_TO_NEW(NEW_TO_OLD(x)) == x``
  holds for every old label (verified by ``tests/test_taxonomy_v4.py``).
* **Canonicalisation deterministic.** ``canonicalize_role_v4(raw)`` returns
  one of the 34 v4 labels OR ``"other"``. Same prose-normalisation pipeline
  as v3.
* **Grouped evaluation pure.** ``collapse_to_v3(v4_label)`` is a 1-to-1
  function that's bit-identical to the rev 2 + Phase 1 evaluation surface.

Used by:
* the Phase 4a corpus builder (``scripts/structured/build_structured_corpus_v4.py``)
* the Phase 4a model training (via ``configs/phase4a_*.yaml``)
* the Phase 4a evaluation pipeline (4 metric streams + stress table)
* the deterministic template renderer (mapping each v4 label to canonical
  Arabic prose; see ``inference/template_renderer.py``)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .schema import (
    ROLE_LABELS as ROLE_LABELS_V3,
    arabic_normalize,
)


# ---------------------------------------------------------------------------
# v4 label set (34 labels). Order matters: matches __init__'s ID-mapping
# discipline (first-seen → ID 0).
# ---------------------------------------------------------------------------
ROLE_LABELS_V4: List[str] = [
    # core nominal roles (unchanged from v3)
    "mubtada", "khabar", "fail", "naib_fail",
    # complements / objects
    "mafoul_bih",
    # NEW: split out مفعول مطلق explicitly; mafoul_other becomes the residual
    "mafoul_mutlaq",
    "mafoul_other",
    # inna sisters (unchanged)
    "ism_inna", "khabar_inna",
    # kana sisters (unchanged)
    "ism_kana", "khabar_kana",
    # modifiers / appositives (unchanged)
    "naat", "badal", "hal", "tamyeez",
    # idafa / prep (unchanged)
    "mudaaf_ilayh", "ism_majrur",
    # coordination / emphatics (unchanged)
    "matuf",
    # adverbials — NEW: split out time + place; dharf becomes the residual
    "dharf_zaman", "dharf_makan",
    "dharf",
    # particles
    "harf_jarr", "harf_atf",
    # NEW: split out 3 high-frequency harf_other sub-types; harf_other = residual
    "harf_nafy", "harf_nasb", "harf_tahqiq",
    "harf_other",
    # verbs — NEW: split out past + present + defective; fil becomes the residual
    "fil_madi", "fil_mudari", "fil_naqis",
    "fil",
    # vocative / housekeeping (unchanged)
    "munada",
    "punctuation",
    # catch-all (unchanged; mawsool will carve out of this in Phase 4b)
    "other",
]

assert len(ROLE_LABELS_V4) == 34, f"Phase 4a must define exactly 34 labels, got {len(ROLE_LABELS_V4)}"
assert len(set(ROLE_LABELS_V4)) == 34, "ROLE_LABELS_V4 has duplicates"

N_ROLE_V4 = len(ROLE_LABELS_V4)
ROLE_TO_ID_V4: Dict[str, int] = {l: i for i, l in enumerate(ROLE_LABELS_V4)}
ID_TO_ROLE_V4: Dict[int, str] = {i: l for l, i in ROLE_TO_ID_V4.items()}


# ---------------------------------------------------------------------------
# v4 → v3 collapse table (frozen, bijective on v3 labels)
# ---------------------------------------------------------------------------
NEW_TO_OLD: Dict[str, str] = {
    # splits → parent
    "dharf_zaman":     "dharf",
    "dharf_makan":     "dharf",
    "dharf":           "dharf",            # residual stays itself
    "fil_madi":        "fil",
    "fil_mudari":      "fil",
    "fil_naqis":       "fil",
    "fil":             "fil",
    "harf_nafy":       "harf_other",
    "harf_nasb":       "harf_other",
    "harf_tahqiq":     "harf_other",
    "harf_other":      "harf_other",
    "mafoul_mutlaq":   "mafoul_other",
    "mafoul_other":    "mafoul_other",
    # all unchanged labels: identity
    "mubtada":         "mubtada",
    "khabar":          "khabar",
    "fail":            "fail",
    "naib_fail":       "naib_fail",
    "mafoul_bih":      "mafoul_bih",
    "ism_inna":        "ism_inna",
    "khabar_inna":     "khabar_inna",
    "ism_kana":        "ism_kana",
    "khabar_kana":     "khabar_kana",
    "naat":            "naat",
    "badal":           "badal",
    "hal":             "hal",
    "tamyeez":         "tamyeez",
    "mudaaf_ilayh":    "mudaaf_ilayh",
    "ism_majrur":      "ism_majrur",
    "matuf":           "matuf",
    "harf_jarr":       "harf_jarr",
    "harf_atf":        "harf_atf",
    "munada":          "munada",
    "punctuation":     "punctuation",
    "other":           "other",
}

# Sanity: every v4 label has a v3 mapping; every v3 label is reachable.
assert set(NEW_TO_OLD.keys()) == set(ROLE_LABELS_V4), \
    "NEW_TO_OLD missing or extra v4 keys"
_v3_targets = set(NEW_TO_OLD.values())
assert _v3_targets <= set(ROLE_LABELS_V3), \
    f"NEW_TO_OLD targets a label outside v3: {_v3_targets - set(ROLE_LABELS_V3)}"
assert set(ROLE_LABELS_V3) <= _v3_targets, \
    f"NEW_TO_OLD doesn't cover all v3 labels: {set(ROLE_LABELS_V3) - _v3_targets}"


def collapse_to_v3(v4_label: Optional[str]) -> Optional[str]:
    """Map a v4 label down to its v3 canonical (used for grouped evaluation)."""
    if v4_label is None:
        return None
    return NEW_TO_OLD.get(v4_label, "other")


# ---------------------------------------------------------------------------
# v3 → list-of-v4 (informational; one v3 label has multiple v4 children)
# ---------------------------------------------------------------------------
OLD_TO_NEW_CHILDREN: Dict[str, List[str]] = {}
for v4, v3 in NEW_TO_OLD.items():
    OLD_TO_NEW_CHILDREN.setdefault(v3, []).append(v4)


# ---------------------------------------------------------------------------
# Canonicalisation: raw Arabic role prose → v4 label
# ---------------------------------------------------------------------------
# Patterns are matched against ``arabic_normalize(raw)`` output, so use
# bare ا and bare ي (alif + ya folded). Order matters: most-specific first.
#
# Strategy: re-uses v3 canonicalisation for everything that DOESN'T split,
# and adds 9 new pattern groups for the splits. The split patterns must run
# BEFORE the v3 generic patterns so a raw "ظرف زمان" matches "dharf_zaman"
# instead of falling through to v3's generic "ظرف" → "dharf".
# ---------------------------------------------------------------------------
_V4_PATTERNS: List[tuple[re.Pattern, str]] = [
    # === splits — most specific first ===
    # dharf splits
    (re.compile(r"ظرف\s*زمان"), "dharf_zaman"),
    (re.compile(r"ظرف\s*مكان"), "dharf_makan"),
    # fil splits — note "ماضٍ" normalises to "ماض" (NFC + diacritic strip)
    (re.compile(r"فعل\s*ماض"),  "fil_madi"),
    (re.compile(r"فعل\s*مضارع"), "fil_mudari"),
    (re.compile(r"فعل\s*ناقص"), "fil_naqis"),
    # harf_other splits — order: most specific first; "حرف نفي وجزم" must
    # match "harf_nafy" (negation + jussive is a sub-type of negation, not
    # accusative-marker), so anchor on "نفي" first.
    (re.compile(r"حرف\s*نفي\s*وجزم"), "harf_nafy"),
    (re.compile(r"حرف\s*نفي"),         "harf_nafy"),
    (re.compile(r"حرف\s*توكيد\s*ونصب"), "harf_nasb"),
    (re.compile(r"حرف\s*ناصب"),         "harf_nasb"),
    (re.compile(r"^حرف\s*نصب$"),        "harf_nasb"),
    (re.compile(r"حرف\s*تحقيق"), "harf_tahqiq"),
    # mafoul splits
    (re.compile(r"مفعول\s*مطلق"), "mafoul_mutlaq"),
]


def canonicalize_role_v4(raw: Optional[str]) -> Optional[str]:
    """Map a raw Arabic role string to a v4 canonical label.

    Order:
      1) Try the 9 v4 split patterns (most specific first).
      2) If none match, delegate to the v3 canonicaliser; the v3 result
         either IS a v4 label (unchanged labels) or maps to a v4 parent
         residual (e.g. v3 ``dharf`` → v4 ``dharf`` residual).
    """
    if raw is None:
        return None
    s = arabic_normalize(raw)
    if not s:
        return None

    # Step 1: v4-specific patterns
    for rx, label in _V4_PATTERNS:
        if rx.search(s):
            return label

    # Step 2: fall back to v3
    from .schema import canonicalize_role as canonicalize_role_v3
    v3_label = canonicalize_role_v3(raw)
    if v3_label is None:
        return None
    # The v3 label might already be a v4 residual (e.g. "fil", "dharf",
    # "harf_other", "mafoul_other") — those names match v4 directly.
    if v3_label in ROLE_TO_ID_V4:
        return v3_label
    # If v3 returned something we don't know, route to v4 "other" defensively
    return "other"


# ---------------------------------------------------------------------------
# Auto-fallback policy (Phase 4a §16): identify labels at risk
# ---------------------------------------------------------------------------
# Frozen list of labels that may auto-fallback to their parent during eval
# if their held-out support < 50 OR per-class F1 < 60%. Used by
# ``scripts/structured/eval_phase4.py`` to construct the "stable" macro.
AUTO_FALLBACK_AT_RISK: List[str] = ["harf_tahqiq"]


def auto_fallback(v4_label: str, support: int, f1: float, *,
                  min_support: int = 50, min_f1: float = 0.60) -> str:
    """Return v4_label normally, or its v3 parent if support/F1 trigger fallback.

    Used by the eval pipeline to construct the "stable" macro view per §16.
    """
    if v4_label in AUTO_FALLBACK_AT_RISK and (support < min_support or f1 < min_f1):
        return NEW_TO_OLD.get(v4_label, "other")
    return v4_label
