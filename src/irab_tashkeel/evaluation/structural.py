"""Structural i'rāb evaluator.

Extracts {pos, case, role, marker} from a generated Arabic i'rāb string via a
tolerant regex pipeline, then computes:

  * well_formed_rate — fraction of strings parsed (case OR mabni recovered)
  * case_accuracy    — case match (rafʿ/naṣb/jarr/jazm/mabni)
  * role_f1          — macro-F1 over the role set
  * marker_em        — exact marker match (الضمة، الفتحة، …)
  * chrF             — chrF score (auxiliary, not primary)

This is the metric track for the research plan's "structural extraction"
protocol — string-similarity scores like BLEU give partial credit to wrong
case endings, which is misleading for morphology.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Closed taxonomies
# ---------------------------------------------------------------------------
# Order matters: longer/more-specific terms first so partial substrings don't
# match a shorter term first (e.g. "اسم علم" before "اسم").
POS_TERMS = [
    "فعل ماضٍ", "فعل ماض", "فعل مضارع", "فعل أمر",
    "اسم علم", "اسم إشارة", "اسم موصول", "اسم فعل",
    "أداة تعريف",
    "حرف جر", "حرف عطف", "حرف نفي", "حرف استفهام",
    "حرف توكيد", "حرف نداء", "حرف مصدري", "حرف شرط",
    "حرف استثناء", "حرف استقبال", "حرف",
    "صفة", "ضمير", "اسم", "فعل",
]

CASES = {
    "marfu":  ["مرفوع", "رفعه", "في محل رفع"],
    "mansub": ["منصوب", "نصبه", "في محل نصب"],
    "majrur": ["مجرور", "جره",  "في محل جر"],
    "majzum": ["مجزوم", "جزمه"],
    "mabni":  ["مبني", "مبنية"],
}

ROLES = [
    "نائب فاعل", "مفعول به", "مفعول فيه", "مفعول مطلق", "مفعول لأجله", "مفعول معه",
    "اسم إن", "خبر إن", "اسم كان", "خبر كان",
    "اسم مجرور بحرف الجر", "اسم مجرور",
    "مضاف إليه", "خبر جملة", "خبر",
    "مبتدأ", "فاعل",
    "نعت", "صفة", "حال", "تمييز", "بدل", "عطف بيان",
    "ظرف زمان", "ظرف مكان", "ظرف", "منادى",
]

MARKERS = [
    "تنوين الفتح", "تنوين الضم", "تنوين الكسر",
    "الضمة المقدرة", "الفتحة المقدرة", "الكسرة المقدرة",
    "الضمة الظاهرة", "الفتحة الظاهرة", "الكسرة الظاهرة",
    "الضمة", "الفتحة", "الكسرة", "السكون",
    "الواو", "الألف", "الياء", "النون",
]

# Pre-compile regex per term. Use word-boundary anchors so a short term
# (e.g. "حرف") doesn't accidentally match inside a longer prefixed token
# (e.g. "بحرف"). \b in Python's regex engine works with Arabic letters.
def _compile(terms: Iterable[str]) -> List[Tuple[str, re.Pattern]]:
    return [(t, re.compile(rf"\b{re.escape(t)}\b")) for t in terms]


_POS_PATTERNS = _compile(POS_TERMS)
# CONTAINED FIX (May 2026): sort ROLES patterns by descending length so that
# when both "اسم مجرور بحرف الجر" and "اسم مجرور" could match, the longer
# (more specific) one wins. The MASAQ role audit (docs/MASAQ_role_audit.md)
# found that 70% of role disagreements were vocabulary mismatches caused by
# this priority. Fix is intentionally CONTAINED to ROLES only — POS, CASE,
# and MARKER extraction are unchanged.
_ROLE_PATTERNS = _compile(sorted(ROLES, key=len, reverse=True))
_MARKER_PATTERNS = _compile(MARKERS)
_CASE_PATTERNS = {label: _compile(words) for label, words in CASES.items()}


def _normalize(s: str) -> str:
    """Strip diacritics + NFC + collapse whitespace for robust matching."""
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[ً-ْٰ]", "", s)  # diacritics + dagger alif
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
@dataclass
class IrabAnalysis:
    """Structured fields extracted from a single per-word i'rāb string."""
    pos: Optional[str] = None
    case: Optional[str] = None       # one of CASES keys (marfu/mansub/majrur/majzum/mabni) or None
    role: Optional[str] = None
    marker: Optional[str] = None
    well_formed: bool = False        # True iff at least case OR mabni recovered

    def to_dict(self) -> dict:
        return {
            "pos": self.pos, "case": self.case, "role": self.role,
            "marker": self.marker, "well_formed": self.well_formed,
        }


def extract(irab_text: str) -> IrabAnalysis:
    """Parse one i'rāb string into structured fields."""
    text = _normalize(irab_text)
    if not text:
        return IrabAnalysis()

    pos = next((t for t, p in _POS_PATTERNS if p.search(text)), None)

    case: Optional[str] = None
    for label, pats in _CASE_PATTERNS.items():
        if any(p.search(text) for _, p in pats):
            case = label
            break

    role = next((t for t, p in _ROLE_PATTERNS if p.search(text)), None)
    marker = next((t for t, p in _MARKER_PATTERNS if p.search(text)), None)

    well_formed = case is not None
    return IrabAnalysis(pos=pos, case=case, role=role, marker=marker, well_formed=well_formed)


# ---------------------------------------------------------------------------
# Per-corpus metrics
# ---------------------------------------------------------------------------
@dataclass
class StructuralMetrics:
    """Aggregated metrics over a list of (gold, pred) i'rāb-string pairs."""
    n: int = 0
    n_well_formed: int = 0
    n_case_correct: int = 0
    n_pos_correct: int = 0
    n_marker_correct: int = 0
    n_fully_correct: int = 0   # case ∧ role ∧ marker all match (the Mix A metric)
    role_confusion: Counter = field(default_factory=Counter)  # (gold_role, pred_role) -> count
    role_gold_count: Counter = field(default_factory=Counter)
    role_pred_count: Counter = field(default_factory=Counter)

    def update(self, gold_text: str, pred_text: str):
        self.n += 1
        gold = extract(gold_text)
        pred = extract(pred_text)

        if pred.well_formed:
            self.n_well_formed += 1
        case_ok = gold.case is not None and gold.case == pred.case
        pos_ok = gold.pos is not None and gold.pos == pred.pos
        marker_ok = gold.marker is not None and gold.marker == pred.marker
        role_ok = gold.role is not None and gold.role == pred.role
        if case_ok:
            self.n_case_correct += 1
        if pos_ok:
            self.n_pos_correct += 1
        if marker_ok:
            self.n_marker_correct += 1
        # "Fully correct word" = case + role + marker all right.
        # POS is excluded because gold often lacks an explicit POS phrase.
        if case_ok and role_ok and marker_ok:
            self.n_fully_correct += 1

        g_role = gold.role or "<none>"
        p_role = pred.role or "<none>"
        self.role_confusion[(g_role, p_role)] += 1
        self.role_gold_count[g_role] += 1
        self.role_pred_count[p_role] += 1

    def role_f1_macro(self) -> float:
        f1s = []
        roles = set(self.role_gold_count) | set(self.role_pred_count)
        for r in roles:
            if r == "<none>":
                continue
            tp = self.role_confusion.get((r, r), 0)
            fn = self.role_gold_count.get(r, 0) - tp
            fp = self.role_pred_count.get(r, 0) - tp
            denom = (2 * tp + fp + fn)
            if denom == 0:
                continue
            f1s.append(2 * tp / denom)
        return sum(f1s) / len(f1s) if f1s else 0.0

    def report(self) -> Dict[str, float]:
        if self.n == 0:
            return {}
        return {
            "n": float(self.n),
            "well_formed_rate":   self.n_well_formed / self.n,
            "case_accuracy":      self.n_case_correct / self.n,
            "pos_accuracy":       self.n_pos_correct / self.n,
            "marker_em":          self.n_marker_correct / self.n,
            "role_f1_macro":      self.role_f1_macro(),
            "fully_correct_word": self.n_fully_correct / self.n,
        }

    def pretty(self) -> str:
        r = self.report()
        if not r:
            return "(empty)"
        return (
            f"n={int(r['n'])}  "
            f"well-formed={r['well_formed_rate']:.3f}  "
            f"pos={r['pos_accuracy']:.3f}  "
            f"case={r['case_accuracy']:.3f}  "
            f"role-F1={r['role_f1_macro']:.3f}  "
            f"marker={r['marker_em']:.3f}  "
            f"fully={r['fully_correct_word']:.3f}"
        )


def evaluate_pairs(pairs: Iterable[Tuple[str, str]]) -> StructuralMetrics:
    """Convenience: pass an iterable of (gold, pred) pairs, get aggregated metrics."""
    m = StructuralMetrics()
    for gold, pred in pairs:
        m.update(gold, pred)
    return m


# ---------------------------------------------------------------------------
# Sentence-level evaluation (when output is a single multi-line answer)
# ---------------------------------------------------------------------------
_WORD_LINE_RE = re.compile(r"^\s*([ء-يٰٱً-ْ]+)\s*[:：]\s*(.+?)\s*$")


def split_sentence_iraab(answer: str) -> List[Tuple[str, str]]:
    """Split a sentence-level i'rāb answer into per-word (word, irab_line) pairs.

    Tolerates Gazelle/Yarob style multi-line layouts. Lines without a colon
    are skipped (continuations of the previous word's i'rāb merged in).
    """
    answer = unicodedata.normalize("NFC", answer or "")
    out: List[Tuple[str, str]] = []
    cur_word: Optional[str] = None
    cur_irab: List[str] = []
    for raw in answer.splitlines():
        m = _WORD_LINE_RE.match(raw)
        if m:
            if cur_word is not None:
                out.append((cur_word, " ".join(cur_irab).strip()))
            cur_word = m.group(1)
            cur_irab = [m.group(2)]
        elif cur_word is not None and raw.strip():
            cur_irab.append(raw.strip())
    if cur_word is not None:
        out.append((cur_word, " ".join(cur_irab).strip()))
    return out
