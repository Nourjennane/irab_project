"""Adversarial perturbation set + structural-extractor audit.

Builds a probe corpus by deterministically corrupting each Gazelle gold
i'rāb string in a single dimension at a time (case, role, or marker),
then runs the structural extractor on the corrupted output to verify the
extractor flags the corruption on the right field while leaving the other
fields unchanged.

Two reported numbers:
  - **Sensitivity** = P(extractor disagrees with original gold on the
    corrupted field | corrupted_field != none).
  - **Specificity** = P(extractor matches gold on every field | the record
    is the unmodified original gold).

Sensitivity bound: corruptions are designed to actually change the
extractor's output on the targeted field; failures here mean the
extractor pattern set is incomplete (e.g., a synonym for مرفوع that the
regex misses), which is a real metric-validity finding.

This module does NOT call any LLM. It does not change Sonnet RAG's
headline. Its purpose is to audit the SCORER, not the model.

Usage:
    python -m irab_tashkeel.evaluation.perturb --build      # writes data/perturbed_eval.jsonl
    python -m irab_tashkeel.evaluation.perturb --audit      # writes runs/perturbation_audit/summary.json
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..data.gazelle import load_gazelle_iraab
from .structural import extract, split_sentence_iraab


# ---------------------------------------------------------------------------
# Substitution tables
# ---------------------------------------------------------------------------
# Each (rule, replacement) pair: applies str.replace(rule, replacement, 1).
# Order within a category matters: earlier rules win when both match.

CASE_FLIPS: List[Tuple[List[str], List[str]]] = [
    # rafʿ ↔ naṣb
    (
        ["مرفوع وعلامة رفعه الضمة الظاهرة", "مرفوع وعلامة رفعه الضمة"],
        ["منصوب وعلامة نصبه الفتحة الظاهرة", "منصوب وعلامة نصبه الفتحة"],
    ),
    (
        ["منصوب وعلامة نصبه الفتحة الظاهرة", "منصوب وعلامة نصبه الفتحة"],
        ["مرفوع وعلامة رفعه الضمة الظاهرة", "مرفوع وعلامة رفعه الضمة"],
    ),
    # jarr → naṣb
    (
        ["مجرور وعلامة جره الكسرة الظاهرة", "مجرور وعلامة جره الكسرة"],
        ["منصوب وعلامة نصبه الفتحة الظاهرة", "منصوب وعلامة نصبه الفتحة"],
    ),
    # jazm ↔ naṣb (verbs)
    (
        ["مجزوم وعلامة جزمه السكون"],
        ["منصوب وعلامة نصبه الفتحة الظاهرة"],
    ),
]

ROLE_FLIPS: List[Tuple[str, str]] = [
    ("فاعل", "مفعول به"),
    ("مفعول به", "فاعل"),
    ("مبتدأ", "خبر"),
    ("خبر", "مبتدأ"),
    ("مضاف إليه", "نعت"),
    ("نعت", "مضاف إليه"),
    ("اسم مجرور", "ظرف"),
]

MARKER_MANGLES: List[Tuple[str, str]] = [
    # Within-rafʿ: ضمة → واو (sound-masculine-plural marker)
    ("الضمة الظاهرة", "الواو"),
    # Within-naṣb: فتحة → ألف (dual marker)
    ("الفتحة الظاهرة", "الألف"),
    # Within-jarr: كسرة → ياء (sound-masculine-plural-jarr marker)
    ("الكسرة الظاهرة", "الياء"),
    # Within-jazm: سكون → حذف النون
    ("السكون", "حذف النون"),
]


@dataclass
class PerturbedRecord:
    sentence: str
    word: str
    gold_irab: str
    perturbed_irab: str
    corrupted_field: str            # "case" | "role" | "marker" | "none"
    corruption_rule: Optional[str] = None  # description of the rule applied
    extracted_gold_case: Optional[str] = None
    extracted_gold_role: Optional[str] = None
    extracted_gold_marker: Optional[str] = None
    extracted_pred_case: Optional[str] = None
    extracted_pred_role: Optional[str] = None
    extracted_pred_marker: Optional[str] = None


# ---------------------------------------------------------------------------
# Perturbation engine
# ---------------------------------------------------------------------------
def _try_case_flip(irab: str) -> Optional[Tuple[str, str]]:
    """Attempt one case flip; return (new_irab, rule_desc) or None."""
    for src_terms, dst_terms in CASE_FLIPS:
        for src in src_terms:
            if src in irab:
                # Pair with the first dst that starts the same way (we just want one swap)
                dst = dst_terms[0]
                return irab.replace(src, dst, 1), f"case:{src}→{dst}"
    return None


def _try_role_flip(irab: str) -> Optional[Tuple[str, str]]:
    for src, dst in ROLE_FLIPS:
        if src in irab:
            return irab.replace(src, dst, 1), f"role:{src}→{dst}"
    return None


def _try_marker_mangle(irab: str) -> Optional[Tuple[str, str]]:
    for src, dst in MARKER_MANGLES:
        if src in irab:
            return irab.replace(src, dst, 1), f"marker:{src}→{dst}"
    return None


def perturb_one(word: str, gold_irab: str, sentence: str) -> List[PerturbedRecord]:
    """Build up to 4 records per word: control + (case|role|marker) when applicable."""
    out: List[PerturbedRecord] = []

    g = extract(gold_irab)

    def make_record(perturbed: str, field: str, rule: Optional[str]) -> PerturbedRecord:
        p = extract(perturbed)
        return PerturbedRecord(
            sentence=sentence, word=word, gold_irab=gold_irab,
            perturbed_irab=perturbed, corrupted_field=field,
            corruption_rule=rule,
            extracted_gold_case=g.case, extracted_gold_role=g.role,
            extracted_gold_marker=g.marker,
            extracted_pred_case=p.case, extracted_pred_role=p.role,
            extracted_pred_marker=p.marker,
        )

    # 1. control (uncorrupted)
    out.append(make_record(gold_irab, "none", None))

    # 2. case flip
    if (cf := _try_case_flip(gold_irab)) is not None:
        out.append(make_record(cf[0], "case", cf[1]))
    # 3. role flip
    if (rf := _try_role_flip(gold_irab)) is not None:
        out.append(make_record(rf[0], "role", rf[1]))
    # 4. marker mangle
    if (mm := _try_marker_mangle(gold_irab)) is not None:
        out.append(make_record(mm[0], "marker", mm[1]))

    return out


def build_perturbed_set(out_path: Path | str = "data/perturbed_eval.jsonl") -> Path:
    items = load_gazelle_iraab()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    n_words = 0
    n_by_field: Counter = Counter()
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            pairs = split_sentence_iraab(it.answer)
            for word, gold_irab in pairs:
                if not gold_irab:
                    continue
                n_words += 1
                for rec in perturb_one(word, gold_irab, it.sentence):
                    f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    n_records += 1
                    n_by_field[rec.corrupted_field] += 1
    print(f"wrote {n_records} records ({n_words} source words) → {out}")
    print(f"  by field: {dict(n_by_field)}")
    return out


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def audit(perturbed_path: Path | str = "data/perturbed_eval.jsonl",
          out_path: Path | str = "runs/perturbation_audit/summary.json") -> Dict:
    """For each perturbed record, check that the extractor flags the right field."""
    perturbed_path = Path(perturbed_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        "controls": {"n": 0, "all_match": 0},
        "case":   {"n": 0, "case_flagged": 0, "role_unchanged": 0, "marker_unchanged": 0},
        "role":   {"n": 0, "role_flagged": 0, "case_unchanged": 0, "marker_unchanged": 0},
        "marker": {"n": 0, "marker_flagged": 0, "case_unchanged": 0, "role_unchanged": 0},
    }
    misses: List[Dict] = []

    with open(perturbed_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            field = r["corrupted_field"]

            gc, gr, gm = r["extracted_gold_case"], r["extracted_gold_role"], r["extracted_gold_marker"]
            pc, pr, pm = r["extracted_pred_case"], r["extracted_pred_role"], r["extracted_pred_marker"]

            if field == "none":
                counts["controls"]["n"] += 1
                if pc == gc and pr == gr and pm == gm:
                    counts["controls"]["all_match"] += 1
                else:
                    if len(misses) < 50:
                        misses.append({"kind": "control_mismatch", "rec": r})
                continue

            counts[field]["n"] += 1
            if field == "case":
                if pc != gc:
                    counts[field]["case_flagged"] += 1
                if pr == gr:
                    counts[field]["role_unchanged"] += 1
                if pm == gm:
                    counts[field]["marker_unchanged"] += 1
                if pc == gc and len(misses) < 50:
                    misses.append({"kind": "case_not_flagged", "rec": r})
            elif field == "role":
                if pr != gr:
                    counts[field]["role_flagged"] += 1
                if pc == gc:
                    counts[field]["case_unchanged"] += 1
                if pm == gm:
                    counts[field]["marker_unchanged"] += 1
                if pr == gr and len(misses) < 50:
                    misses.append({"kind": "role_not_flagged", "rec": r})
            elif field == "marker":
                if pm != gm:
                    counts[field]["marker_flagged"] += 1
                if pc == gc:
                    counts[field]["case_unchanged"] += 1
                if pr == gr:
                    counts[field]["role_unchanged"] += 1
                if pm == gm and len(misses) < 50:
                    misses.append({"kind": "marker_not_flagged", "rec": r})

    def safe_pct(num: int, den: int) -> float:
        return (num / den * 100.0) if den else 0.0

    summary = {
        "specificity_controls": {
            "n": counts["controls"]["n"],
            "all_match_rate_pct": safe_pct(counts["controls"]["all_match"], counts["controls"]["n"]),
        },
        "sensitivity_per_field": {
            "case":   {
                "n": counts["case"]["n"],
                "flagged_pct":          safe_pct(counts["case"]["case_flagged"],   counts["case"]["n"]),
                "role_unchanged_pct":   safe_pct(counts["case"]["role_unchanged"], counts["case"]["n"]),
                "marker_unchanged_pct": safe_pct(counts["case"]["marker_unchanged"], counts["case"]["n"]),
            },
            "role":   {
                "n": counts["role"]["n"],
                "flagged_pct":          safe_pct(counts["role"]["role_flagged"],   counts["role"]["n"]),
                "case_unchanged_pct":   safe_pct(counts["role"]["case_unchanged"], counts["role"]["n"]),
                "marker_unchanged_pct": safe_pct(counts["role"]["marker_unchanged"], counts["role"]["n"]),
            },
            "marker": {
                "n": counts["marker"]["n"],
                "flagged_pct":          safe_pct(counts["marker"]["marker_flagged"], counts["marker"]["n"]),
                "case_unchanged_pct":   safe_pct(counts["marker"]["case_unchanged"], counts["marker"]["n"]),
                "role_unchanged_pct":   safe_pct(counts["marker"]["role_unchanged"], counts["marker"]["n"]),
            },
        },
        "misses_sample": misses[:20],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Pretty print
    print(json.dumps({k: v for k, v in summary.items() if k != "misses_sample"},
                     ensure_ascii=False, indent=2))
    print(f"\nwrote → {out_path}")
    return summary


def main():
    import argparse
    p = argparse.ArgumentParser(description="Adversarial perturbation set + extractor audit")
    p.add_argument("--build", action="store_true", help="generate data/perturbed_eval.jsonl")
    p.add_argument("--audit", action="store_true", help="run the extractor audit")
    args = p.parse_args()
    if args.build:
        build_perturbed_set()
    if args.audit:
        audit()
    if not args.build and not args.audit:
        build_perturbed_set()
        audit()


if __name__ == "__main__":
    main()
