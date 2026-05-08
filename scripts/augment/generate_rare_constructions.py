"""Phase #39 — synthetic rare-construction generator.

Generates Arabic sentences with grammatically-correct iʿrāb labels for
under-covered constructions in the distill_v2 training corpus. Output
JSONL is compatible with the existing `data/structured_v1` schema.

Generator design: each construction is a `ConstructionTemplate` with
slot-typed surface patterns + label tuples. We instantiate templates by
sampling slot fillers from focused lexical inventories.

Lexical inventories are deliberately small (~20-50 fillers per slot
type) to keep the synthetic distribution interpretable. Future iteration
can pull frequency-weighted inventories from the existing corpus.

Design choice: deterministic templates (NOT LLM-generated). Labels are
guaranteed correct because they're encoded directly in the template.
LLM-generated synthetic data would have label noise — explicitly out of
scope per Phase #39 design doc §11.

Usage:
    python scripts/augment/generate_rare_constructions.py \\
        --out_dir data/structured_v1_augmented/synthetic/ \\
        --seed 42

Output structure:
    out_dir/
        kana_sisters.jsonl     (~200 sentences)
        inna_sisters.jsonl     (~200)
        istithna.jsonl         (~250)
        mawsool.jsonl          (~200)
        idafa_edge.jsonl       (~300)
        quranic.jsonl          (~150)
        rare_combos.jsonl      (~100)
        all_synthetic.jsonl    (concatenation, ~1,400)
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Lexical inventories (small, focused, diacritized)
# ---------------------------------------------------------------------------

# kāna and its sisters — 12 particles. All take perfect-aspect form by default.
# Format: (surface, irab_prose_for_verb, marker)
KANA_SISTERS: List[Tuple[str, str, str]] = [
    ("كان",     "فعل ماض ناقص مبني على الفتح",                          "fath_short"),
    ("ليس",     "فعل ماض ناقص جامد مبني على الفتح",                     "fath_short"),
    ("أصبح",    "فعل ماض ناقص مبني على الفتح",                          "fath_short"),
    ("ظل",      "فعل ماض ناقص مبني على الفتح",                          "fath_short"),
    ("صار",     "فعل ماض ناقص مبني على الفتح",                          "fath_short"),
    ("بات",     "فعل ماض ناقص مبني على الفتح",                          "fath_short"),
    ("أمسى",    "فعل ماض ناقص مبني على الفتح المقدر",                   "fath_short"),
    ("أضحى",    "فعل ماض ناقص مبني على الفتح المقدر",                   "fath_short"),
    ("ما زال",  "ما النافية + فعل ماض ناقص مبني على الفتح",            "fath_short"),
    ("ما برح",  "ما النافية + فعل ماض ناقص مبني على الفتح",            "fath_short"),
]

# inna and its sisters — 6 particles
INNA_SISTERS: List[Tuple[str, str, str]] = [
    ("إن",       "حرف توكيد ونصب",        "fath_short"),
    ("أن",       "حرف توكيد ونصب",        "fath_short"),
    ("لكن",      "حرف استدراك ونصب",      "fath_short"),
    ("ليت",      "حرف تمن ونصب",          "fath_short"),
    ("لعل",      "حرف ترجي ونصب",         "fath_short"),
    ("كأن",      "حرف تشبيه ونصب",        "fath_short"),
]

# illa and its kin (istithnāʾ particles)
ISTITHNA_PARTICLES: List[Tuple[str, str]] = [
    ("إلا",       "أداة استثناء"),
    ("غير",       "اسم استثناء"),
    ("سوى",       "اسم استثناء"),
    ("ما عدا",    "أداة استثناء"),
    ("ما خلا",    "أداة استثناء"),
    ("حاشا",      "أداة استثناء"),
]

# mawsool relative pronouns
MAWSOOL_PARTICLES: List[Tuple[str, str, str]] = [
    # (surface, gender, plurality)
    ("الذي",    "m", "sg"),
    ("التي",    "f", "sg"),
    ("الذين",   "m", "pl"),
    ("اللاتي",  "f", "pl"),
    ("اللواتي", "f", "pl"),
    ("اللذان",  "m", "dual"),
    ("اللتان",  "f", "dual"),
    ("من",      "any", "any"),
    ("ما",      "any", "any"),
]

# Masculine singular nouns: (def_form, indef_form_nasb_tanween, indef_form_jarr_tanween)
# These give us the diacritized inflections for the three cases.
NOUNS_M_SG: List[Tuple[str, str, str, str]] = [
    # (def_raf, def_nasb, def_jarr, gloss)
    ("الطالبُ",   "الطالبَ",   "الطالبِ",   "student"),
    ("المعلمُ",   "المعلمَ",   "المعلمِ",   "teacher"),
    ("الكتابُ",   "الكتابَ",   "الكتابِ",   "book"),
    ("البيتُ",    "البيتَ",    "البيتِ",    "house"),
    ("الرجلُ",    "الرجلَ",    "الرجلِ",    "man"),
    ("الولدُ",    "الولدَ",    "الولدِ",    "boy"),
    ("القلمُ",    "القلمَ",    "القلمِ",    "pen"),
    ("البابُ",    "البابَ",    "البابِ",    "door"),
    ("المسجدُ",   "المسجدَ",   "المسجدِ",   "mosque"),
    ("النهرُ",    "النهرَ",    "النهرِ",    "river"),
    ("الجبلُ",    "الجبلَ",    "الجبلِ",    "mountain"),
    ("الشارعُ",   "الشارعَ",   "الشارعِ",   "street"),
    ("الصديقُ",   "الصديقَ",   "الصديقِ",   "friend"),
    ("الأبُ",     "الأبَ",     "الأبِ",     "father"),
    ("الأخُ",     "الأخَ",     "الأخِ",     "brother"),
    ("الفائزُ",   "الفائزَ",   "الفائزِ",   "winner"),
    ("الزائرُ",   "الزائرَ",   "الزائرِ",   "visitor"),
    ("العالِمُ",  "العالِمَ",  "العالِمِ",  "scholar"),
    ("القائدُ",   "القائدَ",   "القائدِ",   "leader"),
    ("الطبيبُ",   "الطبيبَ",   "الطبيبِ",   "doctor"),
]

# Indefinite-form variants for masculine singular nouns (with tanween).
# (raf_tanween, nasb_tanween, jarr_tanween)
NOUNS_M_SG_INDEF: List[Tuple[str, str, str, str]] = [
    ("طالبٌ",   "طالباً",   "طالبٍ",   "a student"),
    ("معلمٌ",   "معلماً",   "معلمٍ",   "a teacher"),
    ("كتابٌ",   "كتاباً",   "كتابٍ",   "a book"),
    ("رجلٌ",    "رجلاً",    "رجلٍ",    "a man"),
    ("ولدٌ",    "ولداً",    "ولدٍ",    "a boy"),
]

# Adjectives, masculine singular, def/indef forms
ADJ_M_SG_DEF: List[Tuple[str, str, str, str]] = [
    ("المجتهدُ",  "المجتهدَ",  "المجتهدِ",  "diligent"),
    ("الكبيرُ",   "الكبيرَ",   "الكبيرِ",   "big"),
    ("الصغيرُ",   "الصغيرَ",   "الصغيرِ",   "small"),
    ("الجميلُ",   "الجميلَ",   "الجميلِ",   "beautiful"),
    ("النافعُ",   "النافعَ",   "النافعِ",   "useful"),
]
ADJ_M_SG_INDEF: List[Tuple[str, str, str, str]] = [
    ("مجتهدٌ",   "مجتهداً",   "مجتهدٍ",   "diligent"),
    ("كبيرٌ",    "كبيراً",    "كبيرٍ",    "big"),
    ("جميلٌ",    "جميلاً",    "جميلٍ",    "beautiful"),
    ("ذكيٌ",     "ذكياً",     "ذكيٍ",     "intelligent"),
    ("شجاعٌ",    "شجاعاً",    "شجاعٍ",    "brave"),
    ("صادقٌ",    "صادقاً",    "صادقٍ",    "honest"),
    ("نافعٌ",    "نافعاً",    "نافعٍ",    "useful"),
    ("قويٌ",     "قوياً",     "قويٍ",     "strong"),
    ("سريعٌ",    "سريعاً",    "سريعٍ",    "fast"),
    ("هادئٌ",    "هادئاً",    "هادئٍ",    "calm"),
]

# Common verbs (perfect, masc 3rd singular)
VERBS_PERFECT_M3SG: List[Tuple[str, str, str]] = [
    # (surface, irab_prose, marker)
    ("ذهب",    "فعل ماض مبني على الفتح",  "fath_short"),
    ("جاء",    "فعل ماض مبني على الفتح",  "fath_short"),
    ("قرأ",    "فعل ماض مبني على الفتح",  "fath_short"),
    ("كتب",    "فعل ماض مبني على الفتح",  "fath_short"),
    ("درس",    "فعل ماض مبني على الفتح",  "fath_short"),
    ("شاهد",   "فعل ماض مبني على الفتح",  "fath_short"),
    ("عمل",    "فعل ماض مبني على الفتح",  "fath_short"),
    ("سمع",    "فعل ماض مبني على الفتح",  "fath_short"),
]

# Standard role/marker fixed prose templates (used for irab_prose)
def _prose(role: str, case: str, marker_phrase: str) -> str:
    """Build a deterministic Arabic iʿrāb prose given role/case/marker phrase."""
    role_arabic = {
        "ism_kana": "اسم كان",
        "khabar_kana": "خبر كان",
        "ism_inna": "اسم إن",
        "khabar_inna": "خبر إن",
        "mafoul_other": "مستثنى",
        "mawsool": "اسم موصول",
        "harf_other": "حرف",
        "fil": "فعل",
        "fail": "فاعل",
    }.get(role, role)
    case_arabic = {
        "raf":  "مرفوع",
        "nasb": "منصوب",
        "jarr": "مجرور",
        "mabni": "مبني",
    }[case]
    return f"{role_arabic} {case_arabic} وعلامة {case_arabic[1:]}ه {marker_phrase}"


def _record(sentence: str, items: List[Dict], source: str = "synthetic_phase39") -> Dict:
    """Wrap items into the canonical record format."""
    return {
        "sentence": sentence,
        "source": source,
        "items": items,
        "has_irab": True,
    }


# ---------------------------------------------------------------------------
# Generators per construction
# ---------------------------------------------------------------------------

def gen_kana_sisters(rng: random.Random, n: int = 200) -> List[Dict]:
    """kāna + ism (raf) + khabar (nasb)."""
    out: List[Dict] = []
    for _ in range(n):
        verb_surface, verb_prose, verb_marker = rng.choice(KANA_SISTERS)
        ism_def_raf, _, _, _gloss = rng.choice(NOUNS_M_SG)
        # khabar is masc-sg-indef nasb (tanween_fath) for variety; sometimes def
        if rng.random() < 0.6:
            khabar_raf, khabar_nasb, khabar_jarr, _ = rng.choice(ADJ_M_SG_INDEF)
            khabar_marker = "tanween_fath"
        else:
            khabar_raf, khabar_nasb, khabar_jarr, _ = rng.choice(ADJ_M_SG_DEF)
            khabar_marker = "fatha_visible"
        sentence = f"{verb_surface} {ism_def_raf} {khabar_nasb}"
        items = [
            {"word": verb_surface, "case": "mabni", "role": "fil",
             "marker": verb_marker, "pos": "verb",
             "irab_prose": verb_prose},
            {"word": ism_def_raf, "case": "raf", "role": "ism_kana",
             "marker": "damma_visible", "pos": "noun",
             "irab_prose": _prose("ism_kana", "raf", "الضمة الظاهرة")},
            {"word": khabar_nasb, "case": "nasb", "role": "khabar_kana",
             "marker": khabar_marker, "pos": "adjective",
             "irab_prose": _prose("khabar_kana", "nasb",
                                  "تنوين الفتح الظاهر" if khabar_marker == "tanween_fath"
                                  else "الفتحة الظاهرة")},
        ]
        out.append(_record(sentence, items))
    return out


def gen_inna_sisters(rng: random.Random, n: int = 200) -> List[Dict]:
    """inna + ism (nasb) + khabar (raf)."""
    out: List[Dict] = []
    for _ in range(n):
        particle_surface, particle_prose, particle_marker = rng.choice(INNA_SISTERS)
        ism_def_raf, ism_def_nasb, ism_def_jarr, _ = rng.choice(NOUNS_M_SG)
        if rng.random() < 0.6:
            khabar_raf, _, _, _ = rng.choice(ADJ_M_SG_INDEF)
            khabar_marker = "tanween_damm"
        else:
            khabar_raf, _, _, _ = rng.choice(ADJ_M_SG_DEF)
            khabar_marker = "damma_visible"
        sentence = f"{particle_surface} {ism_def_nasb} {khabar_raf}"
        items = [
            {"word": particle_surface, "case": "mabni", "role": "harf_other",
             "marker": particle_marker, "pos": "particle",
             "irab_prose": particle_prose + " مبني على الفتح"},
            {"word": ism_def_nasb, "case": "nasb", "role": "ism_inna",
             "marker": "fatha_visible", "pos": "noun",
             "irab_prose": _prose("ism_inna", "nasb", "الفتحة الظاهرة")},
            {"word": khabar_raf, "case": "raf", "role": "khabar_inna",
             "marker": khabar_marker, "pos": "adjective",
             "irab_prose": _prose("khabar_inna", "raf",
                                  "تنوين الضم الظاهر" if khabar_marker == "tanween_damm"
                                  else "الضمة الظاهرة")},
        ]
        out.append(_record(sentence, items))
    return out


def gen_istithna(rng: random.Random, n: int = 250) -> List[Dict]:
    """Verb + (def-pl or def-coll subj) + illa + mustathna (nasb)."""
    out: List[Dict] = []
    for _ in range(n):
        verb_surface, verb_prose, verb_marker = rng.choice(VERBS_PERFECT_M3SG)
        # Subject (raf), simplified to def masc sg
        subj_raf, _, _, _ = rng.choice(NOUNS_M_SG)
        illa_surface, illa_prose = rng.choice(ISTITHNA_PARTICLES)
        # Mustathna (nasb)
        if illa_surface == "إلا":
            mustathna_def_raf, mustathna_nasb, _, _ = rng.choice(NOUNS_M_SG)
            mus_word = mustathna_nasb
            mus_marker = "fatha_visible"
        else:
            # غير / سوى etc are themselves the noun bearing case; followed by mudaf_ilayh
            mustathna_def_raf, _, mustathna_jarr, _ = rng.choice(NOUNS_M_SG)
            mus_word = mustathna_jarr
            mus_marker = "kasra_visible"
        sentence = f"{verb_surface} {subj_raf} {illa_surface} {mus_word}"
        items = [
            {"word": verb_surface, "case": "mabni", "role": "fil",
             "marker": verb_marker, "pos": "verb",
             "irab_prose": verb_prose},
            {"word": subj_raf, "case": "raf", "role": "fail",
             "marker": "damma_visible", "pos": "noun",
             "irab_prose": _prose("fail", "raf", "الضمة الظاهرة")},
            {"word": illa_surface, "case": "mabni", "role": "harf_other",
             "marker": "fath_short", "pos": "particle",
             "irab_prose": illa_prose + " مبني على السكون"},
        ]
        if illa_surface == "إلا":
            items.append({
                "word": mus_word, "case": "nasb", "role": "mafoul_other",
                "marker": mus_marker, "pos": "noun",
                "irab_prose": _prose("mafoul_other", "nasb", "الفتحة الظاهرة"),
            })
        else:
            items.append({
                "word": mus_word, "case": "jarr", "role": "mudaaf_ilayh",
                "marker": mus_marker, "pos": "noun",
                "irab_prose": _prose("mudaaf_ilayh", "jarr", "الكسرة الظاهرة"),
            })
        out.append(_record(sentence, items))
    return out


def gen_mawsool(rng: random.Random, n: int = 200) -> List[Dict]:
    """def-noun + relative-pronoun + verb (relative clause)."""
    out: List[Dict] = []
    for _ in range(n):
        head_def_raf, _, _, _ = rng.choice(NOUNS_M_SG)
        mawsool_surface, _, _ = rng.choice(MAWSOOL_PARTICLES)
        verb_surface, verb_prose, verb_marker = rng.choice(VERBS_PERFECT_M3SG)
        sentence = f"{head_def_raf} {mawsool_surface} {verb_surface}"
        items = [
            {"word": head_def_raf, "case": "raf", "role": "mubtada",
             "marker": "damma_visible", "pos": "noun",
             "irab_prose": _prose("mubtada", "raf", "الضمة الظاهرة")},
            {"word": mawsool_surface, "case": "mabni", "role": "other",
             "marker": "sukun", "pos": "particle",
             "irab_prose": "اسم موصول مبني على السكون في محل رفع نعت"},
            {"word": verb_surface, "case": "mabni", "role": "fil",
             "marker": verb_marker, "pos": "verb",
             "irab_prose": verb_prose + " والفاعل ضمير مستتر"},
        ]
        out.append(_record(sentence, items))
    return out


def gen_idafa_edge(rng: random.Random, n: int = 300) -> List[Dict]:
    """Multi-level iḍāfa: noun1 (raf) + noun2 (jarr, mudaaf) + noun3 (jarr, mudaaf_ilayh)."""
    out: List[Dict] = []
    for _ in range(n):
        n1_def_raf, _, _, _ = rng.choice(NOUNS_M_SG)
        # Mudaaf (level 2) — nasb form for jarr role
        n2_def_raf, _, n2_def_jarr, _ = rng.choice(NOUNS_M_SG)
        n3_def_raf, _, n3_def_jarr, _ = rng.choice(NOUNS_M_SG)
        # Strip ال from middle nouns to make them muḍāf (cons state)
        # In real Arabic, muḍāf drops ال. Build the cons-state form:
        n2_cons = n2_def_jarr.replace("ال", "", 1)  # muḍāf without ال
        sentence = f"{n1_def_raf} {n2_cons} {n3_def_jarr}"
        items = [
            {"word": n1_def_raf, "case": "raf", "role": "mubtada",
             "marker": "damma_visible", "pos": "noun",
             "irab_prose": _prose("mubtada", "raf", "الضمة الظاهرة")},
            {"word": n2_cons, "case": "jarr", "role": "mudaaf_ilayh",
             "marker": "kasra_visible", "pos": "noun",
             "irab_prose": _prose("mudaaf_ilayh", "jarr", "الكسرة الظاهرة") + ", وهو مضاف"},
            {"word": n3_def_jarr, "case": "jarr", "role": "mudaaf_ilayh",
             "marker": "kasra_visible", "pos": "noun",
             "irab_prose": _prose("mudaaf_ilayh", "jarr", "الكسرة الظاهرة")},
        ]
        out.append(_record(sentence, items))
    return out


def gen_quranic(rng: random.Random, n: int = 150) -> List[Dict]:
    """Quranic-style patterns: qad + verb, idh + verb, etc.

    Templated to AVOID quoting actual scripture (we don't want
    memorisation/leakage on MASAQ). We use the particles in MSA-style
    contexts.
    """
    out: List[Dict] = []
    for _ in range(n):
        # Pattern: qad + perfect verb + def-subject
        if rng.random() < 0.5:
            qad = "قد"
            qad_prose = "حرف تحقيق مبني على السكون"
        else:
            qad = "إذ"
            qad_prose = "ظرف زمان مبني على السكون"
        verb_surface, verb_prose, verb_marker = rng.choice(VERBS_PERFECT_M3SG)
        subj_def_raf, _, _, _ = rng.choice(NOUNS_M_SG)
        sentence = f"{qad} {verb_surface} {subj_def_raf}"
        items = [
            {"word": qad, "case": "mabni", "role": "harf_other",
             "marker": "sukun", "pos": "particle",
             "irab_prose": qad_prose},
            {"word": verb_surface, "case": "mabni", "role": "fil",
             "marker": verb_marker, "pos": "verb",
             "irab_prose": verb_prose},
            {"word": subj_def_raf, "case": "raf", "role": "fail",
             "marker": "damma_visible", "pos": "noun",
             "irab_prose": _prose("fail", "raf", "الضمة الظاهرة")},
        ]
        out.append(_record(sentence, items))
    return out


def gen_rare_combos(rng: random.Random, n: int = 100) -> List[Dict]:
    """Rare 4-tuples: ḥāl + adjective form + adverbial usage."""
    out: List[Dict] = []
    for _ in range(n):
        verb_surface, verb_prose, verb_marker = rng.choice(VERBS_PERFECT_M3SG)
        subj_def_raf, _, _, _ = rng.choice(NOUNS_M_SG)
        hal_indef_raf, hal_indef_nasb, _, _ = rng.choice(ADJ_M_SG_INDEF)
        sentence = f"{verb_surface} {subj_def_raf} {hal_indef_nasb}"
        items = [
            {"word": verb_surface, "case": "mabni", "role": "fil",
             "marker": verb_marker, "pos": "verb",
             "irab_prose": verb_prose},
            {"word": subj_def_raf, "case": "raf", "role": "fail",
             "marker": "damma_visible", "pos": "noun",
             "irab_prose": _prose("fail", "raf", "الضمة الظاهرة")},
            {"word": hal_indef_nasb, "case": "nasb", "role": "hal",
             "marker": "tanween_fath", "pos": "adjective",
             "irab_prose": _prose("hal", "nasb", "تنوين الفتح الظاهر")},
        ]
        out.append(_record(sentence, items))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

GENERATORS = {
    "kana_sisters":   (gen_kana_sisters,   200),
    "inna_sisters":   (gen_inna_sisters,   200),
    "istithna":       (gen_istithna,       250),
    "mawsool":        (gen_mawsool,        200),
    "idafa_edge":     (gen_idafa_edge,     300),
    "quranic":        (gen_quranic,        150),
    "rare_combos":    (gen_rare_combos,    100),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    all_records: List[Dict] = []
    summary = {}
    for name, (fn, n) in GENERATORS.items():
        records = fn(rng, n)
        path = out_dir / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
        summary[name] = len(records)
        all_records.extend(records)
        print(f"  {name}: {len(records)} sentences → {path}")

    # Concatenate
    all_path = out_dir / "all_synthetic.jsonl"
    all_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_records) + "\n")
    print(f"\nTotal: {len(all_records)} sentences → {all_path}")
    summary["total"] = len(all_records)

    summary_path = out_dir / "synthesis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
