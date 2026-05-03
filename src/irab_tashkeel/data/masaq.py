"""MASAQ — Morpho-syntactically Annotated Quran (Sawalha et al., 2025).

Source: https://data.mendeley.com/datasets/9yvrzxktmr/
Cite as: Sawalha, M.; Atwell, E.; Brierley, C. (2025). MASAQ: Morpho-Syntactic
Analysis of the Quran. Mendeley Data, V1.

This is a SEGMENT-level annotation: each Arabic word is split at clitic
boundaries (proclitics like ب-, و-, ال-; suffix pronouns like -ها, -ك, -نا)
and each segment receives its own row. The CSV ships 157,676 segment rows
covering the entire Quran (~78k words). The 19 columns:

  ID                    sequential row id
  Sura_No, Verse_No     Quranic surah / verse
  Word_No, Segment_No   word index in verse / segment index in word
  Word                  fully-diacritized surface (whole word, repeated for each segment)
  Without_Diacritics    surface without harakat
  Segmented_Word        the clitic substring this row describes
  Morph_Tag             one of ~72 morph categories: PREP, DET, NOUN_PROP, NOUN_ABSTRACT,
                        VERB_IMPERF, VERB_PERF, VERB_IMPER, REL_PRO, DEM_PRO, PERS_PRO,
                        ADJ, ADV, NUM, CONJ, PART_NEG, PART_INTERROG, PART_VOC, …
  Morph_Type            Prefix / Stem / Suffix
  Punctuation_Mark      empty for most rows; ',' '.' '?' for verse breaks
  Invariable_Declinable INVAR (مبني) / DECLN (معرب) / empty
  Syntactic_Role        functional role: PREP, PREP_OBJ, GEN_CONS (مضاف إليه), SUBJ, OBJ,
                        REL_CL_HEAD, ADJ_OF, … (this is closer to traditional Arabic role
                        labels than UD's deprels are)
  Possessive_Construct  CONSTRUCT (مضاف) / NOT_CONSTRUCT
  Case_Mood             NOMINATIVE / ACCUSATIVE / GENITIVE / SUBJUNCTIVE / JUSSIVE /
                        INDICATIVE / INVARIABLE
  Case_Mood_Marker      DAMMA / FATHA / KASRA / SUKUN / WAW / ALIF / YA / NUN / ETC
  Phrase                phrase boundary marker (PHRASE) at the start of a phrase
  Phrasal_Function      role of the parent phrase: PRED (predicate), TOPIC (مبتدأ),
                        OBJ_OF, GEN_CONS, …
  Gloss                 English gloss

How this differs from QAC (the corpus we currently train on):
  - MASAQ keeps an explicit (Case_Mood, Case_Mood_Marker) pair per segment,
    matching exactly the (case, marker) extraction targets of our structural
    metric. QAC's CASE/MOOD features need additional templating to recover the
    marker phrase.
  - MASAQ's Syntactic_Role is closer to the traditional إعراب role taxonomy
    (مضاف إليه, مفعول به, نعت, …) than UD's nmod/obl/obj/nsubj.
  - MASAQ is segment-level (clitics split out); QAC is word-level. Joining
    MASAQ segments back to whole-word i'rāb requires re-aggregation by
    (Sura_No, Verse_No, Word_No) and merging the per-segment markers (e.g.
    "ب + اسم" → "اسم مجرور بحرف الجر وعلامة جره الكسرة").

Decision: this loader is provided for future training augmentation and as a
reference data resource the paper can characterize. We do NOT integrate
MASAQ into the current training pool because (a) MASAQ is Quranic-only
register, which biases away from the MSA-news distribution our gold sets
(Gazelle, PADT-seed) sample; (b) reliable segment-to-prose templating
requires a careful clitic-merge step we have not yet validated.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_PATH = Path("data/masaq/MASAQ.csv")


def load_masaq_segments(path: Path | str = DEFAULT_PATH) -> List[Dict[str, str]]:
    """Return all MASAQ rows as a list of dicts (one per segment)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"MASAQ CSV not found at {p}. "
            "Download from https://data.mendeley.com/datasets/9yvrzxktmr/")
    rows: List[Dict[str, str]] = []
    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_masaq_examples(
    path: Path | str = DEFAULT_PATH,
    max_words: Optional[int] = None,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Group MASAQ rows by (Sura_No, Verse_No, Word_No) and return per-word
    tuples (word, sentence_context, irab_features).

    `irab_features` carries the SEGMENT-level annotations as a list under
    `segments`, plus the aggregated word-level Morph_Tag / Case_Mood /
    Case_Mood_Marker (when unambiguous across the word's stem segment).

    `sentence_context` is the verse joined by spaces. It is approximate
    (segment surfaces re-joined), not the canonical mushaf orthography.
    """
    rows = load_masaq_segments(path)
    by_word: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    by_verse: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    for r in rows:
        key = (r["Sura_No"], r["Verse_No"], r["Word_No"])
        by_word[key].append(r)
        # collect verse-level word surface (first occurrence per Word_No)
        verse_key = (r["Sura_No"], r["Verse_No"])
        by_verse[verse_key].setdefault(r["Word_No"], r["Word"])

    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for (sura, verse, word_no), segs in by_word.items():
        word_surface = segs[0]["Word"]
        verse_words = by_verse[(sura, verse)]
        sentence = " ".join(verse_words[w] for w in sorted(verse_words, key=int))
        # Pick the stem segment for primary case/mood (Morph_Type=='Stem' if present).
        stem = next((s for s in segs if s.get("Morph_Type") == "Stem"), segs[0])
        feats: Dict[str, Any] = {
            "morph_tag": stem.get("Morph_Tag", ""),
            "syntactic_role": stem.get("Syntactic_Role", ""),
            "case_mood": stem.get("Case_Mood", ""),
            "case_mood_marker": stem.get("Case_Mood_Marker", ""),
            "invariable_declinable": stem.get("Invariable_Declinable", ""),
            "possessive_construct": stem.get("Possessive_Construct", ""),
            "phrasal_function": stem.get("Phrasal_Function", ""),
            "segments": [
                {
                    "morph_tag": s.get("Morph_Tag", ""),
                    "morph_type": s.get("Morph_Type", ""),
                    "segmented": s.get("Segmented_Word", ""),
                    "syntactic_role": s.get("Syntactic_Role", ""),
                    "case_mood": s.get("Case_Mood", ""),
                    "case_mood_marker": s.get("Case_Mood_Marker", ""),
                }
                for s in segs
            ],
            "sura_verse_word": f"{sura}:{verse}:{word_no}",
            "gloss": stem.get("Gloss", ""),
        }
        out.append((word_surface, sentence, feats))
        if max_words and len(out) >= max_words:
            break
    return out


def write_sample(
    out_path: Path | str = "data/masaq_sample.jsonl",
    n: int = 100,
    src: Path | str = DEFAULT_PATH,
) -> Path:
    """Write the first `n` words to a JSONL for inspection."""
    examples = load_masaq_examples(src, max_words=n)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for word, sentence, feats in examples:
            f.write(json.dumps(
                {"word": word, "sentence": sentence, "features": feats},
                ensure_ascii=False,
            ) + "\n")
    return out


def schema_summary(rows: Optional[Iterable[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Return a tag-frequency summary suitable for the writeup's data section."""
    from collections import Counter
    if rows is None:
        rows = load_masaq_segments()
    morph = Counter()
    role = Counter()
    case_mood = Counter()
    marker = Counter()
    for r in rows:
        if r.get("Morph_Tag"): morph[r["Morph_Tag"]] += 1
        if r.get("Syntactic_Role"): role[r["Syntactic_Role"]] += 1
        if r.get("Case_Mood"): case_mood[r["Case_Mood"]] += 1
        if r.get("Case_Mood_Marker"): marker[r["Case_Mood_Marker"]] += 1
    return {
        "n_segments": sum(morph.values()) + (sum(1 for r in rows if not r.get("Morph_Tag")) if rows else 0),
        "morph_tag_n_unique": len(morph),
        "morph_tag_top10": morph.most_common(10),
        "syntactic_role_n_unique": len(role),
        "syntactic_role_top10": role.most_common(10),
        "case_mood_distribution": dict(case_mood.most_common()),
        "marker_distribution": dict(marker.most_common()),
    }


# ---------------------------------------------------------------------------
# Templating: MASAQ row → traditional Arabic i'rāb prose
# ---------------------------------------------------------------------------
_CASE_MOOD_TO_AR = {
    "NOMINATIVE": "مرفوع",
    "ACCUSATIVE": "منصوب",
    "GENITIVE":   "مجرور",
    "JUSSIVE":    "مجزوم",
    "SUBJUNCTIVE":"منصوب",
    "INDICATIVE": "مرفوع",
}
_CASE_GEN_FORM = {
    "NOMINATIVE": "رفعه",
    "ACCUSATIVE": "نصبه",
    "GENITIVE":   "جره",
    "JUSSIVE":    "جزمه",
    "SUBJUNCTIVE":"نصبه",
    "INDICATIVE": "رفعه",
}
_MARKER_TO_AR = {
    "DAMMA":  "الضمة الظاهرة",
    "FATHA":  "الفتحة الظاهرة",
    "KASRA":  "الكسرة الظاهرة",
    "SUKUN":  "السكون",
    "WAW":    "الواو",
    "ALIF":   "الألف",
    "YA":     "الياء",
    "NUN":    "النون",
}
_ROLE_TO_AR = {
    "SUBJ":         "فاعل",
    "OBJ":          "مفعول به",
    "GEN_CONS":     "مضاف إليه",
    "PREP_OBJ":     "اسم مجرور بحرف الجر",
    "PRED":         "خبر",
    "TOPIC":        "مبتدأ",
    "ADJ_OF":       "نعت",
    "REL_CL_HEAD":  "اسم موصول",
    "PREP":         "حرف جر",
    "PUNCT":        "علامة ترقيم",
}
_POS_TO_AR_NOUN = {"NOUN_PROP", "NOUN_ABSTRACT", "NOUN_CONCRETE", "NOUN_COL"}
_POS_TO_AR_VERB = {"PV", "IV", "CV", "VERB_PERF", "VERB_IMPERF", "VERB_IMPER"}
_POS_INVAR_HARFAR = {"PREP", "CONJ", "DET", "PART_NEG", "PART_INTERROG", "PART_VOC", "PART"}


def render_word_irab(stem_features: Dict[str, str]) -> str:
    """Render a MASAQ stem-segment dict to a single Arabic i'rāb prose line.

    Returns "" when the templating cannot produce a clean prose form (which
    happens for some clitic-stripped fragments, complex pronouns, etc.).
    """
    morph = stem_features.get("morph_tag", "")
    case_mood = stem_features.get("case_mood", "")
    marker = stem_features.get("case_mood_marker", "")
    role = stem_features.get("syntactic_role", "")
    invar = stem_features.get("invariable_declinable", "")
    constr = stem_features.get("possessive_construct", "")

    role_ar = _ROLE_TO_AR.get(role, "")
    case_ar = _CASE_MOOD_TO_AR.get(case_mood, "")
    case_gen = _CASE_GEN_FORM.get(case_mood, "")
    marker_ar = _MARKER_TO_AR.get(marker, "")

    if morph in _POS_TO_AR_NOUN and case_ar and marker_ar:
        head = role_ar or "اسم"
        # Avoid repeating the case word when the role already contains it
        # (e.g. role="اسم مجرور بحرف الجر" already has "مجرور").
        case_part = "" if case_ar in head else f" {case_ar}"
        out = f"{head}{case_part} وعلامة {case_gen} {marker_ar}"
        if constr == "CONSTRUCT":
            out += " وهو مضاف"
        return out
    if morph in _POS_TO_AR_VERB:
        if case_mood in ("INDICATIVE", "NOMINATIVE"):
            return "فعل مضارع مرفوع وعلامة رفعه الضمة الظاهرة"
        if case_mood in ("SUBJUNCTIVE", "ACCUSATIVE"):
            return "فعل مضارع منصوب وعلامة نصبه الفتحة الظاهرة"
        if case_mood == "JUSSIVE":
            return "فعل مضارع مجزوم وعلامة جزمه السكون"
        if morph == "PV" or morph == "VERB_PERF":
            return "فعل ماض مبني على الفتح"
        if morph == "CV" or morph == "VERB_IMPER":
            return "فعل أمر مبني على السكون"
        return "فعل"
    if morph in _POS_INVAR_HARFAR or invar == "INVAR":
        if morph == "PREP":
            return "حرف جر مبني على الكسر"
        if morph == "CONJ":
            return "حرف عطف مبني"
        if morph == "DET":
            return "أداة تعريف"
        return "حرف مبني"
    if morph in {"REL_PRO", "DEM_PRO", "PERS_PRO", "POSS_PRON", "SUBJ_PRON", "OBJ_PRON"}:
        if case_mood == "NOMINATIVE":
            return "ضمير مبني في محل رفع"
        if case_mood == "ACCUSATIVE":
            return "ضمير مبني في محل نصب"
        if case_mood == "GENITIVE":
            return "ضمير مبني في محل جر"
        return "ضمير مبني"
    return ""


def to_fewshots(max_verses: int = 1500, src: Path | str = DEFAULT_PATH) -> List["object"]:
    """Build FewShotExample objects from MASAQ for the RAG retrieval pool.

    Groups segments by (sura, verse), renders each word's stem features into
    Arabic i'rāb prose, and packages as `{sentence, irab_lines}`. Skips verses
    where fewer than 2 words template cleanly (insufficient signal for RAG).

    Returns the same FewShotExample type used by `llm_baselines`. Capped at
    `max_verses` to keep the augmented pool from dominating the retrieval
    distribution.
    """
    from ..inference.llm_baselines import FewShotExample

    rows = load_masaq_segments(src)
    by_word: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    word_surface: Dict[Tuple[str, str, str], str] = {}
    verse_words: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)

    for r in rows:
        wkey = (r["Sura_No"], r["Verse_No"], r["Word_No"])
        by_word[wkey].append(r)
        word_surface.setdefault(wkey, r.get("Without_Diacritics") or r.get("Word", ""))
        verse_words[(r["Sura_No"], r["Verse_No"])][r["Word_No"]] = word_surface[wkey]

    out: List[FewShotExample] = []
    for verse_key, words_dict in verse_words.items():
        if len(out) >= max_verses:
            break
        sentence = " ".join(words_dict[w] for w in sorted(words_dict, key=int))
        if len(sentence.split()) < 3:
            continue
        block_lines: List[str] = []
        for word_no in sorted(words_dict, key=int):
            wkey = (verse_key[0], verse_key[1], word_no)
            segs = by_word.get(wkey) or []
            stem = next((s for s in segs if s.get("Morph_Type") == "Stem"), segs[0] if segs else {})
            feats = {
                "morph_tag": stem.get("Morph_Tag", ""),
                "case_mood": stem.get("Case_Mood", ""),
                "case_mood_marker": stem.get("Case_Mood_Marker", ""),
                "syntactic_role": stem.get("Syntactic_Role", ""),
                "invariable_declinable": stem.get("Invariable_Declinable", ""),
                "possessive_construct": stem.get("Possessive_Construct", ""),
            }
            irab = render_word_irab(feats)
            if irab:
                block_lines.append(f"{words_dict[word_no]}: {irab}")
        if len(block_lines) < 2:
            continue
        out.append(FewShotExample(sentence=sentence, irab_lines="\n".join(block_lines)))
    return out


# ---------------------------------------------------------------------------
# MASAQ → eval set (Gazelle-compatible format)
# ---------------------------------------------------------------------------
def write_eval_jsonl(out_path: Path | str = "data/masaq_eval.jsonl",
                     target_words: int = 5000,
                     min_words_per_sent: int = 4,
                     max_words_per_sent: int = 25,
                     seed: int = 42,
                     src: Path | str = DEFAULT_PATH) -> Path:
    """Build a held-out eval set in the same JSONL shape as our Gazelle pipeline:
        {"sentence": str, "items": [{"word": str, "irab": str}, ...], ...}

    Filters:
      - skip verses whose word count is outside [min_words_per_sent, max_words_per_sent]
      - skip verses where ANY word fails to template (we want clean per-verse gold)
      - sample randomly until target_words per-word rows are accumulated
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = load_masaq_segments(src)
    by_word: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    word_surface: Dict[Tuple[str, str, str], str] = {}
    verse_words: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    for r in rows:
        wkey = (r["Sura_No"], r["Verse_No"], r["Word_No"])
        by_word[wkey].append(r)
        word_surface.setdefault(wkey, r.get("Without_Diacritics") or r.get("Word", ""))
        verse_words[(r["Sura_No"], r["Verse_No"])][r["Word_No"]] = word_surface[wkey]

    rng = random.Random(seed)
    verse_keys = list(verse_words.keys())
    rng.shuffle(verse_keys)

    n_total_words = 0
    n_verses = 0
    n_skipped = 0
    n_skipped_too_short = 0
    n_skipped_too_long = 0
    n_skipped_template_fail = 0

    with open(out, "w", encoding="utf-8") as f:
        for verse_key in verse_keys:
            if n_total_words >= target_words:
                break
            words_dict = verse_words[verse_key]
            n_words = len(words_dict)
            if n_words < min_words_per_sent:
                n_skipped_too_short += 1
                continue
            if n_words > max_words_per_sent:
                n_skipped_too_long += 1
                continue

            sentence = " ".join(words_dict[w] for w in sorted(words_dict, key=int))
            items = []
            for word_no in sorted(words_dict, key=int):
                wkey = (verse_key[0], verse_key[1], word_no)
                segs = by_word.get(wkey) or []
                stem = next((s for s in segs if s.get("Morph_Type") == "Stem"), segs[0] if segs else {})
                feats = {
                    "morph_tag": stem.get("Morph_Tag", ""),
                    "case_mood": stem.get("Case_Mood", ""),
                    "case_mood_marker": stem.get("Case_Mood_Marker", ""),
                    "syntactic_role": stem.get("Syntactic_Role", ""),
                    "invariable_declinable": stem.get("Invariable_Declinable", ""),
                    "possessive_construct": stem.get("Possessive_Construct", ""),
                }
                irab = render_word_irab(feats)
                if not irab:
                    continue   # skip THIS word; keep the rest of the verse
                items.append({"word": words_dict[word_no], "irab": irab})
            # Need ≥3 templatable words per verse to be a useful eval point
            if len(items) < 3:
                n_skipped_template_fail += 1
                continue

            row = {
                "sentence": sentence,
                "items": items,
                "source": "masaq",
                "sura_verse": f"{verse_key[0]}:{verse_key[1]}",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_total_words += len(items)
            n_verses += 1

    print(f"\n=== MASAQ eval written ===")
    print(f"  out: {out}")
    print(f"  verses kept: {n_verses}")
    print(f"  word-level judgments: {n_total_words}")
    print(f"  skipped — too short: {n_skipped_too_short}")
    print(f"  skipped — too long:  {n_skipped_too_long}")
    print(f"  skipped — template fail (≥1 word didn't render): {n_skipped_template_fail}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MASAQ loader / sampler")
    p.add_argument("--sample", type=int, default=100,
                   help="write N word-level rows to data/masaq_sample.jsonl")
    p.add_argument("--summary", action="store_true",
                   help="print schema summary (morph/role/case/marker tag distributions)")
    p.add_argument("--build_eval", type=int, default=0, metavar="N_WORDS",
                   help="write a Gazelle-compatible eval JSONL with ~N word judgments to data/masaq_eval.jsonl")
    args = p.parse_args()

    if args.build_eval > 0:
        write_eval_jsonl(target_words=args.build_eval)
        import sys; sys.exit(0)

    if args.summary:
        rows = load_masaq_segments()
        s = schema_summary(rows)
        print(json.dumps(s, ensure_ascii=False, indent=2))

    out = write_sample(n=args.sample)
    print(f"wrote {args.sample} word-level rows to {out}")
