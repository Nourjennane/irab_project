"""Phase R2-v2.1 — Gazelle collateral diagnostic.

Locates the specific words where R2-v2.1 changes a Phase 3-A prediction
from CORRECT to WRONG (i.e. collateral damage from KanaReasoner override
firing on spans that overlap with non-kana constructions).

Per-word output:
  - sentence
  - word + position
  - all construction tags (kana_sisters / inna_sisters / idafa / ...)
  - kana span overlap (does this word fall inside a detected kana span?)
  - Phase 3-A pred (case, role, marker, conf) per field
  - R2-v2.1 pred  (case, role, marker, conf) per field
  - Gold (case, role, marker)
  - Per-field correctness: P3A vs v2.1
  - Diagnosis tag: "kana_to_other_collateral" / "kana_intended" / "no_change" / etc.

Output:
  runs/phaseR2_collateral/gazelle_diff.jsonl    — one row per word that flipped
  runs/phaseR2_collateral/summary.md            — collateral counts per construction
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _norm_ar(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[ً-ٰٟ]", "", s)
    s = re.sub(r"[^ء-ي]+", "", s)
    return s


# Reuse construction detectors from eval_per_construction (sentence-level tags)
KANA_SURFACE = {
    "كان", "ليس", "أصبح", "ظل", "صار", "بات", "أمسى", "أضحى",
    "زال", "برح", "فتئ", "انفك",
    "كانت", "ليست", "أصبحت", "صارت", "باتت",
}
INNA_SURFACE = {"إن", "أن", "لكن", "ليت", "لعل", "كأن", "إنّ", "أنّ", "لكنّ", "كأنّ"}
ISTITHNA_SURFACE = {"إلا", "غير", "سوى", "حاشا"}


def detect_sentence_constructions(words: List[str], gold_items: List[Dict]) -> Set[str]:
    norm_words = {_norm_ar(w) for w in words}
    cs: Set[str] = set()
    if norm_words & {_norm_ar(p) for p in KANA_SURFACE}:
        cs.add("kana_sisters")
    if norm_words & {_norm_ar(p) for p in INNA_SURFACE}:
        cs.add("inna_sisters")
    if norm_words & {_norm_ar(p) for p in ISTITHNA_SURFACE}:
        cs.add("istithna")
    # idafa: ≥1 mudaaf_ilayh in gold
    mud_count = sum(1 for g in gold_items if g.get("role") == "mudaaf_ilayh")
    if mud_count >= 1:
        cs.add("idafa")
    if mud_count >= 2:
        cs.add("idafa_multi")
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/phase3a_491240/final")
    ap.add_argument("--memory", default="data/grammar_memory/")
    ap.add_argument("--out_dir", default="runs/phaseR2_collateral")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    import torch.nn.functional as F

    from irab_tashkeel.inference.structured_predictor import (
        StructuredPredictor, StructuredPredictorConfig,
    )
    from irab_tashkeel.data.gazelle import load_gazelle_iraab
    from irab_tashkeel.evaluation.structural import extract, split_sentence_iraab
    from irab_tashkeel.structured.schema import (
        canonicalize_role, ID_TO_CASE, ID_TO_ROLE, ID_TO_MARKER,
    )
    from irab_tashkeel.grammar_memory.memory import GrammarMemory
    from irab_tashkeel.grammar_memory.structural_predictor import StructuralReasoningPredictor
    from irab_tashkeel.grammar_memory.signature import detect_constructions_in_record

    CASE_NORM = {"marfu":"raf", "mansub":"nasb", "majrur":"jarr", "majzum":"jazm",
                 "mabni":"mabni", "raf":"raf", "nasb":"nasb", "jarr":"jarr", "jazm":"jazm"}
    MARKER_NORM = {
        "الضمة الظاهرة":"damma_visible", "الضمة المقدرة":"damma_hidden",
        "الفتحة الظاهرة":"fatha_visible", "الفتحة المقدرة":"fatha_hidden",
        "الكسرة الظاهرة":"kasra_visible", "الكسرة المقدرة":"kasra_hidden",
        "تنوين الضم":"tanween_damm", "تنوين الفتح":"tanween_fath", "تنوين الكسر":"tanween_kasr",
        "السكون":"sukun", "السكون المقدر":"sukun_hidden",
        "الياء":"ya", "الواو":"waw", "الألف":"alif", "النون":"nun", "الفتح":"fath_short",
    }
    def nc(c): return CASE_NORM.get((c or "").strip(), c)
    def nm(m):
        if not m: return m
        m = m.strip()
        if m in MARKER_NORM: return MARKER_NORM[m]
        for k, v in MARKER_NORM.items():
            if k in m: return v
        return m

    print(f"loading model from {args.model}")
    cfg = StructuredPredictorConfig(apply_constraints=False, apply_hierarchical=False,
                                     return_attention=False, render_prose=False, device="auto")
    base_pred = StructuredPredictor(args.model, cfg=cfg)
    memory = GrammarMemory(Path(args.memory))
    v21 = StructuralReasoningPredictor(
        base_predictor=base_pred, memory=memory,
        enabled_families=["kana_sisters"],
    )

    items = load_gazelle_iraab()
    gold_pairs = []
    for it in items:
        pairs = split_sentence_iraab(it.answer)
        if pairs: gold_pairs.append((it.sentence, pairs))
    print(f"loaded {len(gold_pairs)} Gazelle sentences")

    rows: List[Dict] = []
    flips_by_field: Counter = Counter()    # (field, p3a_correct, v21_correct) -> count
    construction_collateral: Dict[str, Counter] = defaultdict(Counter)

    for sent, gpairs in gold_pairs:
        # Gold items
        gold_items = []
        for w, irab in gpairs:
            ext = extract(irab)
            gr = canonicalize_role(ext.role) if (ext and ext.role) else None
            gold_items.append({
                "word": w, "case": nc(ext.case if ext else None),
                "role": gr, "marker": nm(ext.marker if ext else None),
                "irab": irab,
            })
        words = [w for w, _ in gpairs]
        cs = detect_sentence_constructions(words, gold_items)

        # Run Phase 3-A baseline (using predict_sentence)
        p3a_res = base_pred.predict_sentence(sent)
        p3a_by_norm = {_norm_ar(w.word): w for w in p3a_res.items}

        # Run R2-v2.1 wrapper
        v21_res, struct_trace = v21.predict_sentence(sent)
        v21_by_norm = {_norm_ar(w.word): w for w in v21_res.items}

        # Get list of kana spans detected for this sentence
        kana_spans: List[Tuple[int, int]] = []
        for st in struct_trace.span_traces:
            if st.family == "kana_sisters":
                kana_spans.append(st.span)

        # Per word: compare predictions
        for i, gold_item in enumerate(gold_items):
            w = gold_item["word"]
            normed = _norm_ar(w)
            p = p3a_by_norm.get(normed)
            v = v21_by_norm.get(normed)
            if p is None or v is None: continue

            # In any kana span?
            in_kana_span = any(start <= i < end for (start, end) in kana_spans)

            # Per-field correctness
            for field in ("case", "role", "marker"):
                gold_v = gold_item.get(field)
                p_pred = getattr(p, field, None)
                v_pred = getattr(v, field, None)
                if gold_v is None: continue
                p_corr = (p_pred == gold_v)
                v_corr = (v_pred == gold_v)
                if p_corr != v_corr:
                    flips_by_field[(field, p_corr, v_corr)] += 1
                    # Categorize collateral
                    diag = "no_change"
                    if p_corr and not v_corr:
                        diag = "P3A_correct → v21_wrong"
                    elif not p_corr and v_corr:
                        diag = "P3A_wrong   → v21_correct"
                    # Tag affected constructions
                    affected_cs = cs.copy()
                    rows.append({
                        "sentence": sent, "word": w, "position": i,
                        "in_kana_span": in_kana_span,
                        "construction_tags": sorted(affected_cs),
                        "field": field,
                        "gold": gold_v,
                        "p3a": p_pred, "p3a_conf": getattr(p, f"{field}_conf", None),
                        "v21": v_pred, "v21_conf": getattr(v, f"{field}_conf", None),
                        "diagnosis": diag,
                        "kana_spans": kana_spans,
                    })
                    for c in affected_cs:
                        construction_collateral[c][diag] += 1

    out_path = out_dir / "gazelle_diff.jsonl"
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    md = []
    md.append(f"# R2-v2.1 collateral diagnostic — Gazelle\n")
    md.append(f"Total flipped predictions: {len(rows)}\n")
    md.append("## By field × outcome\n")
    md.append("| field | P3A_correct → v21_wrong | P3A_wrong → v21_correct |")
    md.append("|---|---:|---:|")
    for field in ("case", "role", "marker"):
        bad = flips_by_field.get((field, True, False), 0)
        good = flips_by_field.get((field, False, True), 0)
        md.append(f"| {field} | {bad} | {good} |")
    md.append("")
    md.append("## By construction tag\n")
    md.append("| construction | P3A_correct → v21_wrong | P3A_wrong → v21_correct |")
    md.append("|---|---:|---:|")
    for c, ctr in sorted(construction_collateral.items()):
        bad = ctr.get("P3A_correct → v21_wrong", 0)
        good = ctr.get("P3A_wrong   → v21_correct", 0)
        md.append(f"| {c} | {bad} | {good} |")
    md.append("")
    md.append("## All flips (collateral details)\n")
    for r in rows:
        md.append(f"### `{r['word']}` @ pos {r['position']} in: `{r['sentence'][:80]}`")
        md.append(f"- field: **{r['field']}**, diagnosis: **{r['diagnosis']}**")
        md.append(f"- gold: `{r['gold']}`")
        md.append(f"- P3A : `{r['p3a']}` (conf {r['p3a_conf']})")
        md.append(f"- v21 : `{r['v21']}` (conf {r['v21_conf']})")
        md.append(f"- in_kana_span: {r['in_kana_span']}, kana_spans: {r['kana_spans']}")
        md.append(f"- construction tags this word counts toward: {r['construction_tags']}")
        md.append("")

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md))
    print(f"\nWrote {len(rows)} flips to {out_path}")
    print(f"Wrote summary to {md_path}")
    print(f"\n=== Summary ===")
    for field in ("case", "role", "marker"):
        bad = flips_by_field.get((field, True, False), 0)
        good = flips_by_field.get((field, False, True), 0)
        print(f"  {field}: P3A→wrong={bad}, P3A→corrected={good}, net={good-bad}")
    for c, ctr in sorted(construction_collateral.items()):
        bad = ctr.get("P3A_correct → v21_wrong", 0)
        good = ctr.get("P3A_wrong   → v21_correct", 0)
        print(f"  [{c}] bad={bad}, good={good}, net={good-bad}")


if __name__ == "__main__":
    main()
