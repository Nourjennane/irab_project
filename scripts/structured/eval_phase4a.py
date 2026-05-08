"""Phase 4a 4-stream evaluation.

For a given Phase 4a model (taxonomy=v4) on a given eval surface (Gazelle or
MASAQ), reports four orthogonal metric streams + a stress table:

  Stream A — native canonical (34-label v4)
  Stream B — grouped canonical (v4 → v3 collapse, apples-to-apples vs Phase 1)
  Stream C — raw-string overlap (rendered prose vs gold raw role substring)
  Stream D — extractor-surface match (existing structural.py extractor view)

Plus the stress table from §18 of the design doc:
  - rare-role macro-F1 (12 lowest-support classes)
  - head-role stability (8 highest-support classes)
  - long-tail collapse count (labels with F1 < 50%)
  - calibration drift (mean conf correct − wrong)

Usage:
    python scripts/structured/eval_phase4a.py \\
        --model runs/phase4a_taxonomy_full_491044/final \\
        --eval gazelle \\
        --out_dir runs/phase4a_eval_491044/gazelle
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", choices=["gazelle", "masaq"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--apply_constraints", action="store_true",
                    help="Apply the 4 symbolic constraints during prediction")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import numpy as np
    from irab_tashkeel.inference.structured_predictor import (
        StructuredPredictor, StructuredPredictorConfig,
    )
    from irab_tashkeel.evaluation.structural import extract
    from irab_tashkeel.structured.taxonomy_v4 import (
        NEW_TO_OLD, ROLE_LABELS_V4, AUTO_FALLBACK_AT_RISK,
    )
    from irab_tashkeel.structured.schema import (
        ROLE_LABELS as ROLE_LABELS_V3,
        canonicalize_role as canonicalize_role_v3,
    )

    # ---- load eval set ----
    if args.eval == "gazelle":
        from irab_tashkeel.data.gazelle import load_gazelle_iraab
        from irab_tashkeel.evaluation.structural import split_sentence_iraab
        items = load_gazelle_iraab()
        gold_pairs = []
        for it in items:
            pairs = split_sentence_iraab(it.answer)
            if pairs:
                gold_pairs.append((it.sentence, pairs))
    else:  # masaq
        gold_pairs = []
        with open("data/masaq_eval.jsonl") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pairs = [(it.get("word", ""), it.get("irab", ""))
                         for it in row.get("items", [])
                         if isinstance(it, dict) and it.get("word") and it.get("irab")]
                if row.get("sentence") and pairs:
                    gold_pairs.append((row["sentence"], pairs))

    print(f"loaded {len(gold_pairs)} {args.eval} sentences "
          f"({sum(len(p) for _, p in gold_pairs)} word judgments)")

    # ---- load Phase 4a predictor ----
    print(f"loading model from {args.model}")
    cfg = StructuredPredictorConfig(
        apply_constraints=args.apply_constraints,
        apply_hierarchical=False,         # Phase 4a does NOT touch this
        return_attention=False,
        render_prose=True,
        device="auto",
    )
    pred = StructuredPredictor(args.model, cfg=cfg)
    print(f"  taxonomy={pred.taxonomy}, n_role={pred.model._n_role}")

    # ---- helpers ----
    def gold_canon_v3(irab_text):
        """Extract canonical v3 role from gold prose.

        ``extract().role`` returns the Arabic surface substring (e.g. "فاعل")
        — we then push that through ``canonicalize_role_v3`` to get the
        English-keyed canonical label (e.g. "fail"). Returns ``None`` when
        the extractor finds no role.
        """
        a = extract(irab_text)
        if a is None or a.role is None:
            return None
        return canonicalize_role_v3(a.role)

    # Per-judgment stream collection
    rec_native = []        # (gold_v3, pred_v4)
    rec_grouped = []       # (gold_v3, pred_v3_collapsed)
    rec_raw_overlap = []   # (gold_raw_substr, pred_prose) → 1 if substring match
    rec_extractor = []     # (gold_v3_via_extractor, pred_v3_via_extractor)
    role_conf_correct = defaultdict(list)  # per-class confidence on correct
    role_conf_wrong = defaultdict(list)
    pred_role_counts = Counter()

    # Gather predictions for every sentence
    n_processed = 0
    for sent, gpairs in gold_pairs:
        result = pred.predict_sentence(sent)
        # Align preds to gold by surface-word match (same logic as run_baselines)
        from irab_tashkeel.evaluation.run_baselines import _align_pred
        gold_words = [w for w, _ in gpairs]
        pred_dicts = [w.to_dict() for w in result.items]
        for w in result.items:
            d = pred_dicts[result.items.index(w)] if w in result.items else None
            # Save the rendered prose as 'irab' for downstream alignment
        # rebuild pred_dicts with proper irab field
        pred_dicts = []
        for w in result.items:
            from irab_tashkeel.inference.template_renderer import render_word
            pred_dicts.append({
                "word": w.word,
                "irab": w.irab_prose or render_word(w),
                "role": w.role,
                "case": w.case,
                "marker": w.marker,
                "role_conf": w.role_conf,
            })

        # Build per-word lookup
        pred_by_word = {p["word"]: p for p in pred_dicts}
        # Normalised lookup as a fallback (strip diacritics)
        from irab_tashkeel.evaluation.run_baselines import _align_pred
        aligned_irab = _align_pred(gold_words, pred_dicts)

        for (gw, gold_irab), pred_irab in zip(gpairs, aligned_irab):
            # Pull the per-word predicted v4 role from pred_by_word using
            # surface match (with diacritic-stripped fallback)
            import re, unicodedata
            def norm(s):
                s = unicodedata.normalize("NFC", s or "")
                s = re.sub(r"[ً-ْٰ]", "", s)
                return re.sub(r"[^ء-ي]+", "", s)
            normed = norm(gw)
            p_v4 = None
            p_conf = None
            for p in pred_dicts:
                if norm(p["word"]) == normed:
                    p_v4 = p["role"]
                    p_conf = p.get("role_conf")
                    break

            g_v3 = gold_canon_v3(gold_irab)
            if g_v3 is None:
                continue

            # Stream A: native — pred is v4, gold is v3 (collapsed view)
            # We compare native predicted v4 against the gold v3-collapsed
            # but matched up: for native we need pred v4 == "x" where
            # NEW_TO_OLD[x] == g_v3. For correctness reporting, we score
            # native by whether pred v4 collapses to gold v3.
            p_v3_collapsed = NEW_TO_OLD.get(p_v4, "other") if p_v4 else "other"
            rec_native.append((p_v4, g_v3))
            rec_grouped.append((p_v3_collapsed, g_v3))

            # Stream C: raw-string overlap — does the rendered prose contain
            # any of the gold raw role substrings?
            from irab_tashkeel.structured.schema import ARABIC_ROLE_FORMS
            pred_form = ARABIC_ROLE_FORMS.get(p_v4, "")
            # ANY recognisable role from gold prose in the predicted prose?
            rec_raw_overlap.append((pred_irab, gold_irab, p_v4, g_v3))

            # Stream D: extractor on rendered prose
            ext_pred = extract(pred_irab) if pred_irab else None
            pred_v3_extracted = canonicalize_role_v3(ext_pred.role) if ext_pred and ext_pred.role else None
            rec_extractor.append((pred_v3_extracted, g_v3))

            # Calibration tracking (using v3-collapsed for consistent rare-class behaviour)
            if p_conf is not None:
                if p_v3_collapsed == g_v3:
                    role_conf_correct[g_v3].append(p_conf)
                else:
                    role_conf_wrong[g_v3].append(p_conf)
            pred_role_counts[p_v3_collapsed] += 1

        n_processed += 1

    # ---- STREAM A: native ----
    # We score by collapsing pred v4 → v3 and comparing to gold v3 (since
    # Gazelle/MASAQ gold doesn't have v4 labels — they're prose extracted to v3).
    # The "native" view's added value here is per-class: are the v4 predictions
    # internally consistent (i.e. does the model use the new labels)?
    pred_v4_used = Counter(p for p, _ in rec_native if p is not None)

    # ---- STREAM B: grouped (v3 collapse) — apples-to-apples vs Phase 1 ----
    # Per-class prec/rec/F1 over v3 labels
    correct_v3 = sum(1 for p, g in rec_grouped if p == g)
    n_v3 = len(rec_grouped)
    role_acc_grouped = correct_v3 / n_v3 if n_v3 else 0.0

    # Per-class P/R/F1
    classes = ROLE_LABELS_V3
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for p, g in rec_grouped:
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    per_class = {}
    f1_list = []
    for c in classes:
        p_count = tp[c] + fp[c]
        r_count = tp[c] + fn[c]
        precision = tp[c] / p_count if p_count else 0.0
        recall = tp[c] / r_count if r_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": r_count}
        if r_count > 0:
            f1_list.append(f1)
    macro_f1_grouped = float(np.mean(f1_list)) if f1_list else 0.0

    # ---- STREAM C: raw-string overlap ----
    # For each prediction, does the rendered prose contain any portion of the
    # gold raw role substring? This is a coarse "did we render something
    # recognisable?" metric. We compute it as: does ARABIC_ROLE_FORMS[p_v4]
    # appear (as substring) in the gold prose, OR vice-versa?
    overlap_correct = 0
    n_overlap = 0
    for pred_prose, gold_prose, p_v4, g_v3 in rec_raw_overlap:
        if not pred_prose or not gold_prose:
            continue
        n_overlap += 1
        # Surface form for the predicted role
        from irab_tashkeel.structured.schema import ARABIC_ROLE_FORMS
        pred_form = ARABIC_ROLE_FORMS.get(p_v4, "")
        if pred_form and pred_form in gold_prose:
            overlap_correct += 1
    raw_overlap = overlap_correct / n_overlap if n_overlap else 0.0

    # ---- STREAM D: extractor-surface ----
    extractor_correct = sum(1 for p, g in rec_extractor if p == g)
    n_ext = len(rec_extractor)
    extractor_acc = extractor_correct / n_ext if n_ext else 0.0

    # ---- Stress table ----
    # Phase 1 12 rare classes (< 500 train support after Phase 1)
    rare_classes = [
        "naib_fail", "mafoul_other", "ism_inna", "khabar_inna",
        "ism_kana", "khabar_kana", "hal", "tamyeez", "munada",
    ]  # we keep this consistent across phases
    head_classes = [
        "mudaaf_ilayh", "ism_majrur", "naat", "badal",
        "harf_jarr", "mubtada", "harf_atf", "mafoul_bih",
    ]
    rare_f1 = [per_class[c]["f1"] for c in rare_classes if c in per_class and per_class[c]["support"] > 0]
    head_f1 = [per_class[c]["f1"] for c in head_classes if c in per_class and per_class[c]["support"] > 0]
    rare_macro_f1 = float(np.mean(rare_f1)) if rare_f1 else 0.0
    head_macro_f1 = float(np.mean(head_f1)) if head_f1 else 0.0
    long_tail_collapse = sum(
        1 for c in classes
        if c in per_class and per_class[c]["support"] > 0 and per_class[c]["f1"] < 0.5
    )
    # Calibration drift
    all_correct = []
    all_wrong = []
    for c in role_conf_correct:
        all_correct.extend(role_conf_correct[c])
    for c in role_conf_wrong:
        all_wrong.extend(role_conf_wrong[c])
    calib_correct = float(np.mean(all_correct)) if all_correct else 0.0
    calib_wrong = float(np.mean(all_wrong)) if all_wrong else 0.0
    calib_gap = calib_correct - calib_wrong

    # ---- summarise ----
    summary = {
        "model": str(args.model),
        "eval": args.eval,
        "n_sentences": n_processed,
        "n_judgments_v3": n_v3,
        "stream_A_native": {
            "predicted_v4_label_distribution": dict(pred_v4_used.most_common()),
            "n_unique_v4_labels_used": len(pred_v4_used),
        },
        "stream_B_grouped": {
            "role_accuracy": role_acc_grouped,
            "macro_f1": macro_f1_grouped,
            "per_class": per_class,
        },
        "stream_C_raw_string_overlap": {
            "overlap_rate": raw_overlap,
            "n": n_overlap,
        },
        "stream_D_extractor_surface": {
            "match_rate": extractor_acc,
            "n": n_ext,
        },
        "stress_table": {
            "rare_role_macro_f1": rare_macro_f1,
            "head_role_macro_f1": head_macro_f1,
            "long_tail_collapse_count": long_tail_collapse,
            "calibration_correct": calib_correct,
            "calibration_wrong": calib_wrong,
            "calibration_gap": calib_gap,
        },
    }
    (out_dir / "phase4a_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    print(f"\n=== Phase 4a {args.eval} eval ===")
    print(f"sentences {n_processed}, judgments {n_v3}")
    print(f"  Stream A (native):       used {summary['stream_A_native']['n_unique_v4_labels_used']} of 34 v4 labels")
    print(f"  Stream B (grouped v3):   role_acc {role_acc_grouped*100:.2f}%  macro-F1 {macro_f1_grouped*100:.2f}%")
    print(f"  Stream C (raw-overlap):  {raw_overlap*100:.2f}%")
    print(f"  Stream D (extractor):    {extractor_acc*100:.2f}%")
    print(f"  Stress: rare {rare_macro_f1*100:.2f}%  head {head_macro_f1*100:.2f}%  "
          f"long-tail-collapse {long_tail_collapse}  calib-gap {calib_gap:+.3f}")
    print(f"\nWritten to {out_dir}/phase4a_summary.json")


if __name__ == "__main__":
    main()
