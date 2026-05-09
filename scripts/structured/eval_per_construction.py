"""Per-construction evaluation for iʿrāb prediction.

Reports per-construction metrics on Gazelle + MASAQ for a given model
checkpoint, broken down by:

- **kāna sisters** — sentences containing one of the 12 kāna-family
  particles (كان، ليس، أصبح، ظل، صار، بات، أمسى، أضحى، ما زال، ما برح،
  ما فتئ، ما انفك)
- **inna sisters** — sentences containing one of {إن، أن، لكن، ليت،
  لعل، كأن}
- **istithnāʾ** — sentences containing one of {إلا، غير، سوى، ما عدا،
  ما خلا، حاشا}
- **mawṣūl** — sentences containing one of {الذي، التي، الذين، اللاتي،
  اللواتي، اللذان، اللتان، من، ما} in a relative-clause position
- **iḍāfa** — gold tokens with role == 'mudaaf_ilayh'
- **multi-level iḍāfa** — gold sentences with ≥2 consecutive 'mudaaf_ilayh' tokens
- **Quranic patterns** — sentences containing {قد، إذ، إذا، لمّا، كلّما،
  حتى، يا أيها} (proxy for Quranic-style structures)

For each construction subset and overall, report:

- per-construction word-level case_accuracy / role_f1 / marker_em / fully
- per-construction confusion matrix on the role labels involved
- calibration shift: mean confidence on construction tokens vs
  non-construction tokens (signal that the model treats them
  differently)
- distribution: how many sentences contain the construction in each
  test set

Output format: a single JSON file per (model, eval_set), plus a
human-readable Markdown summary.

Usage:
    python scripts/structured/eval_per_construction.py \\
        --model runs/phase3a_491240/final \\
        --eval gazelle \\
        --out_dir runs/per_construction_phase3a/gazelle/

    python scripts/structured/eval_per_construction.py \\
        --model runs/phase3a_491240/final \\
        --eval masaq \\
        --out_dir runs/per_construction_phase3a/masaq/
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
from typing import Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Construction detectors (operate on raw Arabic surface forms with diacritics)
# ---------------------------------------------------------------------------

def _norm_ar(s: str) -> str:
    """Strip diacritics + non-Arabic for surface matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[ً-ٰٟ]", "", s)
    s = re.sub(r"[^ء-ي]+", "", s)
    return s


KANA_SURFACE_FORMS: Set[str] = {
    "كان", "ليس", "أصبح", "ظل", "صار", "بات",
    "أمسى", "أضحى", "زال", "برح", "فتئ", "انفك",
    "كانت", "ليست", "أصبحت", "صارت", "باتت",
}
INNA_SURFACE_FORMS: Set[str] = {"إن", "أن", "لكن", "ليت", "لعل", "كأن", "إنّ", "أنّ", "لكنّ", "كأنّ"}
ISTITHNA_SURFACE_FORMS: Set[str] = {"إلا", "غير", "سوى", "حاشا"}
ISTITHNA_PHRASES: Set[str] = {"ما عدا", "ما خلا"}
MAWSOOL_SURFACE_FORMS: Set[str] = {
    "الذي", "التي", "الذين", "اللاتي", "اللواتي",
    "اللذان", "اللتان", "اللتين", "اللذين",
}
QURANIC_PROXY_FORMS: Set[str] = {"قد", "إذ", "إذا", "لما", "لمّا", "كلما", "كلّما", "حتى"}


def detect_constructions(words: List[str], gold_items: List[Dict]) -> Set[str]:
    """Return the set of construction labels present in this sentence."""
    out: Set[str] = set()

    # Surface-form detection
    norm_words = [_norm_ar(w) for w in words]
    for nw in norm_words:
        if nw in {_norm_ar(x) for x in KANA_SURFACE_FORMS}:
            out.add("kana_sisters")
        if nw in {_norm_ar(x) for x in INNA_SURFACE_FORMS}:
            out.add("inna_sisters")
        if nw in {_norm_ar(x) for x in ISTITHNA_SURFACE_FORMS}:
            out.add("istithna")
        if nw in {_norm_ar(x) for x in MAWSOOL_SURFACE_FORMS}:
            out.add("mawsool")
        if nw in {_norm_ar(x) for x in QURANIC_PROXY_FORMS}:
            out.add("quranic_proxy")

    # Phrase-level detection (for "ما عدا" / "ما خلا")
    full_norm = " ".join(norm_words)
    for phrase in ISTITHNA_PHRASES:
        if _norm_ar(phrase.split()[0]) in norm_words and _norm_ar(phrase.split()[1]) in norm_words:
            out.add("istithna")

    # Role-based detection: iḍāfa
    mudaaf_count = sum(1 for it in gold_items if it.get("role") == "mudaaf_ilayh")
    if mudaaf_count >= 1:
        out.add("idafa")
    # Multi-level iḍāfa: ≥2 consecutive mudaaf_ilayh tokens
    consecutive = 0
    max_consecutive = 0
    for it in gold_items:
        if it.get("role") == "mudaaf_ilayh":
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    if max_consecutive >= 2:
        out.add("idafa_multi")

    # ism_kana / khabar_kana / ism_inna / khabar_inna roles directly
    for it in gold_items:
        r = it.get("role", "")
        if r in {"ism_kana", "khabar_kana"}:
            out.add("kana_sisters")
        if r in {"ism_inna", "khabar_inna"}:
            out.add("inna_sisters")

    return out


CONSTRUCTION_LABELS = [
    "kana_sisters",
    "inna_sisters",
    "istithna",
    "mawsool",
    "idafa",
    "idafa_multi",
    "quranic_proxy",
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to trained model dir (e.g. runs/phase3a_491240/final)")
    ap.add_argument("--eval", choices=["gazelle", "masaq"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--use_retrieval", action="store_true",
                    help="apply Phase R retrieval bias on logits at inference")
    ap.add_argument("--retrieval_memory", default="data/grammar_memory/",
                    help="root dir of grammar memory (per-family JSONL + FAISS)")
    ap.add_argument("--retrieval_lambda", type=float, default=0.3,
                    help="bias multiplier on log(prior); 0 reproduces baseline")
    ap.add_argument("--retrieval_k", type=int, default=5)
    # Phase R2 — structural reasoning (mutually exclusive with --use_retrieval)
    ap.add_argument("--use_structural_reasoning", action="store_true",
                    help="Phase R2: apply per-construction structural reasoners with confidence gating")
    ap.add_argument("--tau_high", type=float, default=0.75,
                    help="confidence threshold for symbolic override")
    ap.add_argument("--tau_med", type=float, default=0.50,
                    help="confidence threshold for strong-bias mode")
    ap.add_argument("--lambda_strong", type=float, default=1.5,
                    help="multiplier for strong-bias log-prior addition (when tau_med <= conf < tau_high)")
    ap.add_argument("--enabled_families", default=None,
                    help="comma-separated whitelist of construction families (default: all reasoners)")
    ap.add_argument("--dump_traces", action="store_true",
                    help="dump per-sentence reasoning traces to traces.jsonl")
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
    from irab_tashkeel.evaluation.structural import extract, split_sentence_iraab
    from irab_tashkeel.structured.schema import (
        ROLE_LABELS, canonicalize_role,
    )

    # The extractor returns case as transliterations and marker as Arabic
    # surface — normalize to the canonical English labels the predictor uses.
    CASE_NORM = {
        "marfu": "raf",
        "mansub": "nasb",
        "majrur": "jarr",
        "majzum": "jazm",
        "mabni": "mabni",
        "raf": "raf", "nasb": "nasb", "jarr": "jarr", "jazm": "jazm",
    }
    MARKER_NORM = {
        "الضمة الظاهرة": "damma_visible",
        "الضمة المقدرة": "damma_hidden",
        "الفتحة الظاهرة": "fatha_visible",
        "الفتحة المقدرة": "fatha_hidden",
        "الكسرة الظاهرة": "kasra_visible",
        "الكسرة المقدرة": "kasra_hidden",
        "تنوين الضم": "tanween_damm",
        "تنوين الفتح": "tanween_fath",
        "تنوين الكسر": "tanween_kasr",
        "السكون": "sukun",
        "السكون المقدر": "sukun_hidden",
        "الياء": "ya",
        "الواو": "waw",
        "الألف": "alif",
        "النون": "nun",
        "الفتح": "fath_short",
    }
    def norm_case(c):
        return CASE_NORM.get((c or "").strip(), c)
    def norm_marker(m):
        if not m:
            return m
        m = m.strip()
        # Try exact match first, then strip common prefixes
        if m in MARKER_NORM:
            return MARKER_NORM[m]
        for k, v in MARKER_NORM.items():
            if k in m:
                return v
        return m

    # Load eval set
    if args.eval == "gazelle":
        from irab_tashkeel.data.gazelle import load_gazelle_iraab
        items = load_gazelle_iraab()
        gold_pairs = []
        for it in items:
            pairs = split_sentence_iraab(it.answer)
            if pairs:
                gold_pairs.append((it.sentence, pairs))
    else:
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

    # Load predictor
    print(f"loading model from {args.model}")
    cfg = StructuredPredictorConfig(
        apply_constraints=False,
        apply_hierarchical=False,
        return_attention=False,
        render_prose=True,
        device="auto",
    )
    base_pred = StructuredPredictor(args.model, cfg=cfg)
    print(f"  taxonomy={base_pred.taxonomy}")

    # Optional Phase R retrieval wrapper or Phase R2 structural reasoning
    pred_retrieval = None
    use_struct = bool(args.use_structural_reasoning)
    if args.use_retrieval and use_struct:
        raise ValueError("--use_retrieval and --use_structural_reasoning are mutually exclusive")
    if args.use_retrieval:
        from irab_tashkeel.grammar_memory.memory import GrammarMemory
        from irab_tashkeel.grammar_memory.retrieval_predictor import RetrievalAugmentedPredictor
        memory = GrammarMemory(Path(args.retrieval_memory))
        pred_retrieval = RetrievalAugmentedPredictor(
            base_predictor=base_pred,
            memory=memory,
            lambda_=args.retrieval_lambda,
            k=args.retrieval_k,
        )
        print(f"  Phase R retrieval ENABLED: lambda={args.retrieval_lambda}, k={args.retrieval_k}")
    elif use_struct:
        from irab_tashkeel.grammar_memory.memory import GrammarMemory
        from irab_tashkeel.grammar_memory.structural_predictor import StructuralReasoningPredictor
        memory = GrammarMemory(Path(args.retrieval_memory))
        enabled_fams = (
            [s.strip() for s in args.enabled_families.split(",") if s.strip()]
            if args.enabled_families else None
        )
        pred_retrieval = StructuralReasoningPredictor(
            base_predictor=base_pred,
            memory=memory,
            tau_high=args.tau_high,
            tau_med=args.tau_med,
            lambda_strong=args.lambda_strong,
            retrieval_k=args.retrieval_k,
            enabled_families=enabled_fams,
        )
        print(f"  Phase R2 structural reasoning ENABLED: "
              f"tau_high={args.tau_high}, tau_med={args.tau_med}, "
              f"lambda_strong={args.lambda_strong}, "
              f"families={enabled_fams or 'all reasoners'}")
    pred = base_pred
    # Trace dump file (optional) — JSONL of structural reasoning traces
    trace_fh = None
    if args.dump_traces and use_struct:
        trace_fh = open(out_dir / "traces.jsonl", "w")

    # Per-construction tracking
    # construction -> list of (case_correct, role_correct, marker_correct, fully_correct,
    #                          gold_role, pred_role, conf)
    per_construction: Dict[str, List[tuple]] = defaultdict(list)
    overall_records: List[tuple] = []
    construction_sentence_count: Counter = Counter()
    total_sentences = 0

    # Process each sentence
    n_processed = 0
    for sent, gpairs in gold_pairs:
        # Build gold items list (word + canonical role) from extracted prose
        gold_items = []
        for w, irab in gpairs:
            extracted = extract(irab)
            gold_role = None
            if extracted is not None and extracted.role is not None:
                gold_role = canonicalize_role(extracted.role)
            gold_items.append({"word": w, "role": gold_role, "irab": irab,
                               "extracted": extracted})

        # Detect constructions in this sentence
        words = [w for w, _ in gpairs]
        constructions_present = detect_constructions(words, gold_items)
        total_sentences += 1
        for c in constructions_present:
            construction_sentence_count[c] += 1
        construction_sentence_count["overall"] += 1

        # Predict (with optional retrieval bias OR structural reasoning)
        retrieval_trace = None
        if pred_retrieval is not None:
            result, retrieval_trace = pred_retrieval.predict_sentence(sent)
        else:
            result = pred.predict_sentence(sent)
        # Optional trace dump for Phase R2
        if trace_fh is not None and retrieval_trace is not None and hasattr(retrieval_trace, "span_traces"):
            trace_fh.write(json.dumps({
                "sentence": sent,
                "n_constructions_detected": retrieval_trace.n_constructions_detected,
                "n_overrides": retrieval_trace.n_overrides,
                "n_strong_bias": retrieval_trace.n_strong_bias,
                "n_fallback": retrieval_trace.n_fallback,
                "spans": [{
                    "span": list(t.span), "family": t.family,
                    "particle_group": t.particle_group, "particle_surface": t.particle_surface,
                    "n_hits": t.n_hits, "confidence": round(t.confidence, 3),
                    "consensus_rate": round(t.consensus_rate, 3), "tier": t.tier,
                    "rule": t.rule, "predicted": t.predicted,
                } for t in retrieval_trace.span_traces],
            }, ensure_ascii=False) + "\n")
        # Align preds to gold by surface match (with normalisation fallback)
        pred_dicts = []
        for w in result.items:
            pred_dicts.append({
                "word": w.word,
                "role": w.role,
                "case": w.case,
                "marker": w.marker,
                "pos": w.pos,
                "role_conf": w.role_conf,
                "case_conf": getattr(w, "case_conf", None),
            })
        # Build word->pred lookup with normalised matching
        pred_by_norm = {_norm_ar(p["word"]): p for p in pred_dicts}

        for gold_item in gold_items:
            normed = _norm_ar(gold_item["word"])
            p = pred_by_norm.get(normed)
            if p is None:
                continue
            ext = gold_item["extracted"]
            if ext is None:
                continue

            # Per-word correctness — normalize extractor output to canonical labels
            gold_case = norm_case(ext.case)
            gold_marker = norm_marker(ext.marker)
            case_correct = (p["case"] == gold_case) if gold_case is not None else False
            role_correct = (gold_item["role"] is not None
                            and p["role"] == gold_item["role"])
            marker_correct = (p["marker"] == gold_marker) if gold_marker is not None else False
            # POS isn't always reliably extracted; require all three structural for "fully"
            fully_correct = case_correct and role_correct and marker_correct

            record = (case_correct, role_correct, marker_correct, fully_correct,
                      gold_item["role"], p["role"], p["role_conf"])
            overall_records.append(record)

            # Tag with each construction this sentence contains
            for c in constructions_present:
                per_construction[c].append(record)

        n_processed += 1
        if n_processed % 200 == 0:
            print(f"  ... processed {n_processed} sentences")

    # Aggregate
    def _summarize(records: List[tuple]) -> Dict:
        if not records:
            return {"n": 0}
        n = len(records)
        case_acc = sum(r[0] for r in records) / n
        role_acc = sum(r[1] for r in records) / n
        marker_acc = sum(r[2] for r in records) / n
        fully = sum(r[3] for r in records) / n
        # Calibration: mean conf on correct vs wrong (role)
        confs_correct = [r[6] for r in records if r[1] and r[6] is not None]
        confs_wrong = [r[6] for r in records if not r[1] and r[6] is not None]
        calib_correct = float(np.mean(confs_correct)) if confs_correct else 0.0
        calib_wrong = float(np.mean(confs_wrong)) if confs_wrong else 0.0
        return {
            "n_words": n,
            "case_acc": round(case_acc, 4),
            "role_acc": round(role_acc, 4),
            "marker_em": round(marker_acc, 4),
            "fully": round(fully, 4),
            "calib_correct": round(calib_correct, 3),
            "calib_wrong": round(calib_wrong, 3),
            "calib_gap": round(calib_correct - calib_wrong, 3),
        }

    summary: Dict = {
        "model": str(args.model),
        "eval": args.eval,
        "total_sentences": total_sentences,
        "construction_sentence_counts": dict(construction_sentence_count),
        "overall": _summarize(overall_records),
        "per_construction": {c: _summarize(per_construction[c]) for c in CONSTRUCTION_LABELS},
    }

    # Confusion matrices for the rare-class roles
    # Track per-(gold, pred) counts for {ism_kana, khabar_kana, ism_inna, khabar_inna,
    # mafoul_other, mawsool} on construction-relevant subsets.
    rare_roles = {"ism_kana", "khabar_kana", "ism_inna", "khabar_inna",
                  "mafoul_other", "mudaaf_ilayh"}
    confusion: Dict[str, Counter] = defaultdict(Counter)
    for c, records in per_construction.items():
        for case_c, role_c, mar_c, fully_c, gold_role, pred_role, _conf in records:
            if gold_role in rare_roles or pred_role in rare_roles:
                confusion[c][f"{gold_role or '<none>'}->{pred_role or '<none>'}"] += 1
    summary["rare_role_confusion"] = {c: dict(confusion[c].most_common(20)) for c in confusion}

    # Write JSON
    out_path = out_dir / "per_construction_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # Write Markdown summary
    md = []
    md.append(f"# Per-construction eval — {args.eval} — model: {args.model}\n")
    md.append(f"Total sentences: {total_sentences}\n")
    md.append(f"Total word judgments: {summary['overall']['n_words']}\n\n")

    md.append("## Construction prevalence\n")
    md.append("| Construction | Sentences containing |")
    md.append("|---|---:|")
    for c in CONSTRUCTION_LABELS + ["overall"]:
        n = construction_sentence_count.get(c, 0)
        pct = (n / total_sentences * 100) if total_sentences else 0
        md.append(f"| {c} | {n} ({pct:.1f}%) |")
    md.append("")

    md.append("## Per-construction accuracy\n")
    md.append("| Construction | n_words | case | role | marker | **fully** | calib gap |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in CONSTRUCTION_LABELS + ["overall"]:
        s = summary["per_construction"].get(c) if c != "overall" else summary["overall"]
        if s is None or s.get("n_words", 0) == 0:
            md.append(f"| {c} | 0 | — | — | — | — | — |")
            continue
        md.append(f"| {c} | {s['n_words']} | "
                  f"{s['case_acc']*100:.1f} | {s['role_acc']*100:.1f} | "
                  f"{s['marker_em']*100:.1f} | **{s['fully']*100:.1f}** | "
                  f"{s['calib_gap']:+.3f} |")
    md.append("")

    if confusion:
        md.append("## Top role confusion patterns per construction\n")
        for c in confusion:
            md.append(f"### {c}\n")
            md.append("| gold → pred | count |")
            md.append("|---|---:|")
            for pair, count in confusion[c].most_common(10):
                md.append(f"| `{pair}` | {count} |")
            md.append("")

    md_path = out_dir / "per_construction_summary.md"
    md_path.write_text("\n".join(md))

    if trace_fh is not None:
        trace_fh.close()

    print(f"\n=== {args.eval} per-construction summary ===")
    for c in CONSTRUCTION_LABELS:
        s = summary["per_construction"][c]
        n = s.get("n_words", 0)
        if n == 0:
            print(f"  {c:>20s}: no words")
            continue
        print(f"  {c:>20s}: n={n:4d} | case={s['case_acc']*100:5.1f} | "
              f"role={s['role_acc']*100:5.1f} | mar={s['marker_em']*100:5.1f} | "
              f"fully={s['fully']*100:5.1f} | calib_gap={s['calib_gap']:+.3f}")
    s = summary["overall"]
    print(f"  {'overall':>20s}: n={s['n_words']:4d} | case={s['case_acc']*100:5.1f} | "
          f"role={s['role_acc']*100:5.1f} | mar={s['marker_em']*100:5.1f} | "
          f"fully={s['fully']*100:5.1f}")
    print(f"\nWritten: {out_path}\n         {md_path}")


if __name__ == "__main__":
    main()
