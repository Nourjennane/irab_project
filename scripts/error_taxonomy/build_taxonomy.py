"""Step 16 of next-gen branch — error-taxonomy population pass.

Walks Phase 3-A predictions on Gazelle + MASAQ, computes per-word
errors with full structural metadata + Stanza dep features, applies
deterministic rule-based tagging across 18 categories, writes:

  - per-error JSONL (docs/error_analysis_v2/<domain>_errors_tagged.jsonl)
  - global histograms (docs/error_analysis_v2/histograms.md)
  - per-construction breakdowns
  - severity-weighted ranking
  - ranked bottleneck report (docs/error_analysis_v2/bottlenecks.md)

NO TRAINING. Pure analysis on the frozen baseline.

Categories (18) — see docs/error_taxonomy.md:

DETERMINISTIC (rule-based tagging):
  T01 morphology_failure
  T02 local_syntax_failure
  T03 long_range_dependency_failure
  T04 nested_clause_failure
  T07 parser_alignment_failure
  T08 annotation_sparsity
  T09 rare_construction_collapse
  T10 confidence_calibration_pathology
  T12 evaluator_limitation
  T13 implicit_governor_failure
  T15 coordination_ambiguity
  T16 clause_attachment_ambiguity
  T18 construction_overlap_interference

CANDIDATE (rule-based but needs human/LLM verification):
  T05 semantic_ambiguity        — flagged by role-confusion patterns
  T06 discourse_context_failure — flagged by pronoun/cross-sentence patterns
  T11 retrieval_mismatch        — same Gazelle/MASAQ asymmetry as Phase R-C/R2
  T14 omitted_element_reasoning — flagged by 'other'-role + low base conf
  T17 semantic_role_confusion   — flagged by case-correct/role-wrong with both valid

Each error can carry multiple tags.
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


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

# 18 category codes
CATEGORIES = [
    "T01_morphology_failure",
    "T02_local_syntax_failure",
    "T03_long_range_dependency_failure",
    "T04_nested_clause_failure",
    "T05_semantic_ambiguity",
    "T06_discourse_context_failure",
    "T07_parser_alignment_failure",
    "T08_annotation_sparsity",
    "T09_rare_construction_collapse",
    "T10_confidence_calibration_pathology",
    "T11_retrieval_mismatch",
    "T12_evaluator_limitation",
    "T13_implicit_governor_failure",
    "T14_omitted_element_reasoning",
    "T15_coordination_ambiguity",
    "T16_clause_attachment_ambiguity",
    "T17_semantic_role_confusion",
    "T18_construction_overlap_interference",
]

# Rare/zero-fully constructions on Gazelle (per frozen baseline metrics)
RARE_GAZELLE = {"istithna", "quranic_proxy"}

# Pronoun POS for discourse heuristic
PRONOUN_ROLES = {"other"}  # frozen-baseline collapse for pronouns

# Coordination dep labels (Stanza UD)
COORDINATION_DEPRELS = {"conj", "cc"}
# Embedded clause dep labels
EMBEDDED_CLAUSE_DEPRELS = {"ccomp", "xcomp", "acl", "advcl", "csubj"}
# Nominal modifier (used for ambiguous attachment)
ATTACHMENT_AMBIGUOUS_DEPRELS = {"acl", "nmod", "amod"}

# Valid roles for "both valid" semantic confusion check
VALID_ROLES_SET = set()  # filled at runtime from schema


def _norm_ar(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[ً-ٰٟ]", "", s)
    s = re.sub(r"[^ء-ي]+", "", s)
    return s


def _dep_distance(head_idx: int, child_idx: int) -> int:
    """Approximate dep distance — surface distance for first pass.

    Stanza head_idx is 1-based, with 0 = root. We approximate
    long-range as |head - child| in surface tokens.
    """
    if head_idx is None or head_idx <= 0:
        return -1
    return abs(head_idx - 1 - child_idx)


# ---------------------------------------------------------------------------
# Rule-based tagger
# ---------------------------------------------------------------------------

def tag_error(err: Dict, sent_meta: Dict) -> List[str]:
    """Apply 18-category tagging rules to one error record. Multiple tags allowed.

    err must have:
        word, position, predicted, gold, predicted_conf,
        construction_tags, dep_info, sentence_idx, sentence_length, domain,
        case_correct, role_correct, marker_correct, fully_correct
    sent_meta has aggregate sentence-level stats.
    """
    tags: List[str] = []
    p = err["predicted"]
    g = err["gold"]
    dep = err.get("dep_info", {}) or {}
    deprel = dep.get("deprel", "")
    head_idx = dep.get("head_idx", -1)
    parser_conf = dep.get("parser_conf", 1.0)
    has_dep_for_word = dep.get("has_dep", False)
    construction_tags = err.get("construction_tags", [])
    domain = err.get("domain", "")

    # T08 annotation_sparsity: gold field is None
    if g.get("case") is None or g.get("role") is None or g.get("marker") is None:
        if not err["fully_correct"]:
            tags.append("T08_annotation_sparsity")

    # T12 evaluator_limitation: gold extracted a non-canonical role
    # (e.g., ism_majrur for a kana-context word, or roles that map to
    # "other" during canonicalisation). Most caught by sparsity above; this
    # specifically flags non-None golds that are likely extractor artifacts.
    gold_role = g.get("role")
    if gold_role and (
        # Known confounded role labels in kana/inna contexts
        (gold_role in ("ism_majrur", "mafoul_other") and
         any(c in construction_tags for c in ("kana_sisters", "inna_sisters")))
        # Roles canonicalised to "other" — extractor couldn't resolve
        or gold_role == "other"
    ):
        if not err["role_correct"]:
            tags.append("T12_evaluator_limitation")

    # T07 parser_alignment_failure: word has no dep info OR low parser conf
    if not has_dep_for_word and not err["fully_correct"]:
        tags.append("T07_parser_alignment_failure")
    elif parser_conf < 0.5 and not err["fully_correct"]:
        tags.append("T07_parser_alignment_failure")

    # T01 morphology_failure: case correct, marker wrong (pure marker-form confusion)
    if err["case_correct"] and not err["marker_correct"] and g.get("marker") is not None:
        tags.append("T01_morphology_failure")

    # T03 long-range dependency failure: head distance >= 4 OR clause-cross
    pos = err["position"]
    head_distance = _dep_distance(head_idx, pos)
    if head_distance >= 4 and not err["fully_correct"]:
        tags.append("T03_long_range_dependency_failure")

    # T02 local_syntax_failure: no construction tag, dep distance ≤ 2
    if (not construction_tags or construction_tags == [] or construction_tags == ["overall"]):
        if 0 <= head_distance <= 2 and not err["fully_correct"]:
            tags.append("T02_local_syntax_failure")

    # T04 nested_clause_failure: word's deprel indicates embedded clause
    if deprel in EMBEDDED_CLAUSE_DEPRELS and not err["fully_correct"]:
        tags.append("T04_nested_clause_failure")

    # T15 coordination_ambiguity
    if deprel in COORDINATION_DEPRELS and not err["fully_correct"]:
        tags.append("T15_coordination_ambiguity")

    # T16 clause_attachment_ambiguity
    if deprel in ATTACHMENT_AMBIGUOUS_DEPRELS and parser_conf < 0.7 and not err["fully_correct"]:
        tags.append("T16_clause_attachment_ambiguity")

    # T18 construction_overlap_interference: word has 2+ construction tags
    real_constructions = [c for c in construction_tags if c not in ("overall",)]
    if len(real_constructions) >= 2 and not err["fully_correct"]:
        tags.append("T18_construction_overlap_interference")

    # T09 rare_construction_collapse: error in a 0% construction
    if domain == "gazelle":
        if any(c in RARE_GAZELLE for c in construction_tags) and not err["fully_correct"]:
            tags.append("T09_rare_construction_collapse")

    # T10 confidence_calibration_pathology: high confidence on wrong prediction
    if not err["case_correct"] and p.get("case_conf", 0) >= 0.8:
        tags.append("T10_confidence_calibration_pathology")
    elif not err["role_correct"] and p.get("role_conf", 0) >= 0.8:
        tags.append("T10_confidence_calibration_pathology")
    elif not err["marker_correct"] and p.get("marker_conf", 0) >= 0.8:
        tags.append("T10_confidence_calibration_pathology")

    # T11 retrieval_mismatch (candidate): kana/istithna/quranic on Gazelle
    # — same surface where Phase R-C/R2 gates failed
    if domain == "gazelle" and any(c in ("kana_sisters", "istithna", "quranic_proxy")
                                     for c in construction_tags):
        if not err["fully_correct"]:
            tags.append("T11_retrieval_mismatch")

    # T13 implicit_governor_failure: word's gold case is non-default (e.g. nasb on noun)
    # but its head is root (no syntactic governor)
    if head_idx == 0 and g.get("case") in ("nasb", "jarr") and not err["case_correct"]:
        tags.append("T13_implicit_governor_failure")

    # T14 omitted_element_reasoning (candidate): predicted "other"
    # role with low confidence — usually pronouns / implicit elements
    if p.get("role") == "other" and p.get("role_conf", 1.0) < 0.5:
        if not err["role_correct"]:
            tags.append("T14_omitted_element_reasoning")

    # T17 semantic_role_confusion (candidate): case correct, role wrong,
    # both predicted and gold roles are canonical (not "other")
    if (err["case_correct"] and not err["role_correct"]
        and p.get("role") and g.get("role")
        and p.get("role") != "other" and g.get("role") != "other"):
        tags.append("T17_semantic_role_confusion")

    # T05 semantic_ambiguity (candidate): hal vs naat, mafoul_bih vs mafoul_other,
    # known semantic-disambiguation pairs
    SEMANTIC_PAIRS = [
        {"hal", "naat"}, {"mafoul_bih", "mafoul_other"},
        {"khabar", "naat"}, {"badal", "naat"},
        {"badal", "matuf"}, {"naib_fail", "fail"},
    ]
    if g.get("role") and p.get("role"):
        gp = {g["role"], p["role"]}
        if any(s == gp for s in SEMANTIC_PAIRS) and not err["role_correct"]:
            tags.append("T05_semantic_ambiguity")

    # T06 discourse_context_failure (candidate): pronoun-like errors
    # (predicted role 'other', POS pronoun)
    if p.get("pos") == "pronoun" and not err["fully_correct"]:
        tags.append("T06_discourse_context_failure")

    return list(set(tags))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/phase3a_491240/final")
    ap.add_argument("--out_dir", default="docs/error_analysis_v2")
    ap.add_argument("--with_stanza", action="store_true",
                    help="Run Stanza for dep features (slower; richer tagging)")
    ap.add_argument("--max_gazelle", type=int, default=None,
                    help="Limit Gazelle sentences (for debug)")
    ap.add_argument("--max_masaq", type=int, default=None,
                    help="Limit MASAQ sentences (for debug)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import torch
    import torch.nn.functional as F

    from irab_tashkeel.inference.structured_predictor import (
        StructuredPredictor, StructuredPredictorConfig,
    )
    from irab_tashkeel.evaluation.structural import extract, split_sentence_iraab
    from irab_tashkeel.structured.schema import (
        ROLE_LABELS, canonicalize_role,
    )
    from irab_tashkeel.grammar_memory.signature import detect_constructions_in_record

    global VALID_ROLES_SET
    VALID_ROLES_SET = set(ROLE_LABELS)

    CASE_NORM = {
        "marfu": "raf", "mansub": "nasb", "majrur": "jarr", "majzum": "jazm",
        "mabni": "mabni", "raf": "raf", "nasb": "nasb", "jarr": "jarr", "jazm": "jazm",
    }
    MARKER_NORM = {
        "الضمة الظاهرة": "damma_visible", "الضمة المقدرة": "damma_hidden",
        "الفتحة الظاهرة": "fatha_visible", "الفتحة المقدرة": "fatha_hidden",
        "الكسرة الظاهرة": "kasra_visible", "الكسرة المقدرة": "kasra_hidden",
        "تنوين الضم": "tanween_damm", "تنوين الفتح": "tanween_fath",
        "تنوين الكسر": "tanween_kasr", "السكون": "sukun",
        "السكون المقدر": "sukun_hidden", "الياء": "ya", "الواو": "waw",
        "الألف": "alif", "النون": "nun", "الفتح": "fath_short",
    }
    def nc(c): return CASE_NORM.get((c or "").strip(), c)
    def nm(m):
        if not m: return m
        m = m.strip()
        if m in MARKER_NORM: return MARKER_NORM[m]
        for k, v in MARKER_NORM.items():
            if k in m: return v
        return m

    print(f"Loading Phase 3-A from {args.model}")
    cfg = StructuredPredictorConfig(
        apply_constraints=False, apply_hierarchical=False,
        return_attention=False, render_prose=False, device="auto",
    )
    pred = StructuredPredictor(args.model, cfg=cfg)

    # Optional Stanza pipeline
    nlp = None
    if args.with_stanza:
        print("Loading Stanza Arabic UD parser...")
        try:
            import stanza
            nlp = stanza.Pipeline(
                lang="ar", processors="tokenize,pos,lemma,depparse",
                tokenize_pretokenized=True, download_method=None,
                use_gpu=True, verbose=False,
            )
            print("Stanza loaded")
        except Exception as e:
            print(f"Stanza failed to load ({e}); continuing without dep features")
            nlp = None

    # Load eval sets
    from irab_tashkeel.data.gazelle import load_gazelle_iraab
    print("Loading Gazelle...")
    gazelle_items = load_gazelle_iraab()
    gazelle_pairs = []
    for it in gazelle_items:
        pairs = split_sentence_iraab(it.answer)
        if pairs:
            gazelle_pairs.append((it.sentence, pairs))
    if args.max_gazelle:
        gazelle_pairs = gazelle_pairs[:args.max_gazelle]
    print(f"  {len(gazelle_pairs)} Gazelle sentences")

    print("Loading MASAQ...")
    masaq_pairs = []
    with open("data/masaq_eval.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            row = json.loads(line)
            pairs = [(it.get("word", ""), it.get("irab", ""))
                     for it in row.get("items", [])
                     if isinstance(it, dict) and it.get("word") and it.get("irab")]
            if row.get("sentence") and pairs:
                masaq_pairs.append((row["sentence"], pairs))
    if args.max_masaq:
        masaq_pairs = masaq_pairs[:args.max_masaq]
    print(f"  {len(masaq_pairs)} MASAQ sentences")

    # Process each domain
    all_records: List[Dict] = []
    sentence_meta_log: List[Dict] = []

    for domain, pairs in [("gazelle", gazelle_pairs), ("masaq", masaq_pairs)]:
        print(f"\n=== Processing {domain} ({len(pairs)} sentences) ===")
        for sent_idx, (sent, gpairs) in enumerate(pairs):
            # Build gold items
            gold_items = []
            for w, irab in gpairs:
                ext = extract(irab)
                gr = canonicalize_role(ext.role) if (ext and ext.role) else None
                gold_items.append({
                    "word": w,
                    "case": nc(ext.case if ext else None),
                    "role": gr,
                    "marker": nm(ext.marker if ext else None),
                    "irab": irab,
                })

            # Predict
            result = pred.predict_sentence(sent)
            pred_by_norm = {_norm_ar(w.word): w for w in result.items}

            # Stanza dep parse (optional)
            dep_per_word = {}
            if nlp is not None:
                try:
                    words_only = [w for w, _ in gpairs]
                    doc = nlp([words_only])
                    if doc.sentences:
                        for w in doc.sentences[0].words:
                            dep_per_word[_norm_ar(w.text)] = {
                                "deprel": w.deprel,
                                "head_idx": w.head,
                                "upos": w.upos,
                                "has_dep": True,
                                "parser_conf": 1.0,
                            }
                except Exception:
                    pass

            # Detect constructions
            words = [w for w, _ in gpairs]
            initial_record = {
                "sentence": " ".join(words),
                "items": [{"word": w,
                           "role": (pred_by_norm.get(_norm_ar(w)).role
                                    if pred_by_norm.get(_norm_ar(w)) else "")}
                          for w in words],
                "source": "_query",
            }
            constructions = detect_constructions_in_record(initial_record)

            # Per-sentence construction tags (sentence-level)
            sentence_constructions = set()
            for c in constructions:
                sentence_constructions.add(c["construction"])

            # Per-word construction membership
            word_constructions: Dict[int, Set[str]] = defaultdict(set)
            for c in constructions:
                start, end = c["span"]
                for i in range(start, min(end, len(words))):
                    word_constructions[i].add(c["construction"])

            # Per-word records
            sent_words_n = len(words)
            for i, gold_item in enumerate(gold_items):
                w = gold_item["word"]
                normed = _norm_ar(w)
                p = pred_by_norm.get(normed)
                if p is None:
                    continue

                gcase = gold_item["case"]
                grole = gold_item["role"]
                gmarker = gold_item["marker"]
                pcase = p.case
                prole = p.role
                pmarker = p.marker

                case_correct = (pcase == gcase) if gcase is not None else None
                role_correct = (prole == grole) if grole is not None else None
                marker_correct = (pmarker == gmarker) if gmarker is not None else None
                # fully = all 3 fields correct (None counts as wrong)
                fully_correct = (
                    case_correct is True
                    and role_correct is True
                    and marker_correct is True
                )

                construction_tags_word = sorted(word_constructions.get(i, set()))
                # Add sentence-level tags too (for per_construction style aggregation)
                sentence_tags_word = sorted(sentence_constructions)

                record = {
                    "domain": domain,
                    "sentence_idx": sent_idx,
                    "sentence": sent,
                    "sentence_length": sent_words_n,
                    "n_constructions_in_sentence": len(constructions),
                    "position": i,
                    "word": w,
                    "predicted": {
                        "case": pcase, "role": prole, "marker": pmarker,
                        "pos": p.pos,
                        "case_conf": p.case_conf, "role_conf": p.role_conf,
                        "marker_conf": p.marker_conf, "pos_conf": p.pos_conf,
                    },
                    "gold": {"case": gcase, "role": grole, "marker": gmarker},
                    "case_correct": case_correct,
                    "role_correct": role_correct,
                    "marker_correct": marker_correct,
                    "fully_correct": fully_correct,
                    "construction_tags": construction_tags_word,
                    "sentence_construction_tags": sentence_tags_word,
                    "dep_info": dep_per_word.get(normed, {}),
                    "is_error": not fully_correct,
                    "tags": [],
                }
                # Tag
                if not fully_correct:
                    record["tags"] = tag_error(record, {})
                all_records.append(record)

            sentence_meta_log.append({
                "domain": domain, "sentence_idx": sent_idx,
                "sentence_length": sent_words_n,
                "n_constructions": len(constructions),
                "constructions": sorted(sentence_constructions),
            })

            if (sent_idx + 1) % 100 == 0:
                print(f"  ... {sent_idx + 1}/{len(pairs)} done")

    # Write per-domain JSONL
    for domain in ("gazelle", "masaq"):
        path = out_dir / f"{domain}_errors_tagged.jsonl"
        with path.open("w") as fh:
            for r in all_records:
                if r["domain"] == domain:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {path}")

    # Sentence meta log
    (out_dir / "sentence_meta.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in sentence_meta_log)
    )

    # ---- Aggregations ----
    error_records = [r for r in all_records if r["is_error"]]
    print(f"\nTotal records: {len(all_records)}")
    print(f"Total errors:  {len(error_records)}")
    by_domain = Counter(r["domain"] for r in error_records)
    print(f"  Gazelle errors: {by_domain['gazelle']}")
    print(f"  MASAQ errors:   {by_domain['masaq']}")

    # Global tag histogram
    global_hist: Counter = Counter()
    severity_weighted: Counter = Counter()  # weight 3 if fully-blocking and case wrong, 2 if role, 1 if marker
    for r in error_records:
        for t in r["tags"]:
            global_hist[t] += 1
            # severity
            w = 0
            if r["case_correct"] is False: w += 3
            if r["role_correct"] is False: w += 2
            if r["marker_correct"] is False: w += 1
            severity_weighted[t] += w

    # Per-domain
    domain_hist: Dict[str, Counter] = {"gazelle": Counter(), "masaq": Counter()}
    for r in error_records:
        for t in r["tags"]:
            domain_hist[r["domain"]][t] += 1

    # Per-construction
    construction_hist: Dict[str, Counter] = defaultdict(Counter)
    for r in error_records:
        for c in r["sentence_construction_tags"]:
            for t in r["tags"]:
                construction_hist[c][t] += 1

    # Histograms.md
    md = []
    md.append(f"# Error Taxonomy — Empirical Histograms\n")
    md.append(f"Generated by `scripts/error_taxonomy/build_taxonomy.py`\n")
    md.append(f"\nTotal records: {len(all_records)}")
    md.append(f"Total errors:  {len(error_records)}")
    md.append(f"  Gazelle errors: {by_domain['gazelle']}")
    md.append(f"  MASAQ errors:   {by_domain['masaq']}\n")
    md.append("## Global tag histogram (counts)\n")
    md.append("| category | count | %errors |")
    md.append("|---|---:|---:|")
    n_err = max(len(error_records), 1)
    for t in CATEGORIES:
        c = global_hist.get(t, 0)
        md.append(f"| {t} | {c} | {100*c/n_err:.1f}% |")
    md.append("")

    md.append("## Severity-weighted ranking\n")
    md.append("| category | severity score |")
    md.append("|---|---:|")
    for t, score in severity_weighted.most_common():
        md.append(f"| {t} | {score} |")
    md.append("")

    md.append("## Gazelle vs MASAQ\n")
    md.append("| category | Gazelle | MASAQ | Gaz % | MASAQ % |")
    md.append("|---|---:|---:|---:|---:|")
    n_g = max(by_domain['gazelle'], 1)
    n_m = max(by_domain['masaq'], 1)
    for t in CATEGORIES:
        g = domain_hist["gazelle"].get(t, 0)
        m = domain_hist["masaq"].get(t, 0)
        md.append(f"| {t} | {g} | {m} | {100*g/n_g:.1f}% | {100*m/n_m:.1f}% |")
    md.append("")

    md.append("## Per-construction tag histograms\n")
    for c in sorted(construction_hist.keys()):
        ctr = construction_hist[c]
        n_c = sum(ctr.values()) or 1
        md.append(f"### {c} (n={n_c} tag-incidences)\n")
        md.append("| category | count | % |")
        md.append("|---|---:|---:|")
        for t in CATEGORIES:
            v = ctr.get(t, 0)
            if v == 0: continue
            md.append(f"| {t} | {v} | {100*v/n_c:.1f}% |")
        md.append("")

    (out_dir / "histograms.md").write_text("\n".join(md))
    print(f"Wrote {out_dir / 'histograms.md'}")

    # ---- Bottlenecks report ----
    n_total = len(error_records)
    def pct(t):
        return 100 * global_hist.get(t, 0) / max(n_total, 1)

    bot = []
    bot.append("# Ranked Bottleneck Report — Step 16 Error Taxonomy\n")
    bot.append(f"Source: Phase 3-A on Gazelle ({by_domain['gazelle']} errors) + MASAQ ({by_domain['masaq']} errors).\n")

    bot.append("## Bottleneck classes (% of all errors carrying tag)\n")
    ranked = sorted(CATEGORIES, key=lambda t: -global_hist.get(t, 0))
    bot.append("| rank | category | % errors | severity score |")
    bot.append("|---:|---|---:|---:|")
    for i, t in enumerate(ranked):
        bot.append(f"| {i+1} | {t} | {pct(t):.1f}% | {severity_weighted.get(t, 0)} |")
    bot.append("")

    bot.append("## Solvability classification\n")
    bot.append("| category | realistic-data | better-syntax | larger-models | reasoning-supervision | annotation-limited |")
    bot.append("|---|:---:|:---:|:---:|:---:|:---:|")
    SOLVABILITY = {
        "T01_morphology_failure":           ("✓", " ", "✓", " ", " "),
        "T02_local_syntax_failure":         ("✓", "✓", "✓", " ", " "),
        "T03_long_range_dependency_failure":(" ", "✓", "✓", "✓", " "),
        "T04_nested_clause_failure":        (" ", "✓", " ", "✓", " "),
        "T05_semantic_ambiguity":           (" ", " ", "✓", "✓", " "),
        "T06_discourse_context_failure":    (" ", " ", "✓", "✓", " "),
        "T07_parser_alignment_failure":     ("✓", "✓", " ", " ", " "),
        "T08_annotation_sparsity":          ("✓", " ", " ", " ", "✓"),
        "T09_rare_construction_collapse":   ("✓", " ", "✓", " ", " "),
        "T10_confidence_calibration_pathology": ("✓", " ", " ", "✓", " "),
        "T11_retrieval_mismatch":           ("✓", " ", " ", " ", " "),
        "T12_evaluator_limitation":         (" ", " ", " ", " ", "✓"),
        "T13_implicit_governor_failure":    (" ", "✓", " ", "✓", " "),
        "T14_omitted_element_reasoning":    (" ", " ", " ", "✓", " "),
        "T15_coordination_ambiguity":       ("✓", "✓", " ", " ", " "),
        "T16_clause_attachment_ambiguity":  ("✓", "✓", " ", "✓", " "),
        "T17_semantic_role_confusion":      (" ", " ", "✓", "✓", " "),
        "T18_construction_overlap_interference": (" ", "✓", " ", "✓", " "),
    }
    for t in ranked:
        flags = SOLVABILITY.get(t, (" ", " ", " ", " ", " "))
        bot.append(f"| {t} | {flags[0]} | {flags[1]} | {flags[2]} | {flags[3]} | {flags[4]} |")
    bot.append("")

    # Aggregate solvability percentages
    realistic = sum(global_hist.get(t, 0) for t in CATEGORIES if SOLVABILITY[t][0] == "✓")
    syntax    = sum(global_hist.get(t, 0) for t in CATEGORIES if SOLVABILITY[t][1] == "✓")
    larger    = sum(global_hist.get(t, 0) for t in CATEGORIES if SOLVABILITY[t][2] == "✓")
    reasoning = sum(global_hist.get(t, 0) for t in CATEGORIES if SOLVABILITY[t][3] == "✓")
    annotation = sum(global_hist.get(t, 0) for t in CATEGORIES if SOLVABILITY[t][4] == "✓")
    total_tag_incidences = sum(global_hist.values()) or 1
    bot.append("## Aggregate addressability (tag-incidence weighted)\n")
    bot.append(f"- Realistic with more data:        {realistic} ({100*realistic/total_tag_incidences:.1f}%)")
    bot.append(f"- Better syntax / parser quality:  {syntax} ({100*syntax/total_tag_incidences:.1f}%)")
    bot.append(f"- Larger / better encoders:        {larger} ({100*larger/total_tag_incidences:.1f}%)")
    bot.append(f"- Reasoning supervision:           {reasoning} ({100*reasoning/total_tag_incidences:.1f}%)")
    bot.append(f"- Annotation-limited (ceiling):    {annotation} ({100*annotation/total_tag_incidences:.1f}%)")
    bot.append("")

    (out_dir / "bottlenecks.md").write_text("\n".join(bot))
    print(f"Wrote {out_dir / 'bottlenecks.md'}")

    # Print summary to stdout
    print("\n=== TOP-LEVEL SUMMARY ===")
    print(f"Errors total: {n_total}")
    print(f"Errors Gazelle: {by_domain['gazelle']}, MASAQ: {by_domain['masaq']}")
    print("\nTop 10 categories by tag-incidence:")
    for t in ranked[:10]:
        print(f"  {t}: {global_hist.get(t, 0)} ({pct(t):.1f}%)")


if __name__ == "__main__":
    main()
