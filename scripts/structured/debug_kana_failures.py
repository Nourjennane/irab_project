"""Phase R2 — KanaReasoner failure debug dump.

For every Gazelle (and optionally MASAQ) kana_sisters construction span,
dumps a per-span record with full introspection so we can diagnose
*why* the reasoner produces non-canonical predictions.

Per-span fields (matches the user's 15-field spec):
  1. sentence          — original sentence
  2. trigger           — the kana-family particle surface form
  3. span              — (start, end_excl) word indices
  4. retrieved         — top-k retrieved analogues with sentence + items
  5. retrieved_deps    — dep info per retrieved item (deprel, head_idx, gov_upos)
  6. alignment         — surface-position alignment used by the reasoner
                         + a hypothetical dep-tree alignment for comparison
  7. dep_deltas        — per-position deprel/gov_upos differences between query
                         (Phase 3-A items) and each retrieved analogue
  8. transformation_candidates — per-position vote distribution (case, role, marker)
                                  with vote counts
  9. consensus_per_pos — per-position consensus rates
 10. final_transformation — what the reasoner emitted
 11. tier_fired       — override / strong_bias / fallback / no_retrievals
 12. before_override  — Phase 3-A baseline predictions per word (case, role, marker, conf)
 13. after_override   — Final predictions per word (case, role, marker, conf)
 14. gold             — Gold labels per word (extracted from Gazelle/MASAQ prose)
 15. calibration      — per-word confidence values, before/after, with correctness flags

Output:
  runs/phaseR2_debug/kana_failures/dump.jsonl   — one JSON line per kana span
  runs/phaseR2_debug/kana_failures/summary.md   — aggregated report
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
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _norm_ar(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[ً-ٰٟ]", "", s)
    s = re.sub(r"[^ء-ي]+", "", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/phase3a_491240/final")
    ap.add_argument("--memory", default="data/grammar_memory/")
    ap.add_argument("--eval", choices=["gazelle", "masaq"], default="gazelle")
    ap.add_argument("--out_dir", default="runs/phaseR2_debug/kana_failures")
    ap.add_argument("--tau_high", type=float, default=0.75)
    ap.add_argument("--tau_med", type=float, default=0.50)
    ap.add_argument("--lambda_strong", type=float, default=1.5)
    ap.add_argument("--retrieval_k", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import numpy as np
    import torch
    import torch.nn.functional as F

    from irab_tashkeel.inference.structured_predictor import (
        StructuredPredictor, StructuredPredictorConfig,
    )
    from irab_tashkeel.evaluation.structural import extract, split_sentence_iraab
    from irab_tashkeel.structured.schema import (
        CASE_LABELS, ROLE_LABELS, MARKER_LABELS, POS_LABELS,
        CASE_TO_ID, ROLE_TO_ID, MARKER_TO_ID,
        ID_TO_CASE, ID_TO_ROLE, ID_TO_MARKER,
        canonicalize_role, canonicalize_case, canonicalize_marker,
    )
    from irab_tashkeel.grammar_memory.memory import GrammarMemory
    from irab_tashkeel.grammar_memory.signature import (
        ConstructionInstance, build_signature, detect_constructions_in_record,
    )
    from irab_tashkeel.grammar_memory.structural_reasoner import (
        KanaReasoner, REASONER_REGISTRY, _vote_on_position,
    )

    CASE_NORM = {
        "marfu": "raf", "mansub": "nasb", "majrur": "jarr", "majzum": "jazm",
        "mabni": "mabni",
        "raf": "raf", "nasb": "nasb", "jarr": "jarr", "jazm": "jazm",
    }
    MARKER_NORM = {
        "الضمة الظاهرة": "damma_visible", "الضمة المقدرة": "damma_hidden",
        "الفتحة الظاهرة": "fatha_visible", "الفتحة المقدرة": "fatha_hidden",
        "الكسرة الظاهرة": "kasra_visible", "الكسرة المقدرة": "kasra_hidden",
        "تنوين الضم": "tanween_damm", "تنوين الفتح": "tanween_fath",
        "تنوين الكسر": "tanween_kasr",
        "السكون": "sukun", "السكون المقدر": "sukun_hidden",
        "الياء": "ya", "الواو": "waw", "الألف": "alif", "النون": "nun",
        "الفتح": "fath_short",
    }
    def norm_case(c):
        return CASE_NORM.get((c or "").strip(), c)
    def norm_marker(m):
        if not m:
            return m
        m = m.strip()
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

    print(f"loaded {len(gold_pairs)} {args.eval} sentences")
    print(f"loading model from {args.model}")
    cfg = StructuredPredictorConfig(
        apply_constraints=False, apply_hierarchical=False,
        return_attention=False, render_prose=False, device="auto",
    )
    base_pred = StructuredPredictor(args.model, cfg=cfg)
    memory = GrammarMemory(Path(args.memory))
    reasoner = KanaReasoner()

    out_path = out_dir / "dump.jsonl"
    out_fh = out_path.open("w")

    # Aggregate counters for the summary
    n_kana_spans = 0
    n_with_retrievals = 0
    n_override = 0
    n_strong_bias = 0
    n_fallback = 0
    # Per-position correctness BEFORE vs AFTER override (does override help or hurt?)
    pos_outcome = defaultdict(lambda: Counter())
    # Field by field {case, role, marker} consensus distributions
    consensus_distrib = {"case": [], "role": [], "marker": []}
    # Per-particle stats
    per_particle: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "n_spans": 0, "n_words": 0,
        "case_correct_before": 0, "case_correct_after": 0,
        "role_correct_before": 0, "role_correct_after": 0,
        "marker_correct_before": 0, "marker_correct_after": 0,
        "fully_correct_before": 0, "fully_correct_after": 0,
        "wrong_high_conf": 0,
    })
    # Calibration: for wrong-high-confidence after override, dump (sentence, position, predicted, gold, conf)
    wrong_high_conf_dump = []

    for sent, gpairs in gold_pairs:
        gold_items = []
        for w, irab in gpairs:
            extracted = extract(irab)
            gold_role = None
            if extracted is not None and extracted.role is not None:
                gold_role = canonicalize_role(extracted.role)
            gold_items.append({
                "word": w, "role": gold_role,
                "case": norm_case(extracted.case) if extracted else None,
                "marker": norm_marker(extracted.marker) if extracted else None,
                "irab": irab,
            })

        # ---- Phase 3-A baseline forward (capture full logits + per-position labels) ----
        enc = base_pred._encode_sentence(sent)
        if enc is None:
            continue
        from irab_tashkeel.structured.model import _word_first_pool
        with torch.no_grad():
            enc_out = base_pred.model.encoder(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            )
            hidden = enc_out.last_hidden_state
            pooled = _word_first_pool(hidden, enc["word_starts"], enc["word_mask"])
            if getattr(base_pred.model, "enable_dep_features", False):
                B, W = pooled.shape[:2]
                df = base_pred.model.dep_feature_encoder
                dep_emb_dim = (df.deprel_embed.embedding_dim
                                + df.head_dir_embed.embedding_dim
                                + df.head_dist_embed.embedding_dim
                                + df.gov_pos_embed.embedding_dim)
                dep_emb = pooled.new_zeros(B, W, dep_emb_dim)
                h_aug = torch.cat([pooled, dep_emb], dim=-1)
                pooled_irab = base_pred.model.dep_proj(h_aug)
            else:
                pooled_irab = pooled
            base_case_logits = base_pred.model.case_head(pooled_irab)
            base_role_logits = base_pred.model.role_head(pooled_irab)
            base_marker_logits = base_pred.model.marker_head(pooled_irab)

        words = enc["words"]
        base_case_probs = F.softmax(base_case_logits, dim=-1)
        base_role_probs = F.softmax(base_role_logits, dim=-1)
        base_marker_probs = F.softmax(base_marker_logits, dim=-1)
        base_case_conf, base_case_pred = base_case_probs[0].max(dim=-1)
        base_role_conf, base_role_pred = base_role_probs[0].max(dim=-1)
        base_marker_conf, base_marker_pred = base_marker_probs[0].max(dim=-1)

        # ---- Construction detection (uses initial role argmax) ----
        record_for_detection = {
            "sentence": " ".join(words),
            "items": [{"word": w, "role": ID_TO_ROLE.get(int(base_role_pred[i].item()), "")}
                       for i, w in enumerate(words)],
            "source": "_query",
        }
        constructions = detect_constructions_in_record(record_for_detection)
        kana_spans = [c for c in constructions if c["construction"] == "kana_sisters"]
        if not kana_spans:
            continue

        # Match gold by surface
        gold_by_norm = {_norm_ar(g["word"]): g for g in gold_items}

        for span_desc in kana_spans:
            n_kana_spans += 1
            start, end = span_desc["span"]
            start = max(0, min(start, len(words)))
            end = max(start, min(end, len(words)))
            if end <= start:
                continue
            span_len = end - start
            particle = span_desc["particle_surface"]
            particle_group = span_desc["particle_group"]

            # ---- Retrieve ----
            span_emb = pooled_irab[0, start:end].mean(dim=0).cpu().numpy()
            query = build_signature(record_for_detection, span_desc, sentence_idx=-1)
            hits = memory.retrieve(
                query=query, query_embedding=span_emb,
                k=args.retrieval_k, alpha=0.3, require_particle_group=True,
            )
            if hits:
                n_with_retrievals += 1

            # ---- Run reasoner (capture transformation candidates) ----
            ro = reasoner.reason(
                query_span=[], retrieved=hits, query_words=words,
                span=(start, end),
                particle_group=particle_group, particle_surface=particle,
            )

            # Determine tier
            if not ro.valid:
                tier = "no_retrievals"
                n_fallback += 1
            elif ro.confidence >= args.tau_high:
                tier = "override"; n_override += 1
            elif ro.confidence >= args.tau_med:
                tier = "strong_bias"; n_strong_bias += 1
            else:
                tier = "fallback"; n_fallback += 1

            # ---- Apply override on a COPY of logits to capture after_override ----
            after_case_logits = base_case_logits.clone()
            after_role_logits = base_role_logits.clone()
            after_marker_logits = base_marker_logits.clone()
            if tier == "override":
                for i in range(span_len):
                    word_idx = start + i
                    if word_idx >= len(words):
                        break
                    pred = ro.predicted[i]; cons = ro.consensus_per_pos[i]
                    if pred.get("case") and cons.get("case_rate", 0) >= args.tau_high:
                        cid = CASE_TO_ID.get(pred["case"], -1)
                        if cid >= 0:
                            row = torch.full((len(CASE_LABELS),), -8.0,
                                              dtype=after_case_logits.dtype,
                                              device=after_case_logits.device)
                            row[cid] = 8.0
                            after_case_logits[0, word_idx, :] = row
                    if pred.get("role") and cons.get("role_rate", 0) >= args.tau_high:
                        rid = ROLE_TO_ID.get(pred["role"], -1)
                        if rid >= 0:
                            row = torch.full((len(ROLE_LABELS),), -8.0,
                                              dtype=after_role_logits.dtype,
                                              device=after_role_logits.device)
                            row[rid] = 8.0
                            after_role_logits[0, word_idx, :] = row
                    if pred.get("marker") and cons.get("marker_rate", 0) >= args.tau_high:
                        mid = MARKER_TO_ID.get(pred["marker"], -1)
                        if mid >= 0:
                            row = torch.full((len(MARKER_LABELS),), -8.0,
                                              dtype=after_marker_logits.dtype,
                                              device=after_marker_logits.device)
                            row[mid] = 8.0
                            after_marker_logits[0, word_idx, :] = row
            elif tier == "strong_bias":
                import math
                for i in range(span_len):
                    word_idx = start + i
                    if word_idx >= len(words):
                        break
                    pred = ro.predicted[i]; cons = ro.consensus_per_pos[i]
                    if pred.get("case"):
                        cid = CASE_TO_ID.get(pred["case"], -1)
                        if cid >= 0:
                            after_case_logits[0, word_idx, cid] += args.lambda_strong * math.log(cons.get("case_rate", 0.0) + 1e-3)
                    if pred.get("role"):
                        rid = ROLE_TO_ID.get(pred["role"], -1)
                        if rid >= 0:
                            after_role_logits[0, word_idx, rid] += args.lambda_strong * math.log(cons.get("role_rate", 0.0) + 1e-3)
                    if pred.get("marker"):
                        mid = MARKER_TO_ID.get(pred["marker"], -1)
                        if mid >= 0:
                            after_marker_logits[0, word_idx, mid] += args.lambda_strong * math.log(cons.get("marker_rate", 0.0) + 1e-3)

            after_case_probs = F.softmax(after_case_logits, dim=-1)
            after_role_probs = F.softmax(after_role_logits, dim=-1)
            after_marker_probs = F.softmax(after_marker_logits, dim=-1)
            after_case_conf, after_case_pred = after_case_probs[0].max(dim=-1)
            after_role_conf, after_role_pred = after_role_probs[0].max(dim=-1)
            after_marker_conf, after_marker_pred = after_marker_probs[0].max(dim=-1)

            # ---- Build per-span record ----
            before_per_word = []
            after_per_word = []
            gold_per_word = []
            calib_per_word = []
            transformation_candidates = []
            consensus_per_pos_dump = []
            dep_deltas = []
            alignment_dump = []

            for i in range(span_len):
                word_idx = start + i
                if word_idx >= len(words):
                    break
                w = words[word_idx]
                # Before/after labels
                bcase = ID_TO_CASE.get(int(base_case_pred[word_idx].item()), "")
                brole = ID_TO_ROLE.get(int(base_role_pred[word_idx].item()), "")
                bmarker = ID_TO_MARKER.get(int(base_marker_pred[word_idx].item()), "")
                acase = ID_TO_CASE.get(int(after_case_pred[word_idx].item()), "")
                arole = ID_TO_ROLE.get(int(after_role_pred[word_idx].item()), "")
                amarker = ID_TO_MARKER.get(int(after_marker_pred[word_idx].item()), "")
                # Gold
                gw = gold_by_norm.get(_norm_ar(w), {})
                gcase = gw.get("case"); grole = gw.get("role"); gmarker = gw.get("marker")
                # Confidence
                bcase_c = float(base_case_conf[word_idx].item())
                brole_c = float(base_role_conf[word_idx].item())
                bmarker_c = float(base_marker_conf[word_idx].item())
                acase_c = float(after_case_conf[word_idx].item())
                arole_c = float(after_role_conf[word_idx].item())
                amarker_c = float(after_marker_conf[word_idx].item())

                before_per_word.append({"word": w, "case": bcase, "role": brole, "marker": bmarker,
                                          "case_conf": round(bcase_c, 3), "role_conf": round(brole_c, 3),
                                          "marker_conf": round(bmarker_c, 3)})
                after_per_word.append({"word": w, "case": acase, "role": arole, "marker": amarker,
                                         "case_conf": round(acase_c, 3), "role_conf": round(arole_c, 3),
                                         "marker_conf": round(amarker_c, 3)})
                gold_per_word.append({"word": w, "case": gcase, "role": grole, "marker": gmarker})
                calib_per_word.append({
                    "word": w,
                    "case_correct_before": (bcase == gcase) if gcase else None,
                    "case_correct_after": (acase == gcase) if gcase else None,
                    "role_correct_before": (brole == grole) if grole else None,
                    "role_correct_after": (arole == grole) if grole else None,
                    "marker_correct_before": (bmarker == gmarker) if gmarker else None,
                    "marker_correct_after": (amarker == gmarker) if gmarker else None,
                })

                # Transformation candidates per position (raw vote distribution)
                case_v, role_v, marker_v, n_voters = _vote_on_position(i, hits)
                transformation_candidates.append({
                    "position": i,
                    "n_voters": n_voters,
                    "case_votes": dict(case_v),
                    "role_votes": dict(role_v),
                    "marker_votes": dict(marker_v),
                })
                consensus_per_pos_dump.append({
                    "position": i,
                    "case_rate": round(ro.consensus_per_pos[i].get("case_rate", 0), 3) if ro.valid else None,
                    "role_rate": round(ro.consensus_per_pos[i].get("role_rate", 0), 3) if ro.valid else None,
                    "marker_rate": round(ro.consensus_per_pos[i].get("marker_rate", 0), 3) if ro.valid else None,
                })

                # Aggregate per-particle stats (only if gold is present)
                pp = per_particle[particle]
                pp["n_words"] += 1
                if gcase is not None:
                    if bcase == gcase: pp["case_correct_before"] += 1
                    if acase == gcase: pp["case_correct_after"] += 1
                if grole is not None:
                    if brole == grole: pp["role_correct_before"] += 1
                    if arole == grole: pp["role_correct_after"] += 1
                if gmarker is not None:
                    if bmarker == gmarker: pp["marker_correct_before"] += 1
                    if amarker == gmarker: pp["marker_correct_after"] += 1
                if (gcase and grole and gmarker):
                    if bcase == gcase and brole == grole and bmarker == gmarker:
                        pp["fully_correct_before"] += 1
                    if acase == gcase and arole == grole and amarker == gmarker:
                        pp["fully_correct_after"] += 1
                # Wrong-high-confidence: predicted with > 0.8 conf and wrong on case
                wrong_high = (gcase is not None) and (acase != gcase) and (acase_c > 0.8)
                if wrong_high:
                    pp["wrong_high_conf"] += 1
                    wrong_high_conf_dump.append({
                        "sentence": sent[:120], "word": w,
                        "particle": particle,
                        "predicted_case": acase, "gold_case": gcase, "case_conf": round(acase_c, 3),
                        "predicted_role": arole, "gold_role": grole,
                    })

                # Per-position before/after outcome breakdown
                if gcase is not None:
                    if bcase == gcase and acase == gcase:
                        pos_outcome[("case", i)]["both_correct"] += 1
                    elif bcase != gcase and acase == gcase:
                        pos_outcome[("case", i)]["override_fixed"] += 1
                    elif bcase == gcase and acase != gcase:
                        pos_outcome[("case", i)]["override_broke"] += 1
                    else:
                        pos_outcome[("case", i)]["both_wrong"] += 1

            # Dep deltas: per retrieved hit, compare deprel/governor_upos at each position
            for h in hits:
                hit_dep = []
                for i in range(min(span_len, len(h.instance.items))):
                    item = h.instance.items[i]
                    hit_dep.append({
                        "position": i,
                        "retrieved_word": item.get("word", ""),
                        "retrieved_deprel": item.get("deprel", "<unk>"),
                        "retrieved_gov_upos": item.get("governor_upos", ""),
                        "retrieved_case": item.get("case", ""),
                        "retrieved_role": item.get("role", ""),
                        "retrieved_marker": item.get("marker", ""),
                    })
                dep_deltas.append({
                    "instance_id": h.instance.instance_id,
                    "retrieved_sentence": h.instance.sentence[:90],
                    "cosine": round(h.cosine, 3),
                    "sym_overlap": round(h.sym_overlap, 3),
                    "score": round(h.score, 3),
                    "head_morph": h.instance.head_morph,
                    "head_deprel": h.instance.head_deprel,
                    "head_governor_upos": h.instance.head_governor_upos,
                    "per_position": hit_dep,
                })

                # Alignment dump: surface position; we don't have query dep tree at inference
                alignment_dump.append({
                    "instance_id": h.instance.instance_id,
                    "alignment": [(i, i) for i in range(min(span_len, len(h.instance.items)))],
                    "alignment_method": "surface_position",
                    "note": "Query dep tree NOT available at inference (Stanza not run); only surface alignment is feasible",
                })

            per_particle[particle]["n_spans"] += 1

            # Final transformation: consensus labels (if valid)
            final_transformation = ro.predicted if ro.valid else None

            record = {
                # 1. sentence
                "sentence": sent,
                # 2. trigger
                "trigger": particle,
                "particle_group": particle_group,
                # 3. span
                "span": [start, end],
                # 4. retrieved
                "retrieved": [
                    {"instance_id": h.instance.instance_id,
                     "sentence": h.instance.sentence[:120],
                     "cosine": round(h.cosine, 3),
                     "sym_overlap": round(h.sym_overlap, 3),
                     "score": round(h.score, 3),
                     "items": h.instance.items}
                    for h in hits
                ],
                # 5. retrieved_deps
                "retrieved_deps": dep_deltas,
                # 6. alignment
                "alignment": alignment_dump,
                # 7. dep_deltas — already covered by retrieved_deps with per-position deprel
                # 8. transformation_candidates
                "transformation_candidates": transformation_candidates,
                # 9. consensus_per_pos
                "consensus_per_pos": consensus_per_pos_dump,
                # 10. final_transformation
                "final_transformation": final_transformation,
                # 11. tier_fired
                "tier_fired": tier,
                "reasoner_confidence": round(ro.confidence, 3) if ro.valid else None,
                "reasoner_consensus_rate": round(ro.consensus_rate, 3) if ro.valid else None,
                "n_hits": len(hits),
                "rule": ro.rule,
                # 12. before_override
                "before_override": before_per_word,
                # 13. after_override
                "after_override": after_per_word,
                # 14. gold
                "gold": gold_per_word,
                # 15. calibration
                "calibration": calib_per_word,
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_fh.close()
    print(f"\nDumped {n_kana_spans} kana spans to {out_path}")

    # ---- Summary report ----
    md = []
    md.append(f"# KanaReasoner failure debug summary — {args.eval}\n")
    md.append(f"- Total kana spans detected: {n_kana_spans}")
    md.append(f"- Spans with ≥1 retrieval: {n_with_retrievals}")
    md.append(f"- Tier fired: override={n_override}, strong_bias={n_strong_bias}, "
              f"fallback={n_fallback}\n")

    md.append("## Per-particle breakdown (gold-aligned words only)\n")
    md.append("| particle | spans | words | case_b→a | role_b→a | marker_b→a | fully_b→a | wrong_high_conf |")
    md.append("|---|---:|---:|---|---|---|---|---:|")
    for p, s in sorted(per_particle.items(), key=lambda kv: -kv[1]["n_words"]):
        if s["n_words"] == 0:
            continue
        def pct(num, denom):
            return f"{(num / max(denom, 1)) * 100:.0f}"
        md.append(
            f"| {p} | {s['n_spans']} | {s['n_words']} | "
            f"{pct(s['case_correct_before'], s['n_words'])}→{pct(s['case_correct_after'], s['n_words'])} | "
            f"{pct(s['role_correct_before'], s['n_words'])}→{pct(s['role_correct_after'], s['n_words'])} | "
            f"{pct(s['marker_correct_before'], s['n_words'])}→{pct(s['marker_correct_after'], s['n_words'])} | "
            f"{pct(s['fully_correct_before'], s['n_words'])}→{pct(s['fully_correct_after'], s['n_words'])} | "
            f"{s['wrong_high_conf']} |"
        )
    md.append("")

    md.append("## Per-position case outcome (override_fixed / override_broke / both_correct / both_wrong)\n")
    md.append("| position | both_correct | override_fixed | override_broke | both_wrong |")
    md.append("|---|---:|---:|---:|---:|")
    for i in range(3):
        c = pos_outcome[("case", i)]
        md.append(f"| pos {i} | {c.get('both_correct', 0)} | {c.get('override_fixed', 0)} | "
                  f"{c.get('override_broke', 0)} | {c.get('both_wrong', 0)} |")
    md.append("")

    md.append("## Wrong-high-confidence cases (case ≠ gold AND case_conf > 0.8)\n")
    md.append(f"Count: {len(wrong_high_conf_dump)}\n")
    if wrong_high_conf_dump:
        md.append("| sentence | word | particle | predicted | gold | conf |")
        md.append("|---|---|---|---|---|---:|")
        for r in wrong_high_conf_dump[:30]:
            md.append(f"| `{r['sentence'][:60]}` | {r['word']} | {r['particle']} | "
                      f"{r['predicted_case']} | {r['gold_case']} | {r['case_conf']} |")
        md.append("")

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md))
    print(f"Summary written: {md_path}")
    print(f"\n=== Top-level numbers ===")
    print(f"spans={n_kana_spans} with_retrievals={n_with_retrievals}  "
          f"override={n_override} strong_bias={n_strong_bias} fallback={n_fallback}")
    for p, s in sorted(per_particle.items(), key=lambda kv: -kv[1]["n_words"]):
        if s["n_words"] == 0: continue
        def pct(num, d): return (num / max(d, 1)) * 100
        print(f"  {p:>10}: spans={s['n_spans']} words={s['n_words']} | "
              f"case {pct(s['case_correct_before'], s['n_words']):.0f}→{pct(s['case_correct_after'], s['n_words']):.0f} | "
              f"role {pct(s['role_correct_before'], s['n_words']):.0f}→{pct(s['role_correct_after'], s['n_words']):.0f} | "
              f"fully {pct(s['fully_correct_before'], s['n_words']):.0f}→{pct(s['fully_correct_after'], s['n_words']):.0f} | "
              f"wrong_high_conf={s['wrong_high_conf']}")


if __name__ == "__main__":
    main()
