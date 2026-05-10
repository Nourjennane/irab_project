"""Heuristic auto-annotation of mined ambiguity candidates.

This is a programmatic placeholder for the human grammarian's pass.
For each mined candidate we check whether the gold and predicted
roles share a **surface-ambiguous signature** — i.e., both readings
are grammatically possible from the surface form alone. If so, we
mark the candidate as ``BOTH_VALID`` automatically.

Surface-ambiguous role pairs (where both readings are grammatically
licit on a noun in jarr / kasra / immediately after another word):

  {mudaaf_ilayh, mafoul_bih, mubtada, fail, ism_majrur, naat,
   matuf, badal, ism_inna, khabar_inna, khabar_kana, khabar}
   (the genitive / nominative / oblique surface family)

If both gold and predicted role are in this set, the token is
treated as genuinely ambiguous and both readings are accepted by
the permissive evaluator.

This is **explicitly a heuristic** — a real grammarian would flag
many cases as not actually ambiguous (e.g., a noun governed by an
overt verb is unambiguously *mafoul_bih*, even though the surface
matches *mudaaf_ilayh*'s signature). Documented as a placeholder
that should be replaced by human pass.

After auto-annotation, ``eval_v3.evaluate_with_ambiguity`` is run
and the strict-vs-permissive deltas are reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


# Surface-ambiguous role family — predictions in this set are treated
# as legitimate alternatives when the gold is in the same family.
SURFACE_AMBIGUOUS_FAMILY: Set[str] = {
    "mudaaf_ilayh", "mafoul_bih", "mubtada", "fail",
    "ism_majrur", "naat", "matuf", "badal",
    "ism_inna", "khabar_inna", "khabar_kana", "khabar",
}


def _load_model(ckpt_dir: Path, encoder_name: str, device):
    import torch
    from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel
    sd = torch.load(ckpt_dir / "pytorch_model.bin", map_location="cpu",
                    weights_only=True)
    model = DepAwareStructuredModel(
        encoder_name=encoder_name,
        enable_morph_heads=True, morph_heads_enabled=None,
        enable_dep_features=True,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device); model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs" / "validated_nextgen_recovery"))
    ap.add_argument("--ambiguity_root",
                    default=str(ROOT / "data_v2" / "ambiguity_corpus"))
    ap.add_argument("--out_dir", default=str(ROOT / "docs" / "permissive_eval"))
    ap.add_argument("--encoder_name", default="UBC-NLP/AraT5v2-base-1024")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--datasets", nargs="+",
                    default=["gazelle_test", "masaq_quranic"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("USE_TF", "NO")
    import torch
    from transformers import AutoTokenizer
    from irab_tashkeel.data_v2.schema_v2 import read_jsonl
    from irab_tashkeel.training_v2.eval_hook import predict_for_eval
    from irab_tashkeel.eval_v2 import (
        aggregate_outcomes, extract_outcomes,
    )
    from irab_tashkeel.ambiguity.schema import (
        AmbiguityExample, AmbiguityKind,
    )
    from irab_tashkeel.eval_v3 import evaluate_with_ambiguity

    # ---------- 1. Auto-annotate the queue ----------
    # Heuristic: a candidate is genuinely ambiguous on the role axis if
    # gold and predicted role both belong to the surface-ambiguous family.
    # Crucially, we strip case and marker from the secondary analyses,
    # because (a) the model often gets case/marker wrong together with
    # role (so requiring all three to match would defeat the purpose),
    # and (b) "mark this token as having multiple valid role readings"
    # is the actual semantic claim a grammarian would make. The case
    # and marker remain governed by whichever role the model picks.
    from irab_tashkeel.ambiguity.schema import TokenAnalysis
    n_total = 0
    n_kept = 0
    annotations_per_sentence: Dict[str, List[AmbiguityExample]] = {}
    for kind_dir in Path(args.ambiguity_root).glob("*/"):
        queue = kind_dir / "queue.jsonl"
        if not queue.exists():
            continue
        for line in queue.open():
            d = json.loads(line)
            ex = AmbiguityExample.from_dict(d)
            n_total += 1
            keep = False
            new_secondaries: List[Dict[int, TokenAnalysis]] = []
            for tok_idx, prim in ex.primary_analysis.items():
                if prim.role not in SURFACE_AMBIGUOUS_FAMILY:
                    continue
                for alt in ex.secondary_analyses:
                    a = alt.get(tok_idx)
                    if a is None or a.role is None:
                        continue
                    if a.role in SURFACE_AMBIGUOUS_FAMILY and a.role != prim.role:
                        # Strip case + marker so the permissive matcher
                        # only checks role.
                        new_secondaries.append({tok_idx: TokenAnalysis(
                            case=None, role=a.role, marker=None,
                            governor_token=a.governor_token,
                        )})
                        keep = True
            if keep:
                # Replace the (mis-typed) gold-style secondaries with
                # the role-only variants.
                ex.secondary_analyses = new_secondaries
                ex.confidence = 1.0
                ex.annotator_id = "heuristic_surface_family"
                annotations_per_sentence.setdefault(ex.sentence_id, []).append(ex)
                n_kept += 1

    print(f"\nAuto-annotation:")
    print(f"  total mined candidates: {n_total}")
    print(f"  kept as BOTH_VALID:     {n_kept} ({100*n_kept/max(n_total,1):.1f}%)")
    print(f"  spans across {len(annotations_per_sentence)} unique sentences")

    # ---------- 2. Run model + permissive eval ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint
        if (Path(args.checkpoint) / "tokenizer.json").exists()
        else args.encoder_name,
    )
    model = _load_model(Path(args.checkpoint), args.encoder_name, device)

    sentences = []
    for ds in args.datasets:
        p = ROOT / "data_v2" / "annotated" / ds / "all.jsonl"
        if p.exists():
            sentences.extend(list(read_jsonl(str(p))))
    print(f"\nEval set: {len(sentences)} sentences")

    print("Running predictions...")
    preds = predict_for_eval(model, tokenizer, sentences,
                              batch_size=args.batch_size, device=device)

    # Strict baseline
    outcomes = extract_outcomes(sentences, preds)
    strict = aggregate_outcomes([o for o in outcomes if o.is_fully_observable])

    # Permissive
    perm = evaluate_with_ambiguity(sentences, preds, annotations_per_sentence)

    # ---------- 3. Write report ----------
    report = {
        "auto_annotation": {
            "total_mined":       n_total,
            "kept_as_both_valid": n_kept,
            "kept_pct":          round(100 * n_kept / max(n_total, 1), 2),
            "n_unique_sentences": len(annotations_per_sentence),
            "heuristic":         "surface_role_family",
            "family":            sorted(SURFACE_AMBIGUOUS_FAMILY),
        },
        "strict": {
            "n_words":     strict["n_words"],
            "case_acc":    strict["case_acc"],
            "role_acc":    strict["role_acc"],
            "marker_em":   strict["marker_em"],
            "fully":       strict["fully"],
            "calib_gap":   strict["calib_gap"],
        },
        "permissive": {
            "n_total":                perm.n_total,
            "n_strict_correct":       perm.n_strict_correct,
            "n_permissive_correct":   perm.n_permissive_correct,
            "n_ambiguous_tokens":     perm.n_ambiguous_tokens,
            "n_ambiguous_resolved":   perm.n_ambiguous_resolved,
            "strict_fully":           round(perm.strict_fully, 4),
            "permissive_fully":       round(perm.permissive_fully, 4),
            "ambiguity_resolved_acc": round(perm.ambiguity_resolved_accuracy, 4),
            "delta":                  round(perm.permissive_fully - perm.strict_fully, 4),
        },
    }
    (out_dir / "permissive_eval.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    # Markdown
    lines = ["# Permissive Evaluation — Heuristic Auto-Annotation", "",
             "## Methodology", "",
             f"- Mined ambiguity candidates: **{n_total}** across "
             "7 ambiguity kinds (see `data_v2/ambiguity_corpus/`)",
             f"- Auto-marked as `BOTH_VALID` if gold and predicted role "
             f"are both in the **surface-ambiguous role family**: "
             f"{sorted(SURFACE_AMBIGUOUS_FAMILY)}",
             f"- Kept: **{n_kept} candidates** "
             f"({100*n_kept/max(n_total,1):.1f}% of mined; "
             f"{len(annotations_per_sentence)} unique sentences)",
             "",
             "**Caveat.** This is a heuristic placeholder for the human "
             "grammarian's pass. A real annotator would reject many of "
             "these — e.g., when an overt verb governs the noun, the "
             "*mafoul_bih* reading is unambiguous despite surface "
             "compatibility with *mudaaf_ilayh*. Treat the permissive "
             "delta below as an upper bound.",
             "",
             "## Strict baseline (no permissive scoring)",
             "",
             "| metric | value |",
             "|---|---:|",
             f"| n_words (fully-observable) | {strict['n_words']} |",
             f"| case_acc | {strict['case_acc']} |",
             f"| role_acc | {strict['role_acc']} |",
             f"| marker_em | {strict['marker_em']} |",
             f"| **fully** | **{strict['fully']}** |",
             f"| calib_gap | {strict['calib_gap']} |",
             "",
             "## Permissive eval",
             "",
             "| metric | value |",
             "|---|---:|",
             f"| total tokens | {perm.n_total} |",
             f"| strict-correct | {perm.n_strict_correct} |",
             f"| permissive-correct | {perm.n_permissive_correct} |",
             f"| tokens flagged ambiguous | {perm.n_ambiguous_tokens} |",
             f"| ambiguous tokens resolved | {perm.n_ambiguous_resolved} |",
             f"| **strict_fully** | **{perm.strict_fully:.4f}** |",
             f"| **permissive_fully** | **{perm.permissive_fully:.4f}** |",
             f"| Δ (permissive − strict) | **{perm.permissive_fully - perm.strict_fully:+.4f}** |",
             f"| ambiguity_resolved_acc | {perm.ambiguity_resolved_accuracy:.4f} |",
             ""]
    (out_dir / "permissive_eval_report.md").write_text("\n".join(lines))
    print(f"\n✓ wrote {out_dir}/permissive_eval_report.md")
    print(f"  strict_fully     = {perm.strict_fully:.4f}")
    print(f"  permissive_fully = {perm.permissive_fully:.4f}")
    print(f"  Δ (perm − strict) = {perm.permissive_fully - perm.strict_fully:+.4f}")


if __name__ == "__main__":
    main()
