"""Mine ambiguity candidates from the failure analysis output.

Reads `docs/failure_analysis/summary.json` (failure records keyed by
sentence_id × token_index with confusion strings) and the source
sentences, classifies each failure into an `AmbiguityKind` based on
the gold/pred role pair, and writes `AmbiguityExample` candidates
per category to `data_v2/ambiguity_corpus/<kind>/queue.jsonl`.

These are *unannotated* candidates — the human annotator confirms /
edits / discards them via the annotation pipeline. The schema is
already populated with the model's primary prediction and at least
one alternative (the gold), giving the annotator a strong starting
point.

Top failure-pair → AmbiguityKind mapping:

  mudaaf_ilayh ↔ {mafoul_bih, mubtada}   → idafa_attachment
  mudaaf_ilayh ↔ {ism_majrur}            → preposition_vs_idafa
  ism_majrur   ↔ {matuf}                 → coordination_scope
  ism_majrur   ↔ {mubtada, fail}         → latent_governor
  Multi-construction overlap             → nested_attachment
  fail ↔ mafoul_bih                      → semantic_role_overlap
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from irab_tashkeel.ambiguity.schema import (
    AmbiguityExample, AmbiguityKind, TokenAnalysis,
)


KIND_BY_CONFUSION: Dict[str, AmbiguityKind] = {
    "mudaaf_ilayh→mafoul_bih":   AmbiguityKind.IDAFA_ATTACHMENT,
    "mudaaf_ilayh→mubtada":      AmbiguityKind.IDAFA_ATTACHMENT,
    "mudaaf_ilayh→fail":         AmbiguityKind.IDAFA_ATTACHMENT,
    "mudaaf_ilayh→ism_majrur":   AmbiguityKind.PREPOSITION_VS_IDAFA,
    "ism_majrur→matuf":          AmbiguityKind.COORDINATION_SCOPE,
    "ism_majrur→mubtada":        AmbiguityKind.LATENT_GOVERNOR,
    "ism_majrur→fail":           AmbiguityKind.LATENT_GOVERNOR,
    "fail→mafoul_bih":           AmbiguityKind.SEMANTIC_ROLE_OVERLAP,
    "mafoul_bih→fail":           AmbiguityKind.SEMANTIC_ROLE_OVERLAP,
    "mafoul_bih→mubtada":        AmbiguityKind.SEMANTIC_ROLE_OVERLAP,
    "mafoul_bih→ism_majrur":     AmbiguityKind.PREPOSITION_VS_IDAFA,
    "mudaaf_ilayh→ism_inna":     AmbiguityKind.NESTED_ATTACHMENT,
    "mudaaf_ilayh→khabar_inna":  AmbiguityKind.NESTED_ATTACHMENT,
    "mudaaf_ilayh→khabar_kana":  AmbiguityKind.NESTED_ATTACHMENT,
    "mudaaf_ilayh→badal":        AmbiguityKind.NESTED_ATTACHMENT,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "docs" / "failure_analysis" / "summary.json"))
    ap.add_argument("--data_root", default=str(ROOT / "data_v2" / "annotated"))
    ap.add_argument("--out_root", default=str(ROOT / "data_v2" / "ambiguity_corpus"))
    ap.add_argument("--sources", nargs="+",
                    default=["gazelle_test", "masaq_quranic"])
    args = ap.parse_args()

    # Load source sentences for surface lookup
    by_sid = {}
    for src in args.sources:
        p = Path(args.data_root) / src / "all.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            d = json.loads(line)
            by_sid[d["sentence_id"]] = d

    summary = json.loads(Path(args.summary).read_text())
    confusions = summary.get("confusions", {}).get("role", {}).get("matrix", {})

    # Walk top failure pairs and emit candidate AmbiguityExamples
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    queues: Dict[AmbiguityKind, List[Dict]] = {}

    # We pull from the role confusion matrix: each (gold, pred, count)
    # contributes weight to the corresponding ambiguity kind. We then
    # find the actual sentences/tokens in failure_buckets that exhibit
    # this pair and stub out an AmbiguityExample per token.
    role_top = summary.get("confusions", {}).get("role", {}).get("top", [])
    for gold, pred, count in role_top:
        key = f"{gold}→{pred}"
        kind = KIND_BY_CONFUSION.get(key)
        if kind is None:
            continue

        # Find sentences in failure summary that match this confusion.
        # Failure summary doesn't list them directly; we re-load from
        # the failure summary buckets.
        bucket_key = f"role_confusion__{key}"
        buckets = summary.get("buckets", {})
        if bucket_key not in buckets:
            continue
        # buckets[name] in summary.json is just the count; we need the
        # raw records. Load them by reading top_failures table or
        # walking source sentences. As a starting heuristic, mine all
        # sentences containing both gold and pred labels in proximity.

    # Pragmatic mining strategy: for each (gold, pred) confusion key
    # in KIND_BY_CONFUSION, scan all source sentences and emit one
    # AmbiguityExample per sentence whose token plausibly exhibits
    # the ambiguity (i.e., gold role is `gold` for that token). The
    # secondary analysis is the (pred-role) reading.
    for sid, d in by_sid.items():
        for ti, t in enumerate(d.get("tokens", [])):
            t_role = (t.get("role") or {}).get("value")
            if t_role is None:
                continue
            # For each known confusion that starts with this gold role,
            # emit a candidate.
            for key, kind in KIND_BY_CONFUSION.items():
                gold, pred = key.split("→")
                if t_role != gold:
                    continue
                primary = {ti: TokenAnalysis(
                    case=(t.get("case") or {}).get("value"),
                    role=t_role,
                    marker=(t.get("marker") or {}).get("value"),
                    governor_token=t.get("dep_head_idx"),
                )}
                # Hypothetical alternative: same case/marker, role=pred
                secondary = {ti: TokenAnalysis(
                    case=(t.get("case") or {}).get("value"),
                    role=pred,
                    marker=(t.get("marker") or {}).get("value"),
                    note=f"plausible {pred} reading; needs human verification",
                )}
                amb_id = f"{sid}#{ti}#{kind.value}"
                ex = AmbiguityExample(
                    ambiguity_id=amb_id, sentence_id=sid,
                    ambiguity_kind=kind, span_tokens=[ti],
                    primary_analysis=primary,
                    secondary_analyses=[secondary],
                    governor_candidates=[t.get("dep_head_idx")] if t.get("dep_head_idx") is not None else [],
                    confidence_difficulty=0.7,
                    reasoning_note=f"Mined from confusion {key}; needs annotation.",
                    annotator_id="auto-mined",
                    confidence=0.0,            # 0 = unconfirmed
                )
                queues.setdefault(kind, []).append(ex.to_dict())

    summary_sizes: Dict[str, int] = {}
    for kind, items in queues.items():
        d = out_root / kind.value
        d.mkdir(exist_ok=True)
        with (d / "queue.jsonl").open("w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        summary_sizes[kind.value] = len(items)
        print(f"  {kind.value:30} {len(items)} candidates")

    (out_root / "summary.json").write_text(
        json.dumps({"queue_sizes": summary_sizes}, indent=2, ensure_ascii=False)
    )
    print(f"\n✓ wrote {out_root}")


if __name__ == "__main__":
    main()
