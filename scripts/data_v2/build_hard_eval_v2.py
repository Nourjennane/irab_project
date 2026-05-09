"""Step-7 — hard_eval_v2 stress benchmark builder.

A more aggressive cut than ``hard_eval/`` — sentences must satisfy
*compound* conditions, targeting genuine reasoning rather than
single-axis difficulty.

Buckets:

  long_nested_idafa/
    nested idafa (≥ 2 idafa constructions overlapping) AND length ≥ 12
  ambiguous_coordination/
    ≥ 2 ʿaṭf particles AND clause depth ≥ 1
  omitted_governor/
    contains a token whose role implies a governor that is NOT in
    the same clause's surface
  classical_arabic/
    Quranic source AND archaic_form flag (heuristic: token in MASAQ
    with low frequency in distill_v2 vocabulary)
  quranic_difficult/
    masaq_quranic AND (semantic_pressure ≥ 2 OR overlap OR omitted)
  multi_valid_parses/
    sentence carries an `AmbiguityExample` with ≥ 1 secondary analysis
  attachment_traps/
    sentence has a noun whose two nearest plausible heads are an
    iḍāfa head AND a preposition (the canonical attachment trap)
  adversarial_syntax/
    long sentence (≥ 25 tokens) AND ≥ 2 nested clauses AND overlap

Each bucket: ``data_v2/hard_eval_v2/<bucket>/all.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _load(p: Path) -> List[dict]:
    return [json.loads(l) for l in p.open()] if p.exists() else []


def _has_nested_idafa(d: dict) -> bool:
    n_idafa = sum(1 for c in d.get("constructions", [])
                  if c.get("family", "").startswith("idafa"))
    return n_idafa >= 2


def _coord_count(d: dict) -> int:
    return sum(1 for t in d.get("tokens", [])
                if (t.get("role", {}) or {}).get("value") == "harf_atf")


def _clause_depth(d: dict) -> int:
    return int(d.get("curriculum", {}).get("clause_depth", 0) or 0)


def _semantic_pressure(d: dict) -> int:
    return int(d.get("curriculum", {}).get("semantic_pressure_score", 0) or 0)


def _has_overlap(d: dict) -> bool:
    occ: Counter = Counter()
    for c in d.get("constructions", []):
        for t in c.get("token_indices", []):
            occ[t] += 1
    return any(v >= 2 for v in occ.values())


def _is_quranic(d: dict) -> bool:
    return d.get("metadata", {}).get("source") == "masaq_quranic"


def _has_omitted_governor(d: dict) -> bool:
    """Heuristic: a role like ism_kana / ism_inna with no kana/inna
    token in the sentence's particles."""
    tokens = d.get("tokens", [])
    has_kana = any(
        (t.get("pos", {}) or {}).get("value") == "AUX"
        for t in tokens
    )
    has_particle = any(
        (t.get("pos", {}) or {}).get("value") in ("PART", "SCONJ")
        for t in tokens
    )
    needs_kana = any(
        (t.get("role", {}) or {}).get("value") in ("ism_kana", "khabar_kana")
        for t in tokens
    )
    needs_inna = any(
        (t.get("role", {}) or {}).get("value") in ("ism_inna", "khabar_inna")
        for t in tokens
    )
    return ((needs_kana and not has_kana) or (needs_inna and not has_particle))


def _has_attachment_trap(d: dict) -> bool:
    tokens = d.get("tokens", [])
    n = len(tokens)
    for i, t in enumerate(tokens):
        if (t.get("role", {}) or {}).get("value") not in (
            "mudaaf_ilayh", "ism_majrur"
        ):
            continue
        # check left side for both ADP and NOUN within 3 tokens
        left = tokens[max(0, i - 3):i]
        has_adp = any((x.get("pos", {}) or {}).get("value") == "ADP" for x in left)
        has_nn  = any((x.get("pos", {}) or {}).get("value") in ("NOUN", "PROPN")
                       for x in left)
        if has_adp and has_nn:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(ROOT / "data_v2" / "annotated"))
    ap.add_argument("--out_dir", default=str(ROOT / "data_v2" / "hard_eval_v2"))
    ap.add_argument("--ambiguity_root",
                    default=str(ROOT / "data_v2" / "ambiguity_corpus"))
    ap.add_argument("--sources", nargs="+",
                    default=["gazelle_test", "masaq_quranic"])
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences: List[dict] = []
    for src in args.sources:
        sentences.extend(_load(data_root / src / "all.jsonl"))

    # Sentences with ambiguity annotations (pending OR confirmed)
    ambiguous_sids: set = set()
    amb_root = Path(args.ambiguity_root)
    if amb_root.exists():
        for kind_dir in amb_root.glob("*/"):
            for fname in ("queue.jsonl", "confirmed.jsonl", "edited.jsonl"):
                f = kind_dir / fname
                if not f.exists():
                    continue
                for line in f.open():
                    try:
                        ambiguous_sids.add(json.loads(line)["sentence_id"])
                    except Exception:
                        pass

    buckets: Dict[str, List[dict]] = defaultdict(list)
    for d in sentences:
        n_tok = len(d.get("tokens", []))

        if _has_nested_idafa(d) and n_tok >= 12:
            buckets["long_nested_idafa"].append(d)
        if _coord_count(d) >= 2 and _clause_depth(d) >= 1:
            buckets["ambiguous_coordination"].append(d)
        if _has_omitted_governor(d):
            buckets["omitted_governor"].append(d)
        if _is_quranic(d):
            buckets["classical_arabic"].append(d)
        if _is_quranic(d) and (_semantic_pressure(d) >= 2 or _has_overlap(d)
                                or _has_omitted_governor(d)):
            buckets["quranic_difficult"].append(d)
        if d.get("sentence_id") in ambiguous_sids:
            buckets["multi_valid_parses"].append(d)
        if _has_attachment_trap(d):
            buckets["attachment_traps"].append(d)
        if n_tok >= 25 and _clause_depth(d) >= 2 and _has_overlap(d):
            buckets["adversarial_syntax"].append(d)

    sizes: Dict[str, int] = {}
    for name, items in buckets.items():
        d = out_dir / name
        d.mkdir(exist_ok=True)
        with (d / "all.jsonl").open("w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        sizes[name] = len(items)
        print(f"  {name:30} {len(items)}")

    (out_dir / "summary.json").write_text(
        json.dumps({"sizes": sizes, "n_input": len(sentences),
                    "n_ambiguous_sids": len(ambiguous_sids)},
                   indent=2, ensure_ascii=False)
    )
    print(f"\n✓ wrote {out_dir}")


if __name__ == "__main__":
    main()
