"""Step-3 of the supervision phase — build hard-case eval subsets.

Slices the held-out corpora into structural difficulty buckets and
writes one ``all.jsonl`` per bucket under ``data_v2/hard_eval/``.

Buckets (each a strict superset filter on schema_v2 metadata):

  long_range/             — sentence has any head_distance ≥ 5
  nested_clause/          — clause_depth ≥ 2
  overlap/                — ≥ 2 constructions share a token
  ambiguity/              — semantic_pressure_score ≥ 2
  quranic_hard/           — masaq_quranic + (semantic_pressure ≥ 2 OR
                              construction overlap)
  rare_constructions/     — at least one construction whose family
                              appears < 10× in the source corpus
  multi_clause/           — clause_depth ≥ 1 AND token count ≥ 15
  attachment_ambiguity/   — ambiguity_score ≥ 0.3 AND
                              dependency_depth ≥ 4

Each output sentence is unchanged (full schema_v2 jsonl); only the
*partition* changes. We do NOT re-annotate — the bucketing is a pure
selection over existing metadata.

Plus: ``data_v2/hard_eval/summary.json`` with sizes per bucket.
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


def _load(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open()]


def _has_long_range(d: dict) -> bool:
    tokens = d.get("tokens", [])
    n = len(tokens)
    for i, t in enumerate(tokens):
        h = t.get("dep_head_idx")
        if h is not None and h >= 0 and h < n and abs(h - i) >= 5:
            return True
    return False


def _clause_depth(d: dict) -> int:
    return int(d.get("curriculum", {}).get("clause_depth", 0) or 0)


def _semantic_pressure(d: dict) -> int:
    return int(d.get("curriculum", {}).get("semantic_pressure_score", 0) or 0)


def _ambiguity_score(d: dict) -> float:
    return float(d.get("curriculum", {}).get("ambiguity_score", 0.0) or 0.0)


def _dep_depth(d: dict) -> int:
    return int(d.get("curriculum", {}).get("dependency_depth", 0) or 0)


def _has_overlap(d: dict) -> bool:
    occ: Counter = Counter()
    for c in d.get("constructions", []):
        for t in c.get("token_indices", []):
            occ[t] += 1
    return any(c >= 2 for c in occ.values())


def _is_quranic(d: dict) -> bool:
    return d.get("metadata", {}).get("source") == "masaq_quranic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(ROOT / "data_v2" / "annotated"))
    ap.add_argument("--out_dir", default=str(ROOT / "data_v2" / "hard_eval"))
    ap.add_argument("--sources", nargs="+",
                    default=["gazelle_test", "masaq_quranic", "ud_padt_test"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences: List[dict] = []
    for src in args.sources:
        p = Path(args.data_root) / src / "all.jsonl"
        sentences.extend(_load(p))
    print(f"loaded {len(sentences)} sentences across {args.sources}")

    # Family rarity over the held-out pool
    fam_counts: Counter = Counter()
    for d in sentences:
        for c in d.get("constructions", []):
            fam_counts[c.get("family", "")] += 1
    rare_families = {f for f, c in fam_counts.items() if c < 10}

    buckets: Dict[str, List[dict]] = defaultdict(list)
    for d in sentences:
        if _has_long_range(d):
            buckets["long_range"].append(d)
        if _clause_depth(d) >= 2:
            buckets["nested_clause"].append(d)
        if _has_overlap(d):
            buckets["overlap"].append(d)
        if _semantic_pressure(d) >= 2:
            buckets["ambiguity"].append(d)
        if _is_quranic(d) and (_semantic_pressure(d) >= 2 or _has_overlap(d)):
            buckets["quranic_hard"].append(d)
        for c in d.get("constructions", []):
            if c.get("family", "") in rare_families:
                buckets["rare_constructions"].append(d)
                break
        n_tok = len(d.get("tokens", []))
        if _clause_depth(d) >= 1 and n_tok >= 15:
            buckets["multi_clause"].append(d)
        if _ambiguity_score(d) >= 0.3 and _dep_depth(d) >= 4:
            buckets["attachment_ambiguity"].append(d)

    # Write
    summary: Dict[str, int] = {}
    for name, items in buckets.items():
        d = out_dir / name
        d.mkdir(exist_ok=True)
        with (d / "all.jsonl").open("w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        summary[name] = len(items)
        print(f"  {name:25} {len(items)}")

    (out_dir / "summary.json").write_text(
        json.dumps({
            "sizes": summary,
            "rare_families": sorted(rare_families),
            "n_total_input": len(sentences),
        }, indent=2, ensure_ascii=False)
    )
    print(f"\n✓ wrote {out_dir}")


if __name__ == "__main__":
    main()
