"""Phase 7 (recovery patch item 7) — adversarial test splits.

Build diagnostic eval splits from existing held-out sources whose
sentences exhibit specific adversarial properties:

  - construction_template      (group by canonical construction signature)
  - lexical_family             (group by content-word lemma set)
  - dependency_pattern         (group by deprel sequence)
  - clause_depth               (bucket by clause depth ≥ 2)
  - repeated_phrase_signature  (5-gram fingerprint)

The aim is to measure REAL generalization: structure-family X may
appear in train, but the *specific realization* in the eval split is
forbidden from train.

These splits are NEVER used as training input — they are pure eval
artifacts. They live under ``data_v2/adversarial_splits/<axis>/all.jsonl``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _normalise(text: str) -> str:
    text = re.sub(r"[ً-ٰٟؐ-ؚۖ-ۭ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fingerprint(words: List[str], n: int = 5) -> str:
    if len(words) < n:
        return " ".join(words)
    grams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
    return hashlib.sha1("|".join(grams).encode()).hexdigest()[:16]


def _construction_signature(d: Dict) -> str:
    fams = sorted({c.get("family", "") for c in d.get("constructions", [])})
    return "+".join(fams) or "none"


def _dep_pattern(d: Dict) -> str:
    rels = [t.get("dep_label", {}).get("value", "_") for t in d.get("tokens", [])]
    return ">".join(rels[:20])


def _clause_depth(d: Dict) -> int:
    return d.get("curriculum", {}).get("clause_depth", 0) or 0


def _content_lemmas(d: Dict) -> Set[str]:
    out: Set[str] = set()
    for t in d.get("tokens", []):
        pos = t.get("pos", "")
        if isinstance(pos, dict):
            pos = pos.get("value", "")
        if not isinstance(pos, str):
            pos = ""
        if pos.startswith(("NOUN", "VERB", "PROPN", "ADJ")):
            lemma = t.get("lemma") or t.get("surface", "")
            if lemma:
                out.add(lemma)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(ROOT / "data_v2" / "annotated"))
    ap.add_argument("--out_dir", default=str(ROOT / "data_v2" / "adversarial_splits"))
    ap.add_argument("--train_sources", nargs="+",
                    default=["distill_v2", "ud_padt_train", "ud_padt_dev"])
    ap.add_argument("--test_sources", nargs="+",
                    default=["gazelle_test", "masaq_quranic"])
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load(name: str) -> List[Dict]:
        p = data_root / name / "all.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.open()]

    train = []
    for n in args.train_sources:
        train.extend(load(n))
    test = []
    for n in args.test_sources:
        test.extend([(d, n) for d in load(n)])

    print(f"loaded {len(train)} train, {len(test)} test")

    # Train signatures
    train_constr_sigs = {_construction_signature(d) for d in train}
    train_dep_patterns = {_dep_pattern(d) for d in train}
    train_lemmas = set()
    for d in train:
        train_lemmas |= _content_lemmas(d)
    train_fingerprints = {_fingerprint([t.get("surface", "") for t in d.get("tokens", [])])
                           for d in train}

    splits: Dict[str, List[Dict]] = defaultdict(list)
    for d, src in test:
        # construction template — keep if signature exists in train
        sig = _construction_signature(d)
        if sig != "none" and sig in train_constr_sigs:
            splits["construction_template"].append({"sentence": d, "src": src})

        # dependency pattern — keep if pattern in train
        dp = _dep_pattern(d)
        if dp in train_dep_patterns:
            splits["dependency_pattern"].append({"sentence": d, "src": src})

        # lexical family — disjoint from train lemmas
        lemmas = _content_lemmas(d)
        if lemmas and lemmas.isdisjoint(train_lemmas):
            splits["lexical_disjoint"].append({"sentence": d, "src": src})

        # clause depth
        if _clause_depth(d) >= 2:
            splits["nested_clauses"].append({"sentence": d, "src": src})

        # repeated phrase
        fp = _fingerprint([t.get("surface", "") for t in d.get("tokens", [])])
        if fp in train_fingerprints:
            # already-seen 5-gram pattern → flag for removal from honest eval
            splits["repeated_phrase_RED_FLAG"].append({"sentence": d, "src": src})

    # Summary
    summary = {k: len(v) for k, v in splits.items()}
    print("Adversarial split sizes:")
    for k, n in summary.items():
        print(f"  {k:35} {n}")

    for k, items in splits.items():
        d = out_dir / k
        d.mkdir(exist_ok=True)
        with (d / "all.jsonl").open("w") as f:
            for it in items:
                f.write(json.dumps(it["sentence"], ensure_ascii=False) + "\n")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote splits to {out_dir}")


if __name__ == "__main__":
    main()
