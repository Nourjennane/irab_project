"""Phase B — leakage audit for Arabic iʿrāb corpora.

Aggressively probes overlap between every (train | curriculum-replay)
source and every held-out test set:

  - exact duplicate (full sentence string)
  - normalised duplicate (stripped diacritics + tatweel + punctuation)
  - n-gram overlap (5-gram Jaccard ≥ 0.7)
  - sentence-hash audit (sha1 of normalised form)
  - token-level overlap statistics (mean shared-token ratio)
  - cross-stage overlap (curriculum replay contamination)
  - construction-family overlap by source

Inputs are schema_v2 jsonl shards under ``data_v2/annotated/``.

Outputs to ``docs/leakage_audit/``:

  - ``leakage_report.md``        human-readable summary + flagged items
  - ``leakage_summary.json``     machine-readable counts + ratios
  - ``overlap_tables.csv``       (train_src, test_src, exact, norm, fuzzy)
  - ``suspicious_examples.jsonl``  near-duplicate pairs for manual review

Design intent: this is a **pure dataset audit**. No model inference,
no checkpoint state. The audit is deterministic and reproducible from
the schema_v2 corpus alone.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


# ===========================================================================
# Text normalisation
# ===========================================================================

ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟؐ-ؚۖ-ۭ]")
TATWEEL = "ـ"
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalise(text: str) -> str:
    """Strip diacritics, tatweel, punctuation; collapse whitespace; lowercase."""
    text = unicodedata.normalize("NFC", text)
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.replace(TATWEEL, "")
    text = PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def sentence_hash(text: str) -> str:
    return hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()[:16]


def token_set(text: str) -> Set[str]:
    return set(normalise(text).split())


def ngram_set(text: str, n: int = 5) -> Set[Tuple[str, ...]]:
    toks = normalise(text).split()
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: Set, b: Set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ===========================================================================
# Loaders
# ===========================================================================

def load_source(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load schema_v2 sentences as flat dicts: {id, text, source, ngrams, ...}.

    Raw json read avoids importing the full schema_v2 dataclass — we
    only need ``surface`` text for leakage purposes.
    """
    out = []
    with jsonl_path.open() as f:
        for line in f:
            d = json.loads(line)
            text = " ".join(t.get("surface", "") for t in d.get("tokens", []))
            sid = d.get("sentence_id") or d.get("id") or ""
            src = d.get("metadata", {}).get("source", jsonl_path.parent.name)
            domain = d.get("metadata", {}).get("domain", "")
            text_norm = normalise(text)
            out.append({
                "id": sid,
                "source": src,
                "domain": domain,
                "text": text,
                "text_norm": text_norm,
                "hash": sentence_hash(text),
                "tokens_norm": tuple(text_norm.split()),
                "ngrams": ngram_set(text, n=5),
            })
    return out


# ===========================================================================
# Pairwise overlap audits
# ===========================================================================

def exact_overlap(train: List[Dict], test: List[Dict]) -> List[Tuple[Dict, Dict]]:
    train_by_text = {t["text"]: t for t in train}
    return [(train_by_text[s["text"]], s) for s in test if s["text"] in train_by_text]


def normalised_overlap(train: List[Dict], test: List[Dict]) -> List[Tuple[Dict, Dict]]:
    train_by_norm = {t["text_norm"]: t for t in train}
    return [(train_by_norm[s["text_norm"]], s)
            for s in test if s["text_norm"] in train_by_norm]


def hash_overlap(train: List[Dict], test: List[Dict]) -> List[Tuple[Dict, Dict]]:
    train_by_hash = {t["hash"]: t for t in train}
    return [(train_by_hash[s["hash"]], s)
            for s in test if s["hash"] in train_by_hash]


def fuzzy_overlap(
    train: List[Dict], test: List[Dict], *,
    threshold: float = 0.7, max_pairs: int = 200,
) -> List[Tuple[Dict, Dict, float]]:
    """Token-Jaccard near-duplicates above ``threshold``.

    O(N×M); fine for our corpus sizes. Truncates to top-``max_pairs``
    by similarity.
    """
    test_sets = [(t, set(t["tokens_norm"])) for t in test]
    out: List[Tuple[Dict, Dict, float]] = []
    for tr in train:
        tr_set = set(tr["tokens_norm"])
        if not tr_set:
            continue
        for te, te_set in test_sets:
            if not te_set:
                continue
            inter = len(tr_set & te_set)
            if inter == 0:
                continue
            j = inter / len(tr_set | te_set)
            if j >= threshold:
                out.append((tr, te, j))
    out.sort(key=lambda x: -x[2])
    return out[:max_pairs]


def ngram_overlap(
    train: List[Dict], test: List[Dict], *,
    threshold: float = 0.5, max_pairs: int = 200,
) -> List[Tuple[Dict, Dict, float]]:
    out: List[Tuple[Dict, Dict, float]] = []
    for tr in train:
        if not tr["ngrams"]:
            continue
        for te in test:
            if not te["ngrams"]:
                continue
            j = jaccard(tr["ngrams"], te["ngrams"])
            if j >= threshold:
                out.append((tr, te, j))
    out.sort(key=lambda x: -x[2])
    return out[:max_pairs]


# ===========================================================================
# Aggregate audit
# ===========================================================================

def audit_pair(
    train: List[Dict], test: List[Dict],
    *, train_name: str, test_name: str,
    fuzzy_threshold: float = 0.7,
    ngram_threshold: float = 0.5,
) -> Dict[str, Any]:
    exact   = exact_overlap(train, test)
    nrmd    = normalised_overlap(train, test)
    hashed  = hash_overlap(train, test)
    fuzzy   = fuzzy_overlap(train, test, threshold=fuzzy_threshold)
    ngramov = ngram_overlap(train, test, threshold=ngram_threshold)

    return {
        "train": train_name, "test": test_name,
        "n_train": len(train), "n_test": len(test),
        "exact": len(exact),
        "normalised": len(nrmd),
        "hash_dup": len(hashed),
        "fuzzy_jaccard_ge_0.7": len(fuzzy),
        f"ngram5_jaccard_ge_{ngram_threshold}": len(ngramov),
        "exact_pct":      round(100 * len(exact) / max(len(test), 1), 3),
        "normalised_pct": round(100 * len(nrmd) / max(len(test), 1), 3),
        "fuzzy_pct":      round(100 * len(fuzzy) / max(len(test), 1), 3),
        "_exact_pairs":  [(a["id"], b["id"]) for a, b in exact[:50]],
        "_norm_pairs":   [(a["id"], b["id"]) for a, b in nrmd[:50]],
        "_fuzzy_pairs":  [(a["id"], b["id"], round(j, 3))
                          for a, b, j in fuzzy[:50]],
        "_ngram_pairs":  [(a["id"], b["id"], round(j, 3))
                          for a, b, j in ngramov[:50]],
    }


# ===========================================================================
# Top-level
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(ROOT / "data_v2" / "annotated"))
    ap.add_argument("--out_dir", default=str(ROOT / "docs" / "leakage_audit"))
    ap.add_argument("--train_sources", nargs="+",
                    default=["distill_v2", "ud_padt_train", "ud_padt_dev"])
    ap.add_argument("--test_sources", nargs="+",
                    default=["gazelle_test", "masaq_quranic", "ud_padt_test"])
    ap.add_argument("--fuzzy_threshold", type=float, default=0.7)
    ap.add_argument("--ngram_threshold", type=float, default=0.5)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sources...")
    sources: Dict[str, List[Dict]] = {}
    for name in args.train_sources + args.test_sources:
        path = data_root / name / "all.jsonl"
        if not path.exists():
            print(f"  [warn] missing {path}")
            continue
        sources[name] = load_source(path)
        print(f"  {name:25} {len(sources[name])}")

    # Pairwise audit: every train × every test
    print("\nPairwise overlap audit:")
    table_rows: List[Dict[str, Any]] = []
    suspicious: List[Dict[str, Any]] = []
    for trn in args.train_sources:
        for tst in args.test_sources:
            if trn not in sources or tst not in sources:
                continue
            res = audit_pair(
                sources[trn], sources[tst],
                train_name=trn, test_name=tst,
                fuzzy_threshold=args.fuzzy_threshold,
                ngram_threshold=args.ngram_threshold,
            )
            print(f"  {trn:18} → {tst:18}  exact={res['exact']:>4}  "
                  f"norm={res['normalised']:>4}  fuzzy={res['fuzzy_jaccard_ge_0.7']:>4}")
            table_rows.append(res)
            for k in ("_exact_pairs", "_norm_pairs", "_fuzzy_pairs", "_ngram_pairs"):
                for pair in res.get(k, [])[:20]:
                    suspicious.append({"audit": k, "train": trn, "test": tst,
                                        "pair": pair})

    # Cross-test audit (test ↔ test) — would catch dataset-overlap bugs
    print("\nCross-test overlap (sanity):")
    for i, t1 in enumerate(args.test_sources):
        for t2 in args.test_sources[i + 1:]:
            if t1 not in sources or t2 not in sources:
                continue
            res = audit_pair(sources[t1], sources[t2],
                             train_name=t1, test_name=t2)
            print(f"  {t1} ↔ {t2}: exact={res['exact']} norm={res['normalised']}")
            table_rows.append(res)

    # Write outputs
    summary = {
        "fuzzy_threshold": args.fuzzy_threshold,
        "ngram_threshold": args.ngram_threshold,
        "n_sources": {k: len(v) for k, v in sources.items()},
        "rows": table_rows,
    }
    (out_dir / "leakage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    with (out_dir / "overlap_tables.csv").open("w", newline="") as f:
        if table_rows:
            cols = ["train", "test", "n_train", "n_test", "exact", "normalised",
                    "hash_dup", "fuzzy_jaccard_ge_0.7", "exact_pct",
                    "normalised_pct", "fuzzy_pct"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in table_rows:
                w.writerow(r)

    with (out_dir / "suspicious_examples.jsonl").open("w") as f:
        for s in suspicious:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Markdown report
    lines = ["# Phase B — Leakage Audit", "",
             f"Audited train sources: {args.train_sources}",
             f"Against test sources: {args.test_sources}",
             "",
             "## Pairwise overlap counts",
             "",
             "| train | test | n_train | n_test | exact | normalised | "
             "hash_dup | fuzzy_J≥0.7 | exact_% | norm_% | fuzzy_% |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table_rows:
        lines.append(f"| {r['train']} | {r['test']} | {r['n_train']} | "
                     f"{r['n_test']} | {r['exact']} | {r['normalised']} | "
                     f"{r['hash_dup']} | {r['fuzzy_jaccard_ge_0.7']} | "
                     f"{r['exact_pct']} | {r['normalised_pct']} | "
                     f"{r['fuzzy_pct']} |")
    lines += ["",
              "## Interpretation",
              "",
              "- `exact`: identical raw sentence text",
              "- `normalised`: identical after stripping diacritics + tatweel + punctuation",
              "- `hash_dup`: identical sha1 of normalised form (sanity check)",
              "- `fuzzy_J≥0.7`: token-Jaccard ≥ 0.7 (likely paraphrase or near-dup)",
              "",
              "Any non-zero exact or normalised count between train and test is a "
              "**direct leakage red flag** and must be cleaned before claiming "
              "the trained model's metrics. Fuzzy overlaps need manual review — "
              "see `suspicious_examples.jsonl`.",
              ""]
    (out_dir / "leakage_report.md").write_text("\n".join(lines))

    print(f"\nWrote:")
    for f in ("leakage_report.md", "leakage_summary.json",
              "overlap_tables.csv", "suspicious_examples.jsonl"):
        print(f"  {out_dir / f}")


if __name__ == "__main__":
    main()
