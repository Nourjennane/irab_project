"""Corpus-wide diagnostics on the canonical schema_v2 corpus.

Reads ``data_v2/annotated/<source>/all.jsonl`` for each registered
source and produces:

  - construction distribution
  - semantic-pressure histograms
  - ambiguity histograms
  - difficulty distributions
  - overlap statistics
  - clause-depth distributions
  - completeness distributions
  - parser-confidence distributions
  - per-source × per-axis cross-tabs

Outputs:
  data_v2/diagnostics/<axis>.md     — per-axis report
  data_v2/diagnostics/by_source.md  — per-source × per-axis cross-tabs
  data_v2/diagnostics/summary.json  — machine-readable totals
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from irab_tashkeel.data_v2.schema_v2 import Sentence, read_jsonl
from irab_tashkeel.data_v2.constructions.detector import overlap_summary

ANNOT = ROOT / "data_v2" / "annotated"
OUT = ROOT / "data_v2" / "diagnostics"


def _length_bucket(n: int) -> str:
    if n <= 8: return "short(≤8)"
    if n <= 16: return "medium(9-16)"
    if n <= 32: return "long(17-32)"
    return "xlong(>32)"


def _completeness_bucket(p: float) -> str:
    if p >= 0.99: return "fully_observable"
    if p >= 0.66: return "two_of_three"
    if p >= 0.33: return "one_of_three"
    return "none"


def _conf_bucket(p: float) -> str:
    if p < 0.5: return "<0.5"
    if p < 0.7: return "0.5-0.7"
    if p < 0.9: return "0.7-0.9"
    return ">=0.9"


def _ambiguity_bucket(s: float) -> str:
    if s < 0.1: return "0.0-0.1"
    if s < 0.3: return "0.1-0.3"
    if s < 0.5: return "0.3-0.5"
    return ">=0.5"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not ANNOT.exists():
        print(f"no annotated/ directory at {ANNOT}; run build_schema_v2_corpus.py first")
        return

    # Load all sources
    by_source: Dict[str, List[Sentence]] = {}
    for source_dir in sorted(ANNOT.iterdir()):
        if not source_dir.is_dir():
            continue
        all_jsonl = source_dir / "all.jsonl"
        if not all_jsonl.exists():
            continue
        sents = list(read_jsonl(str(all_jsonl)))
        by_source[source_dir.name] = sents
        print(f"  {source_dir.name}: {len(sents)} sentences")

    all_sents = [s for sents in by_source.values() for s in sents]
    print(f"\nTotal: {len(all_sents)} sentences across {len(by_source)} sources\n")

    # ---------------------------------------------------------------- axes
    family_dist:    Counter = Counter()
    semantic_dist:  Counter = Counter()
    ambiguity_dist: Counter = Counter()
    difficulty_dist: Counter = Counter()
    domain_dist:    Counter = Counter()
    quality_dist:   Counter = Counter()
    completeness_dist: Counter = Counter()
    length_dist:    Counter = Counter()
    dep_depth_dist: Counter = Counter()
    clause_depth_dist: Counter = Counter()
    overlap_token_dist: Counter = Counter()
    n_construction_dist: Counter = Counter()
    n_overlap_dist: Counter = Counter()
    parser_conf_role: Counter = Counter()
    parser_conf_dep:  Counter = Counter()

    # Per-source cross-tabs
    by_source_family: Dict[str, Counter] = defaultdict(Counter)
    by_source_difficulty: Dict[str, Counter] = defaultdict(Counter)
    by_source_completeness: Dict[str, Counter] = defaultdict(Counter)

    for source, sents in by_source.items():
        for s in sents:
            for c in s.constructions:
                family_dist[c.family] += 1
                by_source_family[source][c.family] += 1
            if not s.constructions:
                family_dist["_no_construction"] += 1
                by_source_family[source]["_no_construction"] += 1

            semantic_dist[s.curriculum.semantic_pressure_score] += 1
            ambiguity_dist[_ambiguity_bucket(s.curriculum.ambiguity_score)] += 1
            difficulty_dist[s.curriculum.difficulty_level] += 1
            by_source_difficulty[source][s.curriculum.difficulty_level] += 1
            domain_dist[s.metadata.domain] += 1
            quality_dist[s.metadata.annotation_quality] += 1
            completeness_dist[_completeness_bucket(s.completeness.fields_complete_pct)] += 1
            by_source_completeness[source][_completeness_bucket(s.completeness.fields_complete_pct)] += 1
            length_dist[_length_bucket(s.curriculum.sentence_length_tokens or s.n_tokens)] += 1
            dep_depth_dist[s.curriculum.dependency_depth] += 1
            clause_depth_dist[s.curriculum.clause_depth] += 1
            n_construction_dist[len(s.constructions)] += 1
            n_overlap_dist[s.curriculum.nested_construction_count] += 1

            # Overlap per-token
            o = overlap_summary(s)
            for k, v in o.items():
                overlap_token_dist[k] += v

            # Parser confidence — average across tokens
            for t in s.tokens:
                if t.role.is_present and t.role.confidence is not None:
                    parser_conf_role[_conf_bucket(t.role.confidence)] += 1
                if t.dep_label.is_present and t.dep_label.confidence is not None:
                    parser_conf_dep[_conf_bucket(t.dep_label.confidence)] += 1

    # --------------------------------------------------------------- write reports
    def _hist(title: str, counter: Counter, total: int = None,
              key_sort=None) -> List[str]:
        total = total or sum(counter.values())
        out = [f"## {title}\n", "| key | count | % |", "|---|---:|---:|"]
        if key_sort is None:
            keys = sorted(counter.keys(), key=lambda k: -counter[k])
        else:
            keys = sorted(counter.keys(), key=key_sort)
        for k in keys:
            v = counter[k]
            out.append(f"| {k} | {v} | {100*v/max(total,1):.2f}% |")
        out.append("")
        return out

    md = ["# Corpus-Wide Diagnostics\n"]
    md.append(f"Built from `data_v2/annotated/`. Total sentences: {len(all_sents)}.\n")
    md.append("## Sources")
    md.append("| source | n_sentences | n_tokens |")
    md.append("|---|---:|---:|")
    for src, sents in by_source.items():
        n_tokens = sum(s.n_tokens for s in sents)
        md.append(f"| {src} | {len(sents)} | {n_tokens} |")
    md.append("")

    md += _hist("Construction family distribution",            family_dist)
    md += _hist("Semantic pressure (0..3)",                      semantic_dist,
                key_sort=lambda k: k)
    md += _hist("Ambiguity score buckets",                       ambiguity_dist)
    md += _hist("Difficulty level (1..7)",                       difficulty_dist,
                key_sort=lambda k: k)
    md += _hist("Domain",                                        domain_dist)
    md += _hist("Annotation quality",                            quality_dist)
    md += _hist("Completeness bucket",                           completeness_dist)
    md += _hist("Sentence length bucket",                        length_dist)
    md += _hist("Dep tree depth",                                dep_depth_dist,
                key_sort=lambda k: k)
    md += _hist("Clause depth",                                  clause_depth_dist,
                key_sort=lambda k: k)
    md += _hist("Per-token construction-coverage count",         overlap_token_dist)
    md += _hist("Constructions per sentence",                    n_construction_dist,
                key_sort=lambda k: k)
    md += _hist("Nested-construction count per sentence",        n_overlap_dist,
                key_sort=lambda k: k)
    md += _hist("Parser confidence — role labels",                parser_conf_role)
    md += _hist("Parser confidence — dep labels",                 parser_conf_dep)

    (OUT / "corpus_diagnostics.md").write_text("\n".join(md))
    print(f"Wrote {OUT / 'corpus_diagnostics.md'}")

    # Per-source cross-tabs
    sx = ["# Per-Source Cross-Tabs\n"]
    sx.append("## Family × source\n")
    all_families = sorted({f for c in by_source_family.values() for f in c})
    sx.append("| source | " + " | ".join(all_families) + " |")
    sx.append("|---|" + ":---:|" * len(all_families))
    for src, counter in by_source_family.items():
        row = [src] + [str(counter.get(f, 0)) for f in all_families]
        sx.append("| " + " | ".join(row) + " |")
    sx.append("")

    sx.append("## Difficulty × source\n")
    sx.append("| source | " + " | ".join(f"diff{d}" for d in range(1, 8)) + " |")
    sx.append("|---|" + ":---:|" * 7)
    for src, counter in by_source_difficulty.items():
        row = [src] + [str(counter.get(d, 0)) for d in range(1, 8)]
        sx.append("| " + " | ".join(row) + " |")
    sx.append("")

    sx.append("## Completeness × source\n")
    buckets = ["fully_observable", "two_of_three", "one_of_three", "none"]
    sx.append("| source | " + " | ".join(buckets) + " |")
    sx.append("|---|" + ":---:|" * len(buckets))
    for src, counter in by_source_completeness.items():
        row = [src] + [str(counter.get(b, 0)) for b in buckets]
        sx.append("| " + " | ".join(row) + " |")
    sx.append("")

    (OUT / "by_source.md").write_text("\n".join(sx))
    print(f"Wrote {OUT / 'by_source.md'}")

    summary = {
        "total_sentences": len(all_sents),
        "n_sources": len(by_source),
        "by_source": {src: len(sents) for src, sents in by_source.items()},
        "family_distribution": dict(family_dist),
        "difficulty_distribution": dict(difficulty_dist),
        "domain_distribution": dict(domain_dist),
        "quality_distribution": dict(quality_dist),
        "completeness_distribution": dict(completeness_dist),
        "semantic_pressure_distribution": {str(k): v for k, v in semantic_dist.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT / 'summary.json'}")

    print("\n=== Top-line corpus signature ===")
    print(f"Total sentences: {len(all_sents)}")
    print(f"Family top-5: {dict(family_dist.most_common(5))}")
    print(f"Difficulty: {dict(sorted(difficulty_dist.items()))}")
    print(f"Domain: {dict(domain_dist)}")
    print(f"Completeness: {dict(completeness_dist)}")


if __name__ == "__main__":
    main()
