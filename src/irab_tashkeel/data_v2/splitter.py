"""Stratified train / dev / test splitter for schema_v2 corpora.

Deterministic, multi-criterion stratified sampler that prevents
construction families and difficulty levels from collapsing
between train and held-out splits — the kind of leakage / coverage
gap that biased the frozen-baseline Phase 39 / Phase R cycles.

Stratification axes (all jointly controlled)
--------------------------------------------

For each sentence we compute a discrete *stratum key* tuple from:

- **construction_signature** — sorted tuple of construction
  families present (e.g., ``("idafa", "kana_sisters")``)
- **difficulty_level** — 1..7 from ``CurriculumMetadata``
- **semantic_pressure_score** — 0..3
- **domain** — ``msa_news`` / ``quranic`` / ``classical`` / ...
- **length_bucket** — short (≤8) / medium (9-16) / long (17-32) / xlong (>32)
- **dep_depth_bucket** — shallow (0-2) / medium (3-5) / deep (≥6)

Sentences with the same stratum are split proportionally; this
guarantees parity in every axis without needing post-hoc
rebalancing.

Algorithm
---------

1. Group sentences by stratum.
2. For each stratum, sort by ``sha256(sentence_id)`` (deterministic
   pseudo-random order, reproducible across runs).
3. Allocate to train / dev / test in the requested ratios; any
   remainder goes to train.
4. Strata with too few sentences (< the smallest split's required
   1) are routed entirely to train (so dev/test never sees a
   construction it has zero examples of in train).

Outputs
-------

- ``out_dir/train.jsonl``, ``out_dir/dev.jsonl``, ``out_dir/test.jsonl``
- ``out_dir/split_report.md`` — per-axis histogram parity
- ``out_dir/coverage_report.md`` — construction coverage per split
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema_v2 import Sentence, write_jsonl


# ===========================================================================
# Stratum key
# ===========================================================================

def _length_bucket(n: int) -> str:
    if n <= 8: return "short"
    if n <= 16: return "medium"
    if n <= 32: return "long"
    return "xlong"


def _dep_depth_bucket(d: int) -> str:
    if d <= 2: return "shallow"
    if d <= 5: return "medium"
    return "deep"


def _stratum_key(s: Sentence) -> Tuple[str, ...]:
    families = tuple(sorted({c.family for c in s.constructions}))
    return (
        ",".join(families) or "_no_construction",
        f"diff{s.curriculum.difficulty_level}",
        f"sp{s.curriculum.semantic_pressure_score}",
        s.metadata.domain or "unknown",
        _length_bucket(s.curriculum.sentence_length_tokens or s.n_tokens),
        _dep_depth_bucket(s.curriculum.dependency_depth),
    )


def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


# ===========================================================================
# Splitter
# ===========================================================================

@dataclass
class SplitConfig:
    train_ratio: float = 0.85
    dev_ratio:   float = 0.075
    test_ratio:  float = 0.075
    # Strata smaller than `min_stratum_for_eval` go entirely to train.
    min_stratum_for_eval: int = 5

    def validate(self) -> None:
        total = self.train_ratio + self.dev_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")


@dataclass
class SplitResult:
    train: List[Sentence] = field(default_factory=list)
    dev:   List[Sentence] = field(default_factory=list)
    test:  List[Sentence] = field(default_factory=list)
    stratum_assignments: Dict[Tuple[str, ...], Dict[str, int]] = \
        field(default_factory=dict)


def stratified_split(
    sentences: List[Sentence],
    config: Optional[SplitConfig] = None,
) -> SplitResult:
    """Split ``sentences`` into train / dev / test deterministically."""
    config = config or SplitConfig()
    config.validate()

    # Group by stratum
    by_stratum: Dict[Tuple[str, ...], List[Sentence]] = defaultdict(list)
    for s in sentences:
        by_stratum[_stratum_key(s)].append(s)

    result = SplitResult()
    for stratum, group in by_stratum.items():
        group.sort(key=lambda s: _stable_hash(s.sentence_id))

        # Tiny strata → all to train (preserves diversity across splits)
        if len(group) < config.min_stratum_for_eval:
            result.train.extend(group)
            result.stratum_assignments[stratum] = {
                "train": len(group), "dev": 0, "test": 0,
            }
            continue

        n = len(group)
        n_test = max(1, int(round(n * config.test_ratio)))
        n_dev  = max(1, int(round(n * config.dev_ratio)))
        n_train = n - n_dev - n_test
        if n_train < 1:
            # Pathological tiny stratum that survived min_stratum_for_eval
            # but has no train room — route everything to train.
            result.train.extend(group)
            result.stratum_assignments[stratum] = {
                "train": len(group), "dev": 0, "test": 0,
            }
            continue

        train = group[:n_train]
        dev   = group[n_train:n_train + n_dev]
        test  = group[n_train + n_dev:]

        result.train.extend(train)
        result.dev.extend(dev)
        result.test.extend(test)
        result.stratum_assignments[stratum] = {
            "train": len(train), "dev": len(dev), "test": len(test),
        }

    return result


# ===========================================================================
# Diagnostics + IO
# ===========================================================================

def _axis_histogram(sentences: List[Sentence], axis: str) -> Counter:
    ctr: Counter = Counter()
    for s in sentences:
        if axis == "domain":
            ctr[s.metadata.domain] += 1
        elif axis == "difficulty":
            ctr[s.curriculum.difficulty_level] += 1
        elif axis == "semantic_pressure":
            ctr[s.curriculum.semantic_pressure_score] += 1
        elif axis == "annotation_quality":
            ctr[s.metadata.annotation_quality] += 1
        elif axis == "construction_family":
            for c in s.constructions:
                ctr[c.family] += 1
            if not s.constructions:
                ctr["_no_construction"] += 1
        elif axis == "length_bucket":
            ctr[_length_bucket(s.n_tokens)] += 1
        elif axis == "dep_depth_bucket":
            ctr[_dep_depth_bucket(s.curriculum.dependency_depth)] += 1
    return ctr


def _coverage(
    train: List[Sentence], dev: List[Sentence], test: List[Sentence],
    axis: str,
) -> List[Tuple[str, int, int, int]]:
    keys = set()
    for split in (train, dev, test):
        for s in split:
            if axis == "construction_family":
                for c in s.constructions:
                    keys.add(c.family)
                if not s.constructions:
                    keys.add("_no_construction")
            elif axis == "domain":
                keys.add(s.metadata.domain)
    if not keys:
        return []
    train_h = _axis_histogram(train, axis)
    dev_h   = _axis_histogram(dev, axis)
    test_h  = _axis_histogram(test, axis)
    return sorted(
        [(k, train_h.get(k, 0), dev_h.get(k, 0), test_h.get(k, 0))
         for k in keys],
        key=lambda r: -(r[1] + r[2] + r[3]),
    )


def write_split(out_dir: str | Path, result: SplitResult) -> Dict[str, int]:
    """Write the split to JSONL + diagnostic markdown."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out / "train.jsonl"), result.train)
    write_jsonl(str(out / "dev.jsonl"),   result.dev)
    write_jsonl(str(out / "test.jsonl"),  result.test)

    # Split report
    report = ["# Stratified Split Report\n"]
    report.append(f"- train: {len(result.train)} sentences")
    report.append(f"- dev:   {len(result.dev)} sentences")
    report.append(f"- test:  {len(result.test)} sentences")
    report.append(f"- distinct strata: {len(result.stratum_assignments)}\n")

    for axis in ("domain", "difficulty", "semantic_pressure",
                 "annotation_quality", "length_bucket", "dep_depth_bucket"):
        report.append(f"## Axis: {axis}\n")
        report.append("| key | train | dev | test |")
        report.append("|---|---:|---:|---:|")
        all_keys = set()
        all_keys.update(_axis_histogram(result.train, axis).keys())
        all_keys.update(_axis_histogram(result.dev,   axis).keys())
        all_keys.update(_axis_histogram(result.test,  axis).keys())
        train_h = _axis_histogram(result.train, axis)
        dev_h   = _axis_histogram(result.dev,   axis)
        test_h  = _axis_histogram(result.test,  axis)
        for k in sorted(all_keys, key=lambda x: str(x)):
            report.append(f"| {k} | {train_h.get(k, 0)} | "
                          f"{dev_h.get(k, 0)} | {test_h.get(k, 0)} |")
        report.append("")

    (out / "split_report.md").write_text("\n".join(report))

    # Coverage report
    cov = ["# Construction Coverage\n"]
    cov.append("| family | train | dev | test |")
    cov.append("|---|---:|---:|---:|")
    for fam, t, d, te in _coverage(result.train, result.dev, result.test, "construction_family"):
        cov.append(f"| {fam} | {t} | {d} | {te} |")
    (out / "coverage_report.md").write_text("\n".join(cov))

    # Stratum assignments JSON dump
    sa = [{"stratum": list(k), **v}
          for k, v in result.stratum_assignments.items()]
    (out / "stratum_assignments.json").write_text(
        json.dumps(sa, indent=2, ensure_ascii=False)
    )

    return {"train": len(result.train), "dev": len(result.dev),
            "test": len(result.test)}
