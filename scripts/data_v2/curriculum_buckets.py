"""Generate curriculum-stage candidate buckets from the schema_v2 corpus.

Reads ``data_v2/annotated/<source>/all.jsonl`` and assigns each
sentence to a curriculum stage (1..7) based on its
:class:`CurriculumMetadata.difficulty_level`. Within each stage,
records the per-source distribution + construction-family
diversity.

Outputs:

    data_v2/curriculum/stage_{1..7}.jsonl       — per-stage corpus
    data_v2/curriculum/curriculum_report.md     — bucket summary
    data_v2/curriculum/stage_assignments.json   — machine-readable

Stages (defined in ``src/irab_tashkeel/curriculum/README.md``):

    1: pure morphology
    2: local syntax
    3: simple constructions
    4: nested syntax
    5: semantic interactions
    6: discourse-sensitive structures
    7: Quranic / classical complexity

Empirical reality on this corpus
--------------------------------

After running on the full schema_v2 corpus we report stage-by-stage:

  - sentence count
  - source breakdown (per stage, how many from distill_v2 / UD-PADT / MASAQ /
    Gazelle)
  - family diversity (unique construction families per stage)
  - average semantic pressure
  - average dep depth
  - completeness fraction

This is the curriculum-ready corpus report — used to time the
introduction of stages during training (Step 7).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from irab_tashkeel.data_v2.schema_v2 import Sentence, read_jsonl, write_jsonl

ANNOT = ROOT / "data_v2" / "annotated"
OUT = ROOT / "data_v2" / "curriculum"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Load all sources
    by_source: Dict[str, List[Sentence]] = {}
    for source_dir in sorted(ANNOT.iterdir()):
        if not source_dir.is_dir(): continue
        path = source_dir / "all.jsonl"
        if not path.exists(): continue
        sents = list(read_jsonl(str(path)))
        by_source[source_dir.name] = sents

    all_sents = [(src, s) for src, sents in by_source.items() for s in sents]
    print(f"Loaded {len(all_sents)} sentences from {len(by_source)} sources")

    # Bucket by stage
    stages: Dict[int, List[tuple]] = defaultdict(list)
    for src, s in all_sents:
        stage = s.curriculum.difficulty_level
        stages[stage].append((src, s))

    # Per-stage statistics
    stage_stats: Dict[int, Dict] = {}
    for stage in range(1, 8):
        rows = stages.get(stage, [])
        if not rows:
            stage_stats[stage] = {"n": 0}
            continue
        sources = Counter(src for src, _ in rows)
        domains = Counter(s.metadata.domain for _, s in rows)
        families: Counter = Counter()
        for _, s in rows:
            for c in s.constructions:
                families[c.family] += 1
        avg_sp = sum(s.curriculum.semantic_pressure_score for _, s in rows) / len(rows)
        avg_dep = sum(s.curriculum.dependency_depth for _, s in rows) / len(rows)
        avg_compl = sum(s.completeness.fields_complete_pct for _, s in rows) / len(rows)
        avg_amb = sum(s.curriculum.ambiguity_score for _, s in rows) / len(rows)
        avg_len = sum(s.curriculum.sentence_length_tokens or s.n_tokens
                       for _, s in rows) / len(rows)

        stage_stats[stage] = {
            "n": len(rows),
            "sources": dict(sources),
            "domains": dict(domains),
            "family_distribution": dict(families.most_common(10)),
            "n_unique_families": len(families),
            "mean_semantic_pressure": round(avg_sp, 3),
            "mean_dep_depth": round(avg_dep, 3),
            "mean_completeness": round(avg_compl, 3),
            "mean_ambiguity": round(avg_amb, 3),
            "mean_length": round(avg_len, 1),
        }

        # Write per-stage JSONL
        out_path = OUT / f"stage_{stage}.jsonl"
        write_jsonl(str(out_path), [s for _, s in rows])
        print(f"  stage {stage}: n={len(rows)}, families={len(families)}, "
              f"avg_sp={avg_sp:.2f}, avg_dep={avg_dep:.2f} → {out_path.name}")

    (OUT / "stage_assignments.json").write_text(
        json.dumps(stage_stats, indent=2, ensure_ascii=False)
    )

    # Curriculum-ready report
    md = ["# Curriculum-Ready Corpus Report\n"]
    md.append(f"Built from `data_v2/annotated/`. Total sentences: "
              f"{sum(s['n'] for s in stage_stats.values())}.\n")
    md.append("Each row is one curriculum stage. Stages with `n=0` "
              "lack source coverage and the curriculum scheduler must "
              "either down-weight them or trigger targeted annotation.\n")
    md.append("## Stage overview\n")
    md.append("| stage | n_sentences | n_unique_families | avg_dep_depth | "
              "avg_sem_pressure | avg_completeness | avg_length |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for stage in range(1, 8):
        s = stage_stats[stage]
        if s["n"] == 0:
            md.append(f"| {stage} | 0 | — | — | — | — | — |")
            continue
        md.append(f"| {stage} | {s['n']} | {s['n_unique_families']} | "
                  f"{s['mean_dep_depth']:.2f} | "
                  f"{s['mean_semantic_pressure']:.2f} | "
                  f"{s['mean_completeness']:.2f} | "
                  f"{s['mean_length']:.0f} |")
    md.append("")

    md.append("## Per-stage source breakdown\n")
    md.append("| stage | distill_v2 | ud_padt_train | ud_padt_dev | "
              "ud_padt_test | masaq_quranic | gazelle_test |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for stage in range(1, 8):
        s = stage_stats[stage]
        if s["n"] == 0:
            md.append(f"| {stage} | 0 | 0 | 0 | 0 | 0 | 0 |")
            continue
        srcs = s["sources"]
        md.append(f"| {stage} | {srcs.get('distill_v2',0)} | "
                  f"{srcs.get('ud_padt_train',0)} | {srcs.get('ud_padt_dev',0)} | "
                  f"{srcs.get('ud_padt_test',0)} | {srcs.get('masaq_quranic',0)} | "
                  f"{srcs.get('gazelle_test',0)} |")
    md.append("")

    md.append("## Per-stage family distribution (top families)\n")
    for stage in range(1, 8):
        s = stage_stats[stage]
        if s["n"] == 0: continue
        md.append(f"### Stage {stage} (n={s['n']})")
        md.append("| family | count |")
        md.append("|---|---:|")
        for f, c in s["family_distribution"].items():
            md.append(f"| {f} | {c} |")
        md.append("")

    (OUT / "curriculum_report.md").write_text("\n".join(md))
    print(f"\nWrote {OUT / 'curriculum_report.md'}")


if __name__ == "__main__":
    main()
