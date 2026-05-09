# Ablations v2

Structured ablation outputs for the next-generation branch.

Per-experiment per-construction tables (Gazelle + MASAQ + new
hard-construction subsets from eval_v2) live here, named
`<exp_id>_<short_name>/`.

Each ablation directory contains:

- `per_construction_summary.json` (canonical machine-readable)
- `per_construction_summary.md` (human-readable)
- `traces.jsonl` (Phase R2-style decoder reasoning traces, when
  the run uses the structural reasoning module)
- `metrics_diff_vs_phase3a.md` (delta vs the frozen baseline)
- `error_tags.jsonl` (per-error category tags from
  `docs/error_taxonomy.md`)

The frozen-baseline ablation outputs continue to live in
`runs/per_construction_phase3a/` and `runs/phaseR2_*` (legacy);
they are not duplicated here.
