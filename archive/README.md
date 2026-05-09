# Archived Experimental Variants

Variants from the project's research cycle that did not ship as
production but are kept in the repository as case-study evidence
and for paper reproduction.

These artifacts are kept **in place** rather than physically moved,
because moving them would break import paths and the LaTeX document
references. The README below acts as the inventory.

## Production checkpoints (not archived — these are the live system)

| Artifact | What it is | Where |
|---|---|---|
| `validated_nextgen_recovery` | **Production checkpoint** | `runs/validated_nextgen_recovery/` |
| `final_validated/` | Frozen Phase A artifact | `runs/final_validated/` |
| `final_graph_negative_result/` | Frozen graph experiment | `runs/final_graph_negative_result/` |

## Documented negative results (kept on purpose)

| Experiment | Frozen at | Why archived, not shipped |
|---|---|---|
| Graph integration | [`docs/final_graph_negative_result/`](../docs/final_graph_negative_result/) | Tied with recovery on every clean held-out metric — see NEGATIVE_RESULT.md |
| Governor head | [`docs/final_governor_negative_result/`](../docs/final_governor_negative_result/) | Did not displace the dominant idafa-attachment confusion |

Both directories ship full eval data + failure analysis + a
NEGATIVE_RESULT.md that explains the methodology and result.

## Older archival code (in-place, in `src/`)

| Phase | Location | Status |
|---|---|---|
| Phase 4a (taxonomy v4) | `src/irab_tashkeel/structured/taxonomy_v4.py` | substitutable with morph |
| Phase 2 conditioning | `src/irab_tashkeel/morphology/conditioning.py` | joint-dynamics regression |
| Phase 5 case hierarchy | flag in `dep_aware_model.py` | flat / regress |
| Phase 6 marker hierarchy | flag in `dep_aware_model.py` | flat / regress |
| Phase 3.1 relational attention | `src/irab_tashkeel/morphology/relational_reasoning.py` | flat / regress |
| Phase 39 synthetic data | `data/structured_v1_augmented/` | distribution mismatch on Gazelle |
| Phase R retrieval | `src/irab_tashkeel/grammar_memory/retrieval_predictor.py` | partial; MASAQ only |
| Phase R2 reasoning | `src/irab_tashkeel/grammar_memory/structural_{reasoner,predictor}.py` | 0.0 with proper forward |

## Older eval artefacts (in-place, in `runs/`)

The following directories under `runs/` hold *eval-only* artefacts
from earlier project phases. They contain prediction JSON, metric
JSON, and per-construction summaries — no model weights. Kept for
the paper's negative-result tables.

- `runs/per_construction_phase3a/` — frozen per-construction breakdown for Phase 3-A
- `runs/structured_v1_eval_*` — Step-1 rebuild eval slices
- `runs/phase1_morph_eval_490987/` — Phase 1 morph baseline eval
- `runs/phase4a_eval_phase4a_taxonomy_no_morph_491040/` — Phase 4a eval

These should remain untouched; they are evidence for the paper's
ablation history table (§ 9 of `docs/paper/PAPER.md`).

## What we explicitly removed

- `runs/nextgen/` (the leaked stage_7 training output): **deleted**
  during the supervision phase to free disk. The leaked checkpoint
  was a contamination artifact; we kept its eval traces in
  `docs/final_eval/raw/stage7__*.json` and the case-study writeup
  in `docs/paper/PAPER.md` § 8.5.
- `runs/nextgen_recovery/` (the per-stage recovery checkpoints):
  **deleted** after `runs/validated_nextgen_recovery/` was frozen
  via the freeze script. The validated artifact is independent and
  carries the full reproducibility manifest.
- `runs/nextgen_graph/` (per-stage graph training): **deleted**
  after the negative result was archived. The frozen final artefact
  is in `docs/final_graph_negative_result/eval_data/`.

## How to add to this archive in the future

1. Run the freeze script:
   `python scripts/freeze_canonical_artifacts.py --dst_root runs/`
   (extend the script if you have a new candidate to freeze)
2. Add a `NEGATIVE_RESULT.md` if applicable.
3. Update this README's tables.
4. Commit + push.
