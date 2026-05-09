# Archived Experimental Variants

Variants from the original research cycle that did not ship as
production but are kept in the repository as case-study evidence
and for paper reproduction.

These are **not** moved here as files, because doing so would break
import paths and the LaTeX document references. They live where
they always did, marked archival in `docs/CURRENT_STATE.md` and
documented in `docs/REPORT.md` §5.6.

This directory tree mirrors the conceptual archive structure for
**future** archival material on this branch — anything that fails
the next-generation gates and is preserved for the paper rather
than shipped as production.

## Existing archival code (in-place, not moved)

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
