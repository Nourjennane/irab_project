# Final Phase 3-A Summary — Validated Scientific Baseline

**Date:** 2026-05-09
**Branch:** `main` (frozen) and `nextgen-grammatical-reasoning` (active development)
**Production checkpoint:** `runs/phase3a_491240/final/`

This document is the canonical short summary of what Phase 3-A is,
what works, what failed, what was learned. The full report is in
`docs/REPORT.md` (and `docs/paper/REPORT.tex` / `REPORT.pdf`).

---

## Final scientific thesis

> *At 296M parameters / 6 epochs / a ~7K-sentence supervised corpus,
> Arabic iʿrāb performance improves through (a) orthogonal linguistic
> information added to the encoder input (morphology supervision,
> dependency features), and (b) honest evaluation methodology.
> Architectural rearrangement (taxonomy expansion, conditioning,
> hierarchical decoders, relational attention) and inference-side
> mechanisms (retrieval blending, structural reasoning override) all
> plateau or regress.*

The bottleneck at this scale is information coverage and supervision
quality, not architecture.

---

## Final metrics (Phase 3-A on FIXED evaluator)

### Gazelle (n=107 word judgments, 30 sentences)

| construction | n | case | role | marker | *fully* | calib_gap |
|---|---:|---:|---:|---:|---:|---:|
| kana_sisters | 7 | 71.4 | 57.1 | 14.3 | **14.3** | −0.189 |
| inna_sisters | 11 | 90.9 | 27.3 | 72.7 | 27.3 | −0.132 |
| istithna | 5 | 80.0 | 40.0 | 20.0 | 0.0 | −0.291 |
| idafa | 48 | 81.2 | 47.9 | 62.5 | 33.3 | −0.127 |
| quranic_proxy | 8 | 37.5 | 12.5 | 62.5 | 0.0 | +0.245 |
| **overall** | **107** | **72.0** | **40.2** | **62.6** | **25.2** | **−0.101** |

### MASAQ (n=5,007 word judgments, 624 sentences)

| construction | n | case | role | marker | *fully* | calib_gap |
|---|---:|---:|---:|---:|---:|---:|
| kana_sisters | 299 | 86.6 | 12.0 | 31.1 | **11.0** | −0.063 |
| inna_sisters | 1,440 | 85.3 | 15.0 | 33.9 | 13.5 | +0.031 |
| istithna | 602 | 88.2 | 14.4 | 29.4 | 12.5 | +0.084 |
| idafa | 1,606 | 86.4 | 26.8 | 41.0 | 24.2 | +0.050 |
| idafa_multi | 47 | 89.4 | 44.7 | 48.9 | 29.8 | +0.045 |
| quranic_proxy | 565 | 88.5 | 13.3 | 26.0 | 11.3 | +0.049 |
| **overall** | **5,007** | **85.8** | **17.1** | **33.0** | **14.9** | **+0.048** |

---

## What worked

| intervention | result vs prior best |
|---|---|
| **Phase 1** — morphology supervision (7 auxiliary morph heads on UD-PADT, jointly trained) | +5.4 role-F1, +1.5 *fully* on Gazelle. First architectural intervention to lift role-F1. Production after rev 2. |
| **Phase 3** — static UD dependency features (DEPREL + HEAD topology + governor POS, parsed offline by Stanza, concatenated to encoder pooled feature, identity-init projection) | +3.0 case, +3.8 marker, +0.7 *fully* vs Phase 1. Three of four metrics improve. **Production checkpoint.** |
| **Evaluator fix** — kana-aware role extraction in `evaluation/structural.py` | Unblocks Gazelle kana_sisters *fully* from artificial 0% to real 14.3%. Model unchanged. Permanent. |

---

## What failed or plateaued

| intervention | result | category |
|---|---|---|
| Phase 4a — taxonomy expansion (25 → 34 labels) | role-F1 +6.8 individually but only +5.7 combined with morph | partial substitution |
| Phase 2 — soft conditioning (FiLM joint, additive joint, FiLM detached) | 0/3 ship; FiLM-joint regresses all 4 metrics | rearrangement |
| Phase 5 — hierarchical case decoder (role → case bias) | 0/3 wins | rearrangement |
| Phase 6 — hierarchical marker decoder (case+role → marker bias) | 0/3 wins; role-F1 −2.5 | rearrangement |
| Phase 3.1 — dynamic relational attention over UD tree | flat / regress vs Phase 3-A | rearrangement |
| Phase 39 — synthetic rare-construction augmentation (22% mix) | Gazelle role-F1 −12.9, MASAQ +0.4 | distribution mismatch |
| Phase R-C — retrieval + soft logit bias | Gazelle gates fail, MASAQ partial (+0.7 fully, +8.5 idafa_multi) | inference-side; distribution-bound |
| Phase R2 — per-construction reasoning + 3-tier override | 0.0 everywhere with corrected forward path | inference-side; mechanism-neutral |

---

## Key lessons

### 1. Orthogonal information unlocks gains; rearrangement plateaus

Across 7 architectural variants tested on top of the rev 2 baseline,
only Phase 1 (morphology supervision from UD-PADT, a different label
space) and Phase 3 (UD dependency edges, a different signal modality)
produced clean wins. Every variant that re-uses information already
in the model — taxonomy expansion (Phase 4a), morph→iʿrāb conditioning
(Phase 2), output-bias hierarchies (Phases 5, 6), relational attention
over the same dep tree (Phase 3.1) — plateaus or regresses.

### 2. Joint optimisation dynamics, not the form of conditioning, is the bottleneck

Phase 2's FiLM-joint vs FiLM-detached contrast is the load-bearing
finding: the same conditioning module produces qualitatively different
outcomes depending on whether iʿrāb-side gradients flow back into the
morph heads. The morph representation drifts under joint training and
the iʿrāb heads chase the moving target.

### 3. Distribution matching matters more than information quantity

Phase 39's 22%-mix synthetic kāna / istithnāʾ data lifts MASAQ Quranic
uniformly (+0.4 across all metrics) but crashes Gazelle MSA-news role-F1
by −12.9. Same intervention, opposite direction on different test
distributions. The role head overfit to the template surface
distribution. Subsequent retrieval and reasoning experiments showed the
same Gazelle-vs-MASAQ asymmetry, marking distribution match as a
first-class constraint.

### 4. Inference-side reasoning is mechanism-neutral at this scale

Phase R-C and Phase R2 added retrieval-guided reasoning on top of
Phase 3-A. With a corrected forward path (using `model.forward(...)`
instead of a manual encoder re-implementation), R2 produces 0.0
change on every metric on every construction on both surfaces. The
reasoner fires on detected spans; its consensus labels match what
Phase 3-A is already producing. Net signal: zero.

### 5. Honest evaluation methodology adds real measurement value

The Gazelle kana_sisters fully metric was pinned at artificial 0%
because the gold extractor only matched literal "اسم كان" / "خبر كان"
prose, while Gazelle uses particle-specific phrasing for any of 12
kāna-family particles. Adding `_detect_kana_role()` to the evaluator
revealed Phase 3-A was already getting 14.3% kana fully and 57.1%
kana role. The model has been doing kāna reasoning correctly all
along; the evaluator was hiding it.

---

## Frozen status

The Phase 3-A architecture, training recipe, evaluator, and metric
tables are immutable. Future development happens on the
`nextgen-grammatical-reasoning` branch with explicit comparison
back to these numbers.

The empirical evidence summarised above motivates the next-generation
branch's design: scale, supervision diversity, syntactic depth,
semantic and discourse reasoning, longer training, stronger pretrained
Arabic backbones — not architectural rearrangement.
