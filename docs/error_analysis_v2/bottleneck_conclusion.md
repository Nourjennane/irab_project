# Bottleneck Conclusion (HARD)

Total error records analysed: **4339**.
Source: Phase 3-A on Gazelle (80 errors) + MASAQ (4259 errors), with corrected evaluator and Stanza UD dep features.

## Headline finding

**The dominant bottleneck of the frozen Phase 3-A system is annotation sparsity (T08), present in 4057 of 4339 errors (93.5%).** Only 282 errors (6.5%) have a complete gold (case + role + marker), and these are the only errors against which the model's true performance can be measured.

## Residual bottleneck after T08 stripping

Of the fully-observable errors (the model's true error set), the dominant categories are:

| primary tag | count | % residual |
|---|---:|---:|
| T01_morphology_failure | 56 | 19.9% |
| T17_semantic_role_confusion | 54 | 19.1% |
| T10_confidence_calibration_pathology | 49 | 17.4% |
| T15_coordination_ambiguity | 29 | 10.3% |
| T06_discourse_context_failure | 20 | 7.1% |

Residual top bottleneck (after sparsity): **T01_morphology_failure** at 19.9% of observable errors.

## Semantic pressure

Errors needing semantic reasoning (score ≥ 2): **974 (22.4%)**.

## Highest-leverage next-gen interventions (top 5)

| rank | intervention | priority score |
|---:|---|---:|
| 1 | more_annotation | 676.2 |
| 2 | more_data | 617.5 |
| 3 | better_syntax | 488.8 |
| 4 | larger_model | 393.6 |
| 5 | reasoning | 227.9 |

## Hard answer to 'what is the dominant bottleneck'

Looking at the empirical decomposition:

1. **Surface-level dominator (T08 annotation sparsity, 93.5% of errors).** This is the *measurement* bottleneck — the evaluator cannot decide whether the model is right or wrong for nearly every MASAQ word, because gold prose lacks a full (case, role, marker) triple. Without addressing this, *no* next-generation experiment will produce a reliable signal.

2. **True model-side bottleneck (residual after T08 stripping).** Among the small subset of errors where gold is complete, the picture is dominated by 
   **T01_morphology_failure**, **T17_semantic_role_confusion**, and **T10_confidence_calibration_pathology**.

3. **Conclusion.** The frozen Phase 3-A system is **ceiling-bound by annotation completeness on the MASAQ evaluation surface**. The residual model-side bottleneck is mixed: local-syntax + calibration + nested-clause structure — addressable through more annotated data, richer treebank coverage, and reasoning supervision. Larger backbones and more inference-side reasoning (both ruled out by the frozen-baseline case study) are *not* the next lever.

## Direct mapping to next-gen Steps

Empirically driven priority for the nextgen branch:

- **Step 1 (data engine) and richer Layer C/D annotation** address T08 directly. **Highest priority.**
- **Step 4 (grammar graph) + Step 5 (long-context) + Step 11 (discourse)** address the residual long-range / nested / discourse error families.
- **Step 7 (curriculum) + Step 13 (eval v2)** ensure the added supervision is measured cleanly per construction.
- **Step 6 (backbone benchmark)** is justified mostly by the comparison-matrix contribution, not by an expectation that scale alone will help — the frozen baseline's null result tempers expectations.