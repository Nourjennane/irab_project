# Ranked Bottleneck Report — Step 16 Error Taxonomy

Source: Phase 3-A on Gazelle (80 errors) + MASAQ (4259 errors).

## Bottleneck classes (% of all errors carrying tag)

| rank | category | % errors | severity score |
|---:|---|---:|---:|
| 1 | T08_annotation_sparsity | 93.5% | 2319 |
| 2 | T02_local_syntax_failure | 46.9% | 1645 |
| 3 | T10_confidence_calibration_pathology | 45.7% | 946 |
| 4 | T06_discourse_context_failure | 15.8% | 1134 |
| 5 | T15_coordination_ambiguity | 14.8% | 353 |
| 6 | T04_nested_clause_failure | 14.3% | 363 |
| 7 | T03_long_range_dependency_failure | 10.3% | 282 |
| 8 | T01_morphology_failure | 6.0% | 272 |
| 9 | T18_construction_overlap_interference | 3.5% | 59 |
| 10 | T17_semantic_role_confusion | 2.5% | 222 |
| 11 | T14_omitted_element_reasoning | 2.2% | 119 |
| 12 | T13_implicit_governor_failure | 0.3% | 62 |
| 13 | T11_retrieval_mismatch | 0.3% | 18 |
| 14 | T09_rare_construction_collapse | 0.2% | 11 |
| 15 | T12_evaluator_limitation | 0.1% | 12 |
| 16 | T05_semantic_ambiguity | 0.0% | 2 |
| 17 | T07_parser_alignment_failure | 0.0% | 0 |
| 18 | T16_clause_attachment_ambiguity | 0.0% | 0 |

## Solvability classification

| category | realistic-data | better-syntax | larger-models | reasoning-supervision | annotation-limited |
|---|:---:|:---:|:---:|:---:|:---:|
| T08_annotation_sparsity | ✓ |   |   |   | ✓ |
| T02_local_syntax_failure | ✓ | ✓ | ✓ |   |   |
| T10_confidence_calibration_pathology | ✓ |   |   | ✓ |   |
| T06_discourse_context_failure |   |   | ✓ | ✓ |   |
| T15_coordination_ambiguity | ✓ | ✓ |   |   |   |
| T04_nested_clause_failure |   | ✓ |   | ✓ |   |
| T03_long_range_dependency_failure |   | ✓ | ✓ | ✓ |   |
| T01_morphology_failure | ✓ |   | ✓ |   |   |
| T18_construction_overlap_interference |   | ✓ |   | ✓ |   |
| T17_semantic_role_confusion |   |   | ✓ | ✓ |   |
| T14_omitted_element_reasoning |   |   |   | ✓ |   |
| T13_implicit_governor_failure |   | ✓ |   | ✓ |   |
| T11_retrieval_mismatch | ✓ |   |   |   |   |
| T09_rare_construction_collapse | ✓ |   | ✓ |   |   |
| T12_evaluator_limitation |   |   |   |   | ✓ |
| T05_semantic_ambiguity |   |   | ✓ | ✓ |   |
| T07_parser_alignment_failure | ✓ | ✓ |   |   |   |
| T16_clause_attachment_ambiguity | ✓ | ✓ |   | ✓ |   |

## Aggregate addressability (tag-incidence weighted)

- Realistic with more data:        8997 (80.9%)
- Better syntax / parser quality:  3910 (35.2%)
- Larger / better encoders:        3542 (31.8%)
- Reasoning supervision:           4103 (36.9%)
- Annotation-limited (ceiling):    4061 (36.5%)
