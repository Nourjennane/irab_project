# Ceiling Analysis (T08-stripped Residual)

Total errors:                              4339
Errors carrying T08 (annotation sparsity): 4057 (93.5%)
Errors with FULL gold (case+role+marker):  282 (6.5%)

## Per-field gold availability

- gold case   present: 4318 / 4339
- gold role   present: 297 / 4339
- gold marker present: 1471 / 4339

## Residual error tag histogram (T08-stripped, on fully-observable errors)

This is the model's TRUE remaining error structure once the annotation-sparsity dominator is removed.

| tag | count | % of residual errors |
|---|---:|---:|
| T01_morphology_failure | 109 | 38.7% |
| T02_local_syntax_failure | 149 | 52.8% |
| T03_long_range_dependency_failure | 17 | 6.0% |
| T04_nested_clause_failure | 6 | 2.1% |
| T05_semantic_ambiguity | 1 | 0.4% |
| T06_discourse_context_failure | 21 | 7.4% |
| T07_parser_alignment_failure | 0 | 0.0% |
| T08_annotation_sparsity | 0 | 0.0% |
| T09_rare_construction_collapse | 2 | 0.7% |
| T10_confidence_calibration_pathology | 69 | 24.5% |
| T11_retrieval_mismatch | 3 | 1.1% |
| T12_evaluator_limitation | 4 | 1.4% |
| T13_implicit_governor_failure | 4 | 1.4% |
| T14_omitted_element_reasoning | 2 | 0.7% |
| T15_coordination_ambiguity | 33 | 11.7% |
| T16_clause_attachment_ambiguity | 0 | 0.0% |
| T17_semantic_role_confusion | 106 | 37.6% |
| T18_construction_overlap_interference | 16 | 5.7% |

## Residual primary tag distribution

| primary tag | count | % of residual errors |
|---|---:|---:|
| T01_morphology_failure | 56 | 19.9% |
| T17_semantic_role_confusion | 54 | 19.1% |
| T10_confidence_calibration_pathology | 49 | 17.4% |
| T15_coordination_ambiguity | 29 | 10.3% |
| T06_discourse_context_failure | 20 | 7.1% |
| T02_local_syntax_failure | 16 | 5.7% |
| T03_long_range_dependency_failure | 15 | 5.3% |
| T18_construction_overlap_interference | 15 | 5.3% |
| T04_nested_clause_failure | 6 | 2.1% |
| T13_implicit_governor_failure | 4 | 1.4% |
| T12_evaluator_limitation | 4 | 1.4% |
| T11_retrieval_mismatch | 2 | 0.7% |
| T14_omitted_element_reasoning | 2 | 0.7% |
| T05_semantic_ambiguity | 1 | 0.4% |