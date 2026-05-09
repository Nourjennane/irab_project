# Failure Dependency Graph

Total error records: 4339

## Strongest co-occurring pairs (Jaccard ≥ 0.20)

| tag A | tag B | count | Jaccard | PMI |
|---|---|---:|---:|---:|
| T09_rare_construction_collapse | T11_retrieval_mismatch | 7 | 0.583 | +5.89 |
| T08_annotation_sparsity | T10_confidence_calibration_pathology | 1913 | 0.464 | +0.03 |
| T02_local_syntax_failure | T08_annotation_sparsity | 1887 | 0.449 | -0.01 |
| T02_local_syntax_failure | T10_confidence_calibration_pathology | 887 | 0.283 | -0.05 |
| T02_local_syntax_failure | T06_discourse_context_failure | 470 | 0.209 | +0.38 |

## Highest-PMI pairs (statistical association strength)

| tag A | tag B | count | Jaccard | PMI |
|---|---|---:|---:|---:|
| T09_rare_construction_collapse | T11_retrieval_mismatch | 7 | 0.583 | +5.89 |
| T05_semantic_ambiguity | T17_semantic_role_confusion | 1 | 0.009 | +3.69 |
| T12_evaluator_limitation | T17_semantic_role_confusion | 3 | 0.028 | +3.41 |
| T11_retrieval_mismatch | T13_implicit_governor_failure | 1 | 0.040 | +3.25 |
| T09_rare_construction_collapse | T18_construction_overlap_interference | 2 | 0.013 | +2.10 |
| T11_retrieval_mismatch | T17_semantic_role_confusion | 2 | 0.017 | +1.90 |
| T09_rare_construction_collapse | T17_semantic_role_confusion | 1 | 0.009 | +1.75 |
| T11_retrieval_mismatch | T18_construction_overlap_interference | 2 | 0.012 | +1.56 |
| T01_morphology_failure | T11_retrieval_mismatch | 3 | 0.011 | +1.43 |
| T06_discourse_context_failure | T14_omitted_element_reasoning | 49 | 0.067 | +1.17 |
| T03_long_range_dependency_failure | T12_evaluator_limitation | 1 | 0.002 | +0.89 |
| T01_morphology_failure | T09_rare_construction_collapse | 1 | 0.004 | +0.87 |
| T03_long_range_dependency_failure | T15_coordination_ambiguity | 143 | 0.151 | +0.77 |
| T17_semantic_role_confusion | T18_construction_overlap_interference | 8 | 0.032 | +0.75 |
| T12_evaluator_limitation | T15_coordination_ambiguity | 1 | 0.002 | +0.52 |
| T06_discourse_context_failure | T12_evaluator_limitation | 1 | 0.001 | +0.46 |
| T10_confidence_calibration_pathology | T18_construction_overlap_interference | 102 | 0.050 | +0.38 |
| T02_local_syntax_failure | T06_discourse_context_failure | 470 | 0.209 | +0.38 |
| T15_coordination_ambiguity | T17_semantic_role_confusion | 23 | 0.032 | +0.36 |
| T02_local_syntax_failure | T14_omitted_element_reasoning | 64 | 0.031 | +0.35 |
| T03_long_range_dependency_failure | T18_construction_overlap_interference | 20 | 0.035 | +0.25 |
| T03_long_range_dependency_failure | T04_nested_clause_failure | 80 | 0.081 | +0.23 |
| T01_morphology_failure | T02_local_syntax_failure | 153 | 0.071 | +0.23 |
| T02_local_syntax_failure | T04_nested_clause_failure | 339 | 0.146 | +0.15 |
| T02_local_syntax_failure | T15_coordination_ambiguity | 339 | 0.145 | +0.12 |
| T10_confidence_calibration_pathology | T13_implicit_governor_failure | 7 | 0.004 | +0.09 |
| T15_coordination_ambiguity | T18_construction_overlap_interference | 24 | 0.031 | +0.06 |
| T04_nested_clause_failure | T08_annotation_sparsity | 614 | 0.151 | +0.06 |
| T08_annotation_sparsity | T14_omitted_element_reasoning | 94 | 0.023 | +0.05 |
| T06_discourse_context_failure | T08_annotation_sparsity | 664 | 0.163 | +0.04 |

## Independence (low Jaccard, high count)

Tags that appear often but rarely together (suggest independence).

| tag A | tag B | count | Jaccard |
|---|---|---:|---:|
| T08_annotation_sparsity | T09_rare_construction_collapse | 5 | 0.001 |
| T08_annotation_sparsity | T11_retrieval_mismatch | 9 | 0.002 |
| T08_annotation_sparsity | T13_implicit_governor_failure | 10 | 0.002 |
| T10_confidence_calibration_pathology | T13_implicit_governor_failure | 7 | 0.004 |
| T06_discourse_context_failure | T18_construction_overlap_interference | 12 | 0.015 |
| T03_long_range_dependency_failure | T14_omitted_element_reasoning | 8 | 0.015 |
| T06_discourse_context_failure | T17_semantic_role_confusion | 12 | 0.015 |
| T01_morphology_failure | T06_discourse_context_failure | 15 | 0.016 |
| T14_omitted_element_reasoning | T15_coordination_ambiguity | 12 | 0.017 |
| T01_morphology_failure | T17_semantic_role_confusion | 6 | 0.017 |
| T10_confidence_calibration_pathology | T14_omitted_element_reasoning | 39 | 0.019 |
| T01_morphology_failure | T18_construction_overlap_interference | 8 | 0.020 |
| T01_morphology_failure | T15_coordination_ambiguity | 19 | 0.021 |
| T01_morphology_failure | T10_confidence_calibration_pathology | 50 | 0.023 |
| T08_annotation_sparsity | T14_omitted_element_reasoning | 94 | 0.023 |
| T02_local_syntax_failure | T17_semantic_role_confusion | 49 | 0.023 |
| T01_morphology_failure | T03_long_range_dependency_failure | 17 | 0.025 |
| T04_nested_clause_failure | T18_construction_overlap_interference | 20 | 0.027 |
| T04_nested_clause_failure | T06_discourse_context_failure | 38 | 0.030 |
| T02_local_syntax_failure | T14_omitted_element_reasoning | 64 | 0.031 |