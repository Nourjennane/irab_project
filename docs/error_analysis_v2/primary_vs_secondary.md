# Primary vs Secondary Failure Decomposition

Total error records analysed: 4339

## Primary tag distribution (one per error)

| primary tag | count | % errors |
|---|---:|---:|
| T08_annotation_sparsity | 4057 | 93.5% |
| T01_morphology_failure | 56 | 1.3% |
| T17_semantic_role_confusion | 54 | 1.2% |
| T10_confidence_calibration_pathology | 49 | 1.1% |
| T15_coordination_ambiguity | 29 | 0.7% |
| T06_discourse_context_failure | 20 | 0.5% |
| T02_local_syntax_failure | 16 | 0.4% |
| T03_long_range_dependency_failure | 15 | 0.3% |
| T18_construction_overlap_interference | 15 | 0.3% |
| T04_nested_clause_failure | 6 | 0.1% |
| T13_implicit_governor_failure | 4 | 0.1% |
| T12_evaluator_limitation | 4 | 0.1% |
| T11_retrieval_mismatch | 2 | 0.0% |
| T14_omitted_element_reasoning | 2 | 0.0% |
| T05_semantic_ambiguity | 1 | 0.0% |

## Top primary→secondary cascades

Pairs of (primary tag, secondary tag) with co-occurrence count, interpreted as 'primary tag ROOT-CAUSED a downstream tag'.

| primary | secondary | count |
|---|---|---:|
| T08_annotation_sparsity | T10_confidence_calibration_pathology | 1913 |
| T08_annotation_sparsity | T02_local_syntax_failure | 1887 |
| T08_annotation_sparsity | T06_discourse_context_failure | 664 |
| T08_annotation_sparsity | T04_nested_clause_failure | 614 |
| T08_annotation_sparsity | T15_coordination_ambiguity | 610 |
| T08_annotation_sparsity | T03_long_range_dependency_failure | 428 |
| T08_annotation_sparsity | T01_morphology_failure | 151 |
| T08_annotation_sparsity | T18_construction_overlap_interference | 136 |
| T08_annotation_sparsity | T14_omitted_element_reasoning | 94 |
| T01_morphology_failure | T02_local_syntax_failure | 42 |
| T10_confidence_calibration_pathology | T02_local_syntax_failure | 29 |
| T10_confidence_calibration_pathology | T01_morphology_failure | 26 |
| T15_coordination_ambiguity | T02_local_syntax_failure | 26 |
| T17_semantic_role_confusion | T02_local_syntax_failure | 22 |
| T15_coordination_ambiguity | T17_semantic_role_confusion | 21 |
| T06_discourse_context_failure | T17_semantic_role_confusion | 11 |
| T08_annotation_sparsity | T13_implicit_governor_failure | 10 |
| T08_annotation_sparsity | T11_retrieval_mismatch | 9 |
| T06_discourse_context_failure | T02_local_syntax_failure | 9 |
| T06_discourse_context_failure | T01_morphology_failure | 7 |
| T18_construction_overlap_interference | T01_morphology_failure | 7 |
| T18_construction_overlap_interference | T17_semantic_role_confusion | 7 |
| T03_long_range_dependency_failure | T10_confidence_calibration_pathology | 6 |
| T08_annotation_sparsity | T09_rare_construction_collapse | 5 |
| T15_coordination_ambiguity | T01_morphology_failure | 5 |
| T06_discourse_context_failure | T10_confidence_calibration_pathology | 4 |
| T03_long_range_dependency_failure | T17_semantic_role_confusion | 4 |
| T04_nested_clause_failure | T02_local_syntax_failure | 3 |
| T04_nested_clause_failure | T17_semantic_role_confusion | 3 |
| T04_nested_clause_failure | T01_morphology_failure | 3 |
| T12_evaluator_limitation | T17_semantic_role_confusion | 3 |
| T17_semantic_role_confusion | T10_confidence_calibration_pathology | 3 |
| T08_annotation_sparsity | T17_semantic_role_confusion | 2 |
| T11_retrieval_mismatch | T17_semantic_role_confusion | 2 |
| T06_discourse_context_failure | T15_coordination_ambiguity | 2 |
| T14_omitted_element_reasoning | T02_local_syntax_failure | 2 |
| T15_coordination_ambiguity | T10_confidence_calibration_pathology | 2 |
| T13_implicit_governor_failure | T10_confidence_calibration_pathology | 2 |
| T03_long_range_dependency_failure | T01_morphology_failure | 2 |
| T18_construction_overlap_interference | T10_confidence_calibration_pathology | 2 |