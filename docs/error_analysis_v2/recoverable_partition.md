# Recoverable vs Unrecoverable Partition

Total error records: 4339

## Bucket distribution (single bucket per error, from primary tag)

| bucket | count | % | recoverable? |
|---|---:|---:|:---:|
| annotation_limited | 4057 | 93.5% | ❌ |
| morphology | 56 | 1.3% | ✓ |
| role_semantics | 54 | 1.2% | ✓ |
| calibration | 49 | 1.1% | ✓ |
| coordination | 29 | 0.7% | ✓ |
| discourse | 20 | 0.5% | ✓ |
| syntax_local | 16 | 0.4% | ✓ |
| syntax_long_range | 15 | 0.3% | ✓ |
| construction_overlap | 15 | 0.3% | ✓ |
| nested_clause | 6 | 0.1% | ✓ |
| implicit_governor | 4 | 0.1% | ✓ |
| evaluator_limited | 4 | 0.1% | ❌ |
| retrieval_mismatch | 2 | 0.0% | ✓ |
| omitted_element | 2 | 0.0% | ✓ |
| fundamental_ambiguity | 1 | 0.0% | ❌ |

## Intervention applicability (multiple per error allowed)

| intervention | applicable to | % errors |
|---|---:|---:|
| more_annotation | 4057 | 93.5% |
| larger_model | 162 | 3.7% |
| more_data | 152 | 3.5% |
| reasoning | 91 | 2.1% |
| better_syntax | 85 | 2.0% |
| semantic_supervision | 57 | 1.3% |
| discourse_supervision | 20 | 0.5% |
| better_evaluator | 4 | 0.1% |

## Recoverable headline

- Recoverable (some intervention applicable): **268 (6.2%)**
- Unrecoverable (annotation- or evaluator-limited or fundamentally ambiguous): **4062 (93.6%)**