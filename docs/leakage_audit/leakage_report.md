# Phase B — Leakage Audit

Audited train sources: ['distill_v2', 'ud_padt_train', 'ud_padt_dev']
Against test sources: ['gazelle_test', 'masaq_quranic', 'ud_padt_test']

## Pairwise overlap counts

| train | test | n_train | n_test | exact | normalised | hash_dup | fuzzy_J≥0.7 | exact_% | norm_% | fuzzy_% |
|---|---|---|---|---|---|---|---|---|---|---|
| distill_v2 | gazelle_test | 11382 | 30 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| distill_v2 | masaq_quranic | 11382 | 624 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| distill_v2 | ud_padt_test | 11382 | 680 | 17 | 21 | 21 | 65 | 2.5 | 3.088 | 9.559 |
| ud_padt_train | gazelle_test | 6075 | 30 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| ud_padt_train | masaq_quranic | 6075 | 624 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| ud_padt_train | ud_padt_test | 6075 | 680 | 16 | 16 | 16 | 45 | 2.353 | 2.353 | 6.618 |
| ud_padt_dev | gazelle_test | 909 | 30 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| ud_padt_dev | masaq_quranic | 909 | 624 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| ud_padt_dev | ud_padt_test | 909 | 680 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| gazelle_test | masaq_quranic | 30 | 624 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| gazelle_test | ud_padt_test | 30 | 680 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| masaq_quranic | ud_padt_test | 624 | 680 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 |

## Interpretation

- `exact`: identical raw sentence text
- `normalised`: identical after stripping diacritics + tatweel + punctuation
- `hash_dup`: identical sha1 of normalised form (sanity check)
- `fuzzy_J≥0.7`: token-Jaccard ≥ 0.7 (likely paraphrase or near-dup)

Any non-zero exact or normalised count between train and test is a **direct leakage red flag** and must be cleaned before claiming the trained model's metrics. Fuzzy overlaps need manual review — see `suspicious_examples.jsonl`.
