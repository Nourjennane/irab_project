# Curriculum-Ready Corpus Report

Built from `data_v2/annotated/`. Total sentences: 19700.

Each row is one curriculum stage. Stages with `n=0` lack source coverage and the curriculum scheduler must either down-weight them or trigger targeted annotation.

## Stage overview

| stage | n_sentences | n_unique_families | avg_dep_depth | avg_sem_pressure | avg_completeness | avg_length |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2672 | 0 | 0.01 | 0.01 | 0.06 | 16 |
| 2 | 2221 | 0 | 5.74 | 0.76 | 0.17 | 18 |
| 3 | 1046 | 3 | 3.88 | 0.53 | 0.21 | 25 |
| 4 | 387 | 4 | 3.48 | 0.69 | 0.47 | 27 |
| 5 | 12661 | 7 | 3.56 | 2.00 | 0.38 | 35 |
| 6 | 588 | 7 | 0.00 | 3.00 | 1.00 | 18 |
| 7 | 125 | 7 | 0.00 | 1.68 | 0.56 | 10 |

## Per-stage source breakdown

| stage | distill_v2 | ud_padt_train | ud_padt_dev | ud_padt_test | masaq_quranic | gazelle_test |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2418 | 40 | 0 | 0 | 208 | 6 |
| 2 | 0 | 1779 | 250 | 192 | 0 | 0 |
| 3 | 416 | 425 | 57 | 31 | 108 | 9 |
| 4 | 209 | 129 | 17 | 7 | 23 | 2 |
| 5 | 7751 | 3702 | 585 | 450 | 160 | 13 |
| 6 | 588 | 0 | 0 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 | 125 | 0 |

## Per-stage family distribution (top families)

### Stage 1 (n=2672)
| family | count |
|---|---:|

### Stage 2 (n=2221)
| family | count |
|---|---:|

### Stage 3 (n=1046)
| family | count |
|---|---:|
| inna_sisters | 728 |
| kana_sisters | 206 |
| idafa | 112 |

### Stage 4 (n=387)
| family | count |
|---|---:|
| inna_sisters | 468 |
| idafa | 408 |
| kana_sisters | 125 |
| idafa_multi | 93 |

### Stage 5 (n=12661)
| family | count |
|---|---:|
| mawsool | 17689 |
| idafa | 11468 |
| inna_sisters | 5079 |
| idafa_multi | 2223 |
| quranic_proxy | 1926 |
| kana_sisters | 1915 |
| istithna | 951 |

### Stage 6 (n=588)
| family | count |
|---|---:|
| idafa | 1614 |
| mawsool | 469 |
| idafa_multi | 288 |
| inna_sisters | 101 |
| kana_sisters | 92 |
| quranic_proxy | 47 |
| istithna | 37 |

### Stage 7 (n=125)
| family | count |
|---|---:|
| mawsool | 127 |
| idafa | 103 |
| inna_sisters | 73 |
| istithna | 34 |
| kana_sisters | 21 |
| quranic_proxy | 15 |
| idafa_multi | 8 |
