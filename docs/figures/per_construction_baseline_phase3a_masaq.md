# Per-construction eval — masaq — model: runs/phase3a_491240/final

Total sentences: 624

Total word judgments: 5007


## Construction prevalence

| Construction | Sentences containing |
|---|---:|
| kana_sisters | 32 (5.1%) |
| inna_sisters | 141 (22.6%) |
| istithna | 60 (9.6%) |
| mawsool | 0 (0.0%) |
| idafa | 170 (27.2%) |
| idafa_multi | 8 (1.3%) |
| quranic_proxy | 56 (9.0%) |
| overall | 624 (100.0%) |

## Per-construction accuracy

| Construction | n_words | case | role | marker | **fully** | calib gap |
|---|---:|---:|---:|---:|---:|---:|
| kana_sisters | 299 | 86.6 | 12.0 | 31.1 | **11.0** | -0.063 |
| inna_sisters | 1440 | 85.3 | 15.0 | 33.9 | **13.5** | +0.031 |
| istithna | 602 | 88.2 | 14.4 | 29.4 | **12.5** | +0.084 |
| mawsool | 0 | — | — | — | — | — |
| idafa | 1606 | 86.4 | 26.8 | 41.0 | **24.2** | +0.050 |
| idafa_multi | 47 | 89.4 | 44.7 | 48.9 | **29.8** | +0.045 |
| quranic_proxy | 565 | 88.5 | 13.3 | 26.0 | **11.3** | +0.049 |
| overall | 5007 | 85.8 | 17.1 | 33.0 | **14.9** | +0.048 |

## Top role confusion patterns per construction

### idafa

| gold → pred | count |
|---|---:|
| `mudaaf_ilayh->mudaaf_ilayh` | 198 |
| `<none>->ism_inna` | 45 |
| `<none>->khabar_inna` | 12 |
| `<none>->khabar_kana` | 4 |
| `mudaaf_ilayh->fail` | 4 |
| `mudaaf_ilayh->mafoul_bih` | 4 |
| `ism_majrur->mudaaf_ilayh` | 4 |
| `mudaaf_ilayh->ism_inna` | 4 |
| `mudaaf_ilayh->ism_majrur` | 3 |
| `<none>->mafoul_other` | 3 |

### inna_sisters

| gold → pred | count |
|---|---:|
| `<none>->ism_inna` | 62 |
| `mudaaf_ilayh->mudaaf_ilayh` | 53 |
| `<none>->khabar_inna` | 20 |
| `<none>->khabar_kana` | 5 |
| `ism_majrur->mudaaf_ilayh` | 3 |
| `mafoul_bih->khabar_kana` | 3 |
| `mudaaf_ilayh->ism_inna` | 3 |
| `ism_majrur->khabar_inna` | 3 |
| `mafoul_bih->ism_inna` | 2 |
| `<none>->mafoul_other` | 2 |

### istithna

| gold → pred | count |
|---|---:|
| `mudaaf_ilayh->mudaaf_ilayh` | 19 |
| `<none>->ism_inna` | 17 |
| `<none>->khabar_inna` | 9 |
| `ism_majrur->mudaaf_ilayh` | 3 |
| `mudaaf_ilayh->ism_majrur` | 2 |
| `<none>->mudaaf_ilayh` | 2 |
| `mudaaf_ilayh->fail` | 1 |
| `<none>->khabar_kana` | 1 |
| `<none>->mafoul_other` | 1 |
| `<none>->ism_kana` | 1 |

### quranic_proxy

| gold → pred | count |
|---|---:|
| `<none>->ism_inna` | 18 |
| `mudaaf_ilayh->mudaaf_ilayh` | 14 |
| `<none>->khabar_inna` | 2 |
| `ism_majrur->mudaaf_ilayh` | 2 |
| `<none>->khabar_kana` | 2 |
| `<none>->mafoul_other` | 1 |
| `mafoul_bih->khabar_kana` | 1 |
| `mudaaf_ilayh->mafoul_bih` | 1 |

### kana_sisters

| gold → pred | count |
|---|---:|
| `<none>->ism_inna` | 13 |
| `mudaaf_ilayh->mudaaf_ilayh` | 11 |
| `<none>->khabar_kana` | 6 |
| `<none>->ism_kana` | 2 |
| `mudaaf_ilayh->ism_inna` | 1 |
| `ism_majrur->mudaaf_ilayh` | 1 |
| `mafoul_bih->ism_inna` | 1 |

### idafa_multi

| gold → pred | count |
|---|---:|
| `mudaaf_ilayh->mudaaf_ilayh` | 17 |
| `mudaaf_ilayh->ism_majrur` | 1 |
| `<none>->mafoul_other` | 1 |
| `<none>->ism_inna` | 1 |
| `<none>->khabar_inna` | 1 |
