# Per-construction eval — gazelle — model: runs/phase3a_491240/final

Total sentences: 30

Total word judgments: 107


## Construction prevalence

| Construction | Sentences containing |
|---|---:|
| kana_sisters | 2 (6.7%) |
| inna_sisters | 3 (10.0%) |
| istithna | 1 (3.3%) |
| mawsool | 0 (0.0%) |
| idafa | 12 (40.0%) |
| idafa_multi | 0 (0.0%) |
| quranic_proxy | 2 (6.7%) |
| overall | 30 (100.0%) |

## Per-construction accuracy

| Construction | n_words | case | role | marker | **fully** | calib gap |
|---|---:|---:|---:|---:|---:|---:|
| kana_sisters | 7 | 71.4 | 0.0 | 14.3 | **0.0** | -0.533 |
| inna_sisters | 11 | 90.9 | 27.3 | 72.7 | **27.3** | -0.132 |
| istithna | 5 | 80.0 | 40.0 | 20.0 | **0.0** | -0.291 |
| mawsool | 0 | — | — | — | — | — |
| idafa | 45 | 82.2 | 46.7 | 62.2 | **33.3** | -0.133 |
| idafa_multi | 0 | — | — | — | — | — |
| quranic_proxy | 8 | 37.5 | 12.5 | 62.5 | **0.0** | +0.245 |
| overall | 107 | 72.0 | 37.4 | 62.6 | **25.2** | -0.099 |

## Top role confusion patterns per construction

### idafa

| gold → pred | count |
|---|---:|
| `mudaaf_ilayh->mudaaf_ilayh` | 5 |
| `<none>->ism_kana` | 1 |
| `mudaaf_ilayh->khabar_kana` | 1 |
| `khabar->khabar_kana` | 1 |
| `mudaaf_ilayh->harf_jarr` | 1 |
| `fail->ism_kana` | 1 |
| `mudaaf_ilayh->fail` | 1 |
| `ism_inna->ism_inna` | 1 |
| `mudaaf_ilayh->khabar_inna` | 1 |
| `khabar_inna->khabar_inna` | 1 |

### istithna

| gold → pred | count |
|---|---:|
| `mudaaf_ilayh->mudaaf_ilayh` | 1 |

### kana_sisters

| gold → pred | count |
|---|---:|
| `<none>->ism_kana` | 2 |
| `ism_majrur->khabar_kana` | 1 |
| `mudaaf_ilayh->khabar_kana` | 1 |
| `khabar->khabar_kana` | 1 |

### quranic_proxy

| gold → pred | count |
|---|---:|
| `mudaaf_ilayh->fail` | 1 |

### inna_sisters

| gold → pred | count |
|---|---:|
| `mubtada->ism_inna` | 1 |
| `ism_inna->ism_inna` | 1 |
| `mudaaf_ilayh->khabar_inna` | 1 |
| `khabar_inna->khabar_inna` | 1 |
