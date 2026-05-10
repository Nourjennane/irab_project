# Unified evaluation — both conventions, side-by-side

**Primary metric:** *paper convention* — denominator = `n_words`
(every Gazelle/MASAQ word judgment, including those where gold is
missing on a given axis — those count as wrong on that axis).

**Secondary diagnostic:** *fully-observable subset* — denominator =
`n_observable_fully` (tokens where all 3 gold fields are populated).

These are the same model on the same data; only the denominator
differs. The numerators (tokens correct on each axis) are unchanged.

## final_eval

### gazelle

**Paper convention (denominator = n_words):**

| checkpoint | n_words | case | role | marker | fully |
|---|---:|---:|---:|---:|---:|
| phase3a | 134 | 0.6045 | 0.3433 | 0.5000 | **0.2090** |
| stage7 | 134 | 0.6567 | 0.3731 | 0.4851 | **0.1716** |

**Fully-observable subset (denominator = n_observable_fully):**

| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |
|---|---:|---:|---:|---:|---:|
| phase3a | 61 | 0.6378 | 0.5750 | 0.6837 | **0.4590** |
| stage7 | 61 | 0.6929 | 0.6250 | 0.6633 | **0.3770** |

### masaq

**Paper convention (denominator = n_words):**

| checkpoint | n_words | case | role | marker | fully |
|---|---:|---:|---:|---:|---:|
| phase3a | 5007 | 0.8316 | 0.1552 | 0.3094 | **0.1346** |
| stage7 | 5007 | 0.9966 | 0.1995 | 0.4310 | **0.1993** |

**Fully-observable subset (denominator = n_observable_fully):**

| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |
|---|---:|---:|---:|---:|---:|
| phase3a | 999 | 0.8345 | 0.7778 | 0.7175 | **0.6747** |
| stage7 | 999 | 1.0000 | 1.0000 | 0.9995 | **0.9990** |

### ud_test

**Paper convention (denominator = n_words):**

| checkpoint | n_words | case | role | marker | fully |
|---|---:|---:|---:|---:|---:|
| phase3a | 28264 | 0.4344 | 0.0000 | 0.0000 | **0.0000** |
| stage7 | 28264 | 0.4370 | 0.0000 | 0.0000 | **0.0000** |

**Fully-observable subset (denominator = n_observable_fully):**

| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |
|---|---:|---:|---:|---:|---:|
| phase3a | 0 | 0.8197 | 0.0000 | 0.0000 | **0.0000** |
| stage7 | 0 | 0.8246 | 0.0000 | 0.0000 | **0.0000** |

## final_eval_governor

### gazelle

**Paper convention (denominator = n_words):**

| checkpoint | n_words | case | role | marker | fully |
|---|---:|---:|---:|---:|---:|
| governor | 134 | 0.6269 | 0.3582 | 0.5000 | **0.2090** |
| phase3a | 134 | 0.6045 | 0.3433 | 0.5000 | **0.2090** |
| recovery | 134 | 0.6119 | 0.3657 | 0.4776 | **0.2090** |

**Fully-observable subset (denominator = n_observable_fully):**

| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |
|---|---:|---:|---:|---:|---:|
| governor | 61 | 0.6614 | 0.6000 | 0.6837 | **0.4590** |
| phase3a | 61 | 0.6378 | 0.5750 | 0.6837 | **0.4590** |
| recovery | 61 | 0.6457 | 0.6125 | 0.6531 | **0.4590** |

### masaq

**Paper convention (denominator = n_words):**

| checkpoint | n_words | case | role | marker | fully |
|---|---:|---:|---:|---:|---:|
| governor | 5007 | 0.8416 | 0.1606 | 0.3048 | **0.1424** |
| phase3a | 5007 | 0.8316 | 0.1552 | 0.3094 | **0.1346** |
| recovery | 5007 | 0.8452 | 0.1610 | 0.3060 | **0.1418** |

**Fully-observable subset (denominator = n_observable_fully):**

| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |
|---|---:|---:|---:|---:|---:|
| governor | 999 | 0.8445 | 0.8048 | 0.7068 | **0.7137** |
| phase3a | 999 | 0.8345 | 0.7778 | 0.7175 | **0.6747** |
| recovery | 999 | 0.8481 | 0.8068 | 0.7096 | **0.7107** |

### ud_test

**Paper convention (denominator = n_words):**

| checkpoint | n_words | case | role | marker | fully |
|---|---:|---:|---:|---:|---:|
| governor | 28264 | 0.4220 | 0.0000 | 0.0000 | **0.0000** |
| phase3a | 28264 | 0.4344 | 0.0000 | 0.0000 | **0.0000** |
| recovery | 28264 | 0.4302 | 0.0000 | 0.0000 | **0.0000** |

**Fully-observable subset (denominator = n_observable_fully):**

| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |
|---|---:|---:|---:|---:|---:|
| governor | 0 | 0.7963 | 0.0000 | 0.0000 | **0.0000** |
| phase3a | 0 | 0.8197 | 0.0000 | 0.0000 | **0.0000** |
| recovery | 0 | 0.8118 | 0.0000 | 0.0000 | **0.0000** |

## final_eval_graph

### gazelle

**Paper convention (denominator = n_words):**

| checkpoint | n_words | case | role | marker | fully |
|---|---:|---:|---:|---:|---:|
| graph | 134 | 0.6045 | 0.3657 | 0.4776 | **0.2090** |
| phase3a | 134 | 0.6045 | 0.3433 | 0.5000 | **0.2090** |
| recovery | 134 | 0.6119 | 0.3657 | 0.4776 | **0.2090** |

**Fully-observable subset (denominator = n_observable_fully):**

| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |
|---|---:|---:|---:|---:|---:|
| graph | 61 | 0.6378 | 0.6125 | 0.6531 | **0.4590** |
| phase3a | 61 | 0.6378 | 0.5750 | 0.6837 | **0.4590** |
| recovery | 61 | 0.6457 | 0.6125 | 0.6531 | **0.4590** |

### masaq

**Paper convention (denominator = n_words):**

| checkpoint | n_words | case | role | marker | fully |
|---|---:|---:|---:|---:|---:|
| graph | 5007 | 0.8424 | 0.1622 | 0.3082 | **0.1410** |
| phase3a | 5007 | 0.8316 | 0.1552 | 0.3094 | **0.1346** |
| recovery | 5007 | 0.8452 | 0.1610 | 0.3060 | **0.1418** |

**Fully-observable subset (denominator = n_observable_fully):**

| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |
|---|---:|---:|---:|---:|---:|
| graph | 999 | 0.8453 | 0.8128 | 0.7147 | **0.7067** |
| phase3a | 999 | 0.8345 | 0.7778 | 0.7175 | **0.6747** |
| recovery | 999 | 0.8481 | 0.8068 | 0.7096 | **0.7107** |

### ud_test

**Paper convention (denominator = n_words):**

| checkpoint | n_words | case | role | marker | fully |
|---|---:|---:|---:|---:|---:|
| graph | 28264 | 0.4327 | 0.0000 | 0.0000 | **0.0000** |
| phase3a | 28264 | 0.4344 | 0.0000 | 0.0000 | **0.0000** |
| recovery | 28264 | 0.4302 | 0.0000 | 0.0000 | **0.0000** |

**Fully-observable subset (denominator = n_observable_fully):**

| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |
|---|---:|---:|---:|---:|---:|
| graph | 0 | 0.8165 | 0.0000 | 0.0000 | **0.0000** |
| phase3a | 0 | 0.8197 | 0.0000 | 0.0000 | **0.0000** |
| recovery | 0 | 0.8118 | 0.0000 | 0.0000 | **0.0000** |

## final_eval_recovery

### gazelle

**Paper convention (denominator = n_words):**

| checkpoint | n_words | case | role | marker | fully |
|---|---:|---:|---:|---:|---:|
| phase3a | 134 | 0.6045 | 0.3433 | 0.5000 | **0.2090** |
| recovery | 134 | 0.6119 | 0.3657 | 0.4776 | **0.2090** |

**Fully-observable subset (denominator = n_observable_fully):**

| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |
|---|---:|---:|---:|---:|---:|
| phase3a | 61 | 0.6378 | 0.5750 | 0.6837 | **0.4590** |
| recovery | 61 | 0.6457 | 0.6125 | 0.6531 | **0.4590** |

### masaq

**Paper convention (denominator = n_words):**

| checkpoint | n_words | case | role | marker | fully |
|---|---:|---:|---:|---:|---:|
| phase3a | 5007 | 0.8316 | 0.1552 | 0.3094 | **0.1346** |
| recovery | 5007 | 0.8452 | 0.1610 | 0.3060 | **0.1418** |

**Fully-observable subset (denominator = n_observable_fully):**

| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |
|---|---:|---:|---:|---:|---:|
| phase3a | 999 | 0.8345 | 0.7778 | 0.7175 | **0.6747** |
| recovery | 999 | 0.8481 | 0.8068 | 0.7096 | **0.7107** |

### ud_test

**Paper convention (denominator = n_words):**

| checkpoint | n_words | case | role | marker | fully |
|---|---:|---:|---:|---:|---:|
| phase3a | 28264 | 0.4344 | 0.0000 | 0.0000 | **0.0000** |
| recovery | 28264 | 0.4302 | 0.0000 | 0.0000 | **0.0000** |

**Fully-observable subset (denominator = n_observable_fully):**

| checkpoint | n_obs_fully | case (on n_obs_case) | role (on n_obs_role) | marker (on n_obs_marker) | fully |
|---|---:|---:|---:|---:|---:|
| phase3a | 0 | 0.8197 | 0.0000 | 0.0000 | **0.0000** |
| recovery | 0 | 0.8118 | 0.0000 | 0.0000 | **0.0000** |
