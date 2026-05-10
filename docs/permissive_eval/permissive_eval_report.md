# Permissive Evaluation — Heuristic Auto-Annotation

## Methodology

- Mined ambiguity candidates: **4233** across 7 ambiguity kinds (see `data_v2/ambiguity_corpus/`)
- Auto-marked as `BOTH_VALID` if gold and predicted role are both in the **surface-ambiguous role family**: ['badal', 'fail', 'ism_inna', 'ism_majrur', 'khabar', 'khabar_inna', 'khabar_kana', 'mafoul_bih', 'matuf', 'mubtada', 'mudaaf_ilayh', 'naat']
- Kept: **4233 candidates** (100.0% of mined; 486 unique sentences)

**Caveat.** This is a heuristic placeholder for the human grammarian's pass. A real annotator would reject many of these — e.g., when an overt verb governs the noun, the *mafoul_bih* reading is unambiguous despite surface compatibility with *mudaaf_ilayh*. Treat the permissive delta below as an upper bound.

## Strict baseline (no permissive scoring)

| metric | value |
|---|---:|
| n_words (fully-observable) | 1060 |
| case_acc | 0.8368 |
| role_acc | 0.7943 |
| marker_em | 0.7613 |
| **fully** | **0.6962** |
| calib_gap | 0.1112 |

## Permissive eval

| metric | value |
|---|---:|
| total tokens | 1060 |
| strict-correct | 738 |
| permissive-correct | 900 |
| tokens flagged ambiguous | 311 |
| ambiguous tokens resolved | 162 |
| **strict_fully** | **0.6962** |
| **permissive_fully** | **0.8491** |
| Δ (permissive − strict) | **+0.1528** |
| ambiguity_resolved_acc | 0.5209 |
