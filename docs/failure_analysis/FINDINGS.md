# Failure Analysis — Central Finding: the *mudaaf_ilayh* Confusion Family

> The single most important linguistic finding of the project.

## TL;DR

The validated production model (`runs/validated_nextgen_recovery/`)
makes its most consistent, most stubborn errors on a tightly-defined
family of role confusions centred on **mudaaf_ilayh** (the second
member of an idafa construct). Three architectural attacks
(structural recovery patch, graph integration, governor head) all
left this family essentially unchanged. The bottleneck is
**lexical-semantic** (verb-argument knowledge), not **structural**.

## 1 · The dominant role confusions on the held-out set

(`validated_nextgen_recovery` evaluated on full Gazelle + MASAQ;
fully-observable subset.)

| Gold | Predicted | Count | Family |
|---|---|---:|---|
| **mudaaf_ilayh** | **mafoul_bih** | **32** | I (idafa partner ↔ direct object) |
| **mudaaf_ilayh** | **mubtada** | **29** | I (idafa partner ↔ subject) |
| ism_majrur | matuf | 21 | II (preposition object ↔ coordinator object) |
| mudaaf_ilayh | fail | 13 | I |
| mudaaf_ilayh | ism_majrur | 13 | III (idafa partner ↔ preposition object) |
| mafoul_bih | fail | 12 | semantic role overlap |
| ism_majrur | mubtada | 8 | II |
| mudaaf_ilayh | ism_inna | 7 | I (nested) |

Family I (mudaaf_ilayh confusions) accounts for **~120 errors** in
the held-out set — by far the largest single block.

## 2 · Why these confusions are linguistically genuine

The three "hard" labels — *mudaaf_ilayh*, *mafoul_bih*, *ism_majrur* —
all surface as a noun in **jarr** (genitive) case after another word.

| When the second noun is *mudaaf_ilayh* | the first noun governs it (idafa) |
| When the second noun is *mafoul_bih* | a verb upstream governs it |
| When the second noun is *ism_majrur* | a preposition governs it |

A model relying purely on **adjacent surface form + dep-tree
attachment** has no reliable signal to choose between these readings:

- The dep parent is the same (the head noun / verb / preposition).
- The case is the same (jarr).
- The marker is the same (kasra).

The decision requires **lexical-semantic knowledge**: does the upstream
verb take a direct object? Does the second noun fit the verb's
argument structure? Is the first noun a typical "head of idafa" word
(possessor, container, agent)?

Neither dep features nor explicit governor prediction supplies this
information at our data scale.

## 3 · Why architecture cannot fix this alone

We tested two architectural attacks targeted exactly at this family:

| Attack | Idea | Result on the confusion family |
|---|---|---|
| Graph refiner | Pass messages between tokens along dep + construction edges | *Same counts* — see [`docs/final_graph_negative_result/`](../final_graph_negative_result) |
| Governor head | Biaffine prediction of governor token + attachment contrastive | *Same counts* — see [`docs/final_governor_negative_result/`](../final_governor_negative_result) |

Both architectural experiments trained correctly (gradients flowed,
auxiliary losses converged, ablation deltas were positive on
training-time slices). Neither displaced the confusion on the
held-out set.

This convergent negative evidence is the project's main scientific
finding.

## 4 · Why we believe the bottleneck is lexical-semantic

A *mudaaf_ilayh* / *mafoul_bih* / *ism_majrur* decision becomes
trivial when the model knows:

- **What the upstream verb requires.** Verbs that take a direct object
  set the next noun as *mafoul_bih*; verbs that don't allow the
  next noun to be *mudaaf_ilayh* of a head noun.
- **Which head nouns canonically take an idafa partner.** *kitāb*
  ("book") nearly always heads an idafa; *qara'a* ("read") nearly
  never does.

This is **lexical knowledge**, not structural knowledge. A larger
gold corpus annotated with explicit verb-argument structure or with
permissive `alternative_valid_analyses` would shift this ceiling
in a way no graph or governor head can.

## 5 · Calibration on the confusion family

The model is also confidently wrong on these confusions:

| Axis | ECE on failures | High-conf wrong (≥0.95) |
|---|---|---|
| case | 0.42 | 79 tokens |
| role | 0.49 | 83 tokens |
| marker | 0.60 | 70 tokens |

In the [0.9, 1.0) confidence bin — where the model says "I'm 90%+
sure" — per-axis accuracy is only ~0.50 (case), ~0.37 (role), ~0.29
(marker). For an educational/audit-friendly system this is
unacceptable; temperature scaling on a held-out shard would be the
first remedy.

## 6 · Structural breakdown — fully accuracy by family

(Validated recovery on Gazelle + MASAQ fully-observable.)

| Construction family | n | n_correct | fully |
|---|---:|---:|---:|
| inna_sisters | 268 | 188 | 0.702 |
| istithna | 106 | 76 | 0.717 |
| idafa | 521 | 333 | **0.639** |
| **idafa_multi** (nested) | 22 | 4 | **0.182** |

Single-construction idafa is at 0.639 — already below the
inna/istithna level. Nested-idafa collapses to 0.182. The pattern is
the same: more idafa structure → more confusion.

## 7 · Where this finding redirects the project

- **No more architecture experiments** until the lexical-semantic
  supervision layer is built.
- **The annotation queue at [`data_v2/ambiguity_corpus/`](../../data_v2/ambiguity_corpus)
  is the highest-leverage next investment.** It contains 4,233
  mined candidate ambiguities sliced by ambiguity kind:

  | Kind | Candidates |
  |---|---:|
  | latent_governor | 990 |
  | nested_attachment | 912 |
  | idafa_attachment | 684 |
  | semantic_role_overlap | 622 |
  | preposition_vs_idafa | 530 |
  | coordination_scope | 495 |

- **The annotation server is deployable** (`src/irab_tashkeel/annotation/annotation_server.py`).
- **The permissive evaluator is wired** (`src/irab_tashkeel/eval_v3/ambiguity_metrics.py`).
- **Sample efficiency tooling is wired** (`src/irab_tashkeel/active_learning/`).

When annotated data lands, `eval_v3.evaluate_with_ambiguity` will
score predictions permissively — a *mudaaf_ilayh* prediction on a
genuinely ambiguous token will count as correct iff the alternative
analysis says it could be *mudaaf_ilayh*. This single change
plausibly moves Gazelle role +0.05 to +0.10 *without changing the model*.

## 8 · Reproducing this analysis

```bash
PYTHONPATH=src python scripts/analysis/run_failure_analysis.py \
    --checkpoint runs/validated_nextgen_recovery \
    --datasets gazelle_test masaq_quranic \
    --out_dir docs/failure_analysis/
```

Outputs: `top_failures.md`, `role_confusions.md`, `case_confusions.md`,
`marker_confusions.md`, `long_range_failures.md`, `nested_clause_failures.md`,
`overlap_failures.md`, `calibration_failures.md`,
`structural_breakdown.md`, `summary.json`.
