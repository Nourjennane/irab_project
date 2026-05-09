# Error Taxonomy — Drives All Future Development

**Step 16 of the next-generation branch. Living document.**

Before training any next-gen system, we build a full error
taxonomy on the frozen baseline. This taxonomy categorises the
remaining failure modes and is used to:

1. Stratify evaluation v2 (Step 13)
2. Target the data engine's annotation priorities (Step 1)
3. Define the curriculum stages (Step 7)
4. Prioritise the construction schemas (Step 3)
5. Anchor the research logs (Step 15) with shared categories

## Methodology

For each held-out failed prediction in
{Gazelle test, MASAQ test, hard-construction subsets} we tag:

- the failure category (one of the 10 below)
- the construction family the word belongs to
- the calibration class (high-conf-wrong / low-conf-wrong / other)
- the parser confidence at the affected node

Categories are not mutually exclusive; a single error often falls
under 2–3.

## Top-level categories

### 1. Long-range failure

The correct prediction requires information from a non-adjacent
word (≥3 hops in the dep tree, or another sentence).

Examples:
- adjective agreement across intervening modifiers
- case assignment by a remote governor (preposition before a
  iḍāfa chain)
- pronoun antecedent across sentences

Frozen-baseline prevalence: high on MASAQ Quranic, moderate on
Gazelle.

### 2. Nested clause failure

Embedded nominal or verbal clauses where the model assigns a flat
matrix-clause role to a word that is part of a nested-clause
structure.

Frozen-baseline marker: kāna with multi-word khabar (الجو رياحه شديدة);
the model assigns *khabar_kana* to رياحه instead of recognising the
embedded mubtadaʾ-khabar nominal clause.

### 3. Semantic failure

Two parses are syntactically identical and only semantics
disambiguates.

Examples:
- *ḥāl* vs *naʿt* (state vs property)
- *istithnāʾ munqaṭiʿ* vs *muttaṣil* (disconnected vs connected
  exception)
- *mafʿūl liʾajlih* vs *mafʿūl muṭlaq* (purpose vs paronymous object)

Frozen-baseline prevalence: persistent on rare constructions.

### 4. Ambiguity failure

Multiple defensible parses exist; the model picks one without
flagging the alternatives. The §5.4 annotator-disagreement audit
showed 31% of Gazelle words admit at least one alternative.

Frozen-baseline coverage: none — predictions are single-best,
no ambiguity surfacing.

### 5. Discourse failure

Cross-sentence context required:

- referential pronoun resolution
- topic-continuation case effects
- rhetorical-relation effects on argument roles

Frozen-baseline coverage: none — single-sentence predictions only.

### 6. Parser failure

The Stanza UD parser (UAS ≈ 84%) gives wrong head/label, the
predictions inherit the error. Detected by parser confidence +
post-hoc audit.

Frozen-baseline mitigation: 70% has_dep coverage; 30% sentences
fall back to morph-only path.

### 7. Annotation sparsity

Gold doesn't have the label; the model's correct prediction is
scored as wrong (or vice versa). The kāna evaluator fix exposed
this for kana_sisters Gazelle (4/6 had `gold role = None`); other
cells likely have similar gaps.

Mitigation lane: data engine richer annotation (Step 1 Layer C+D).

### 8. Rare-construction collapse

Construction with very few training examples; the head defaults
to a dominant class.

Frozen-baseline failures: istithnāʾ 0% Gazelle fully, quranic_proxy
0% Gazelle fully. Both correlate with corpus undercoverage.

Mitigation lane: targeted hand-annotated examples (~500 per
failing family; user said to *not* re-attempt template synthesis).

### 9. Confidence pathology

Model is highly confident (>0.9) on a wrong prediction. Inverse
of the calibration goal.

Frozen-baseline marker: kāna calibration gap −0.533 before
evaluator fix, −0.189 after.

### 10. Calibration mismatch

Aggregate: confidence on wrong > confidence on right. Currently
−0.101 on Gazelle, +0.048 on MASAQ for Phase 3-A.

## How this drives the next-gen design

| Category | Module that addresses it |
|---|---|
| 1. Long-range | Step 5 long-context, Step 4 grammar graph |
| 2. Nested clause | Step 3 construction schemas, Step 4 graph clause hierarchy |
| 3. Semantic | Step 10 semantic reasoning |
| 4. Ambiguity | Step 8 decoding (surface alternatives), Step 13 ambiguity-robustness eval |
| 5. Discourse | Step 11 discourse reasoning |
| 6. Parser | Step 1 data engine (gold-treebank sources) |
| 7. Annotation sparsity | Step 1 data engine (Layer C+D coverage) |
| 8. Rare construction | Step 1 data engine (broader coverage) + Step 7 curriculum |
| 9. Confidence pathology | Step 8 decoding (calibration-aware reranking) |
| 10. Calibration | Step 13 evaluation, Step 8 decoding |

## Per-category sample size to target

For each category, the next-gen evaluation aims for
n ≥ 100 per category to admit paired statistical claims. The
frozen baseline's n=134 Gazelle is below this floor for any
construction-stratified analysis; data engine + eval_v2 must
produce sufficient n.

## Population pass

Initial population: walk the frozen-baseline error log on
Gazelle and MASAQ, tag each error with categories above, save to
`docs/error_analysis_v2/{gazelle,masaq}_errors_tagged.jsonl`.
This pass should run before any next-gen training begins; it
clarifies which categories are dominant and therefore which
modules (Step 3, 4, 5, 10, 11) deserve early implementation.

## Open questions

- Tagging methodology: human-only or LLM-assisted with human
  audit? The frozen baseline's annotator-disagreement audit
  used a single second annotator on a Gazelle subset; that is
  the baseline rigor.
- How are tag updates versioned across the lifetime of the
  next-gen branch (errors will be re-tagged after each major
  module ships)?
