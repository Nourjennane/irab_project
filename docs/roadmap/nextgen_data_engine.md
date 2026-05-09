# Next-Gen Data Engine — Roadmap

**Step 1 of the next-generation branch. Design only.**

The frozen baseline trained on 77K Haiku-distilled word-rows from
~5K MSA sentences. The Phase 39 / Phase R / Phase R2 cycle showed
that distribution-mismatched data hurts and that information
coverage is the binding constraint at the current scale.

The next-gen system requires **multi-layer supervision** spanning
morphology / local syntax / deep grammatical role / explicit
reasoning, drawn from genuinely diverse sources, with rich
metadata so the curriculum (Step 7) can sequence training and the
evaluator (Step 13) can stratify results.

## Target scale

50K–100K+ supervised sentences minimum.

## Storage layout

```
data_v2/
  raw/              — unmodified source files (Stanford UD, MASAQ raw, etc.)
  processed/        — tokenised + aligned to whitespace, per source
  annotated/        — full Layer A+B+C+D supervision (case/role/marker/morph/dep/reasoning)
  reasoning/        — Layer D explicit reasoning traces
  treebanks/        — UD-PADT, CamelTB, Extended Quranic Treebank converted to common schema
  quranic/          — Quranic Arabic Corpus, MASAQ training half (held out from frozen-baseline test)
  classical/        — classical Arabic textbook excerpts
  msa_news/         — Gazelle-style MSA news
  educational/      — pedagogical exercises and exam solutions
  discourse/        — multi-sentence connected text for Step 11
```

## Layer A — Morphology (per-token)

| Field | Source |
|---|---|
| gender (masc / fem / und) | UD-PADT, MASAQ, MADAMIRA |
| number (sg / dual / plural / und) | UD-PADT, MASAQ |
| person (1 / 2 / 3) | morph parser |
| definiteness (def / indef / und) | surface + dep tree |
| POS (noun / verb / particle / pronoun / adjective / punctuation) | UD UPOS, distill_v2 |
| inflection class | UD features |
| agreement axes | computed pairwise |
| mood (indicative / subjunctive / jussive) | UD features |
| voice (active / passive) | UD features |
| derivation class (Form I-X) | morph parser, optional |

Frozen-baseline coverage for Layer A: 7 morph heads (gender, number,
definite, person, aspect, mood, voice). Next-gen extends the field
list and removes the “und” fallback by dual-sourcing.

## Layer B — Local syntax (per-token)

| Field | Source |
|---|---|
| dependency head index | Stanza UD parser, gold UD-PADT, treebanks |
| dependency label (DEPREL) | UD parser / treebank |
| constituency span (start, end) | converted from UD via Hwa-style algorithm |
| phrase type (NP / VP / PP / SBAR …) | constituency head |
| governor relation (case-assignment edge) | derived |
| attachment ambiguity score | parser confidence + alternative parses |

Frozen-baseline coverage for Layer B: DEPREL, HEAD direction +
distance, governor POS (Phase 3-A inputs). Next-gen extends to
constituency and explicit attachment-ambiguity tracking.

## Layer C — Deep grammatical role (per-token, canonical)

The schema's role list (frozen-baseline 25 labels, Phase 4a 34 labels)
extends to cover:

- مبتدأ، خبر — mubtadaʾ, khabar
- اسم كان، خبر كان — ism / khabar of kāna family
- اسم إن، خبر إن — ism / khabar of inna family
- حال — ḥāl (circumstantial accusative)
- تمييز — tamyīz (specifier)
- بدل — badal (apposition)
- مفعول مطلق، مفعول فيه، مفعول معه — paronymous, locative, comitative objects
- مستثنى — mustathnā (exception target)
- معطوف — coordinated noun
- نعت — naʿt (adjective modifier)
- توكيد — emphatic
- apposition (fine-grained)
- embedded clause roles (matrix-clause role of an embedded clause)
- omitted/implicit elements (ḍamīr mustatir, omitted mubtadaʾ)

Layer C is the legacy `role` axis of the frozen baseline, expanded
to handle the constructions the new architecture needs to reason
about.

## Layer D — Reasoning / explanation

Per-construction, per-decision:

- grammatical justification (what rule applies, in what order)
- derivation chain (which intermediate steps were taken)
- ambiguity discussion (what alternatives were considered, why kept or rejected)
- alternative parses (top-2 or top-3 ranked)
- confidence rationale (why this analysis is preferred)
- transformation logic (e.g. "kāna shifts case from rafʿ-of-mubtadaʾ
  to naṣb on its khabar")

Sources: Arabic textbooks (مغني اللبيب، شرح ابن عقيل), exam-solution
corpora, iʿrāb websites with structured rationales (e.g.
mawdoo3.com / arabic-tools.net pages with full analyses), Quranic
iʿrāb references (e.g. الدر المصون).

Layer D has **no** equivalent in the frozen baseline.

## Per-sentence metadata

Every sentence carries:

- `domain` (msa_news / quranic / classical / educational / pedagogical)
- `source` (corpus name + ID)
- `annotation_quality` (gold human / silver automated / bronze
  inferred)
- `parser_confidence` (Stanza UAS or treebank-source flag)
- `construction_families` (multiset of construction types in the
  sentence)
- `difficulty_level` (1..7, mapping to curriculum stages)
- `sentence_length` (token count)
- `nested_depth_score` (max embedding depth of clauses)
- `discourse_complexity` (number of cross-sentence references)

Difficulty level is computed from the metadata and used by the
curriculum module (Step 7).

## Source candidates and their layer coverage

| Source | A morph | B dep | C role | D reasoning | License |
|---|:---:|:---:|:---:|:---:|---|
| UD Arabic-PADT | ✓ gold | ✓ gold | partial (UD DEPREL≠iʿrāb role) | ✗ | open |
| CamelTB (188K words) | ✓ | ✓ | ✓ via CATiB→iʿrāb conversion | ✗ | research |
| MASAQ (123K Quranic) | ✓ | ✓ | ✓ | partial (templater) | research |
| Extended Quranic Treebank (132K) | ✓ | ✓ | ✓ | ✗ | open |
| I3rab MSA (601 sentences) | ✓ | ✓ | ✓ | partial | open |
| Sonnet-distilled corpus (proposed) | ✓ via Stanza | ✓ via Stanza | ✓ | partial | API budget |
| Educational exam-solutions (web) | ✗ | ✗ | ✓ | ✓ | manual scrape + rights review |
| iʿrāb websites with rationale | ✗ | ✗ | ✓ | ✓ | manual scrape + rights review |
| Classical textbook excerpts | ✗ | ✗ | ✓ | ✓ | rights review |

## Pipeline stages (Step 1 implementation order)

1. **Schema definition** — formalise Layer A/B/C/D + metadata in
   a dataclass in `src/irab_tashkeel/data/schema_v2.py`.
2. **Source ingestion** — per-source loader scripts under
   `data_v2/raw/<source>/load.py`, normalised to `processed/`.
3. **Cross-source schema alignment** — UD ↔ CATiB ↔ MASAQ ↔
   distill_v2 mapping table.
4. **Difficulty / metadata scorer** — compute per-sentence metadata.
5. **Annotated split** — emit `data_v2/annotated/{train,val,test}.jsonl`.
6. **Reasoning extraction** — separately, build
   `data_v2/reasoning/<construction>.jsonl` from textbook /
   exam-solution data.

## Open questions

- License + ethics review for each web-scraped source.
- Cross-register train/test splitting strategy — keep MASAQ test
  held out as in the frozen baseline, but use MASAQ training half.
- Annotation quality auditing — how to score `annotation_quality`
  consistently across sources?
- Should the reasoning-trace data be mixed into the annotated
  split, or kept as a separate supervision signal that only
  some heads see?
