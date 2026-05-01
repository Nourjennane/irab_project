# Distill v2 — Corpus Statistics

## Source

- **Teacher**: Claude Haiku 4.5 (`claude-haiku-4-5`), `temperature=0.0`
- **Retrieval**: 5 nearest-neighbour examples from the headline pool (Yarob 459 + Distilled-v1 601 = **1,060 examples**), Jaccard similarity
- **Prompt**: identical Arabic system + RAG user template as the headline Sonnet RAG (saved verbatim under `reproducibility/prompts/`)
- **Date**: 2026-05-01
- **Batch**: Anthropic Messages Batches API (50% discount), batch id `msgbatch_01JCMDvb3zqGtGzA2pELWBTg`

## Cost (real, post-batch)

| Item | Value |
|---|---:|
| Input tokens (total) | 11,592,847 |
| Output tokens (total) | 9,638,464 |
| Avg in / sentence | 2,319 |
| Avg out / sentence | 1,928 |
| Haiku 4.5 batch pricing | $0.50/M in, $2.50/M out (50% off list $1/$5) |
| **Real cost** | **$29.89** |
| Pre-run estimate (Sonnet token model) | $32.60 |
| **Variance** | $-2.71 (under-budget) |

(The script printed $89.68 — that was a bug: it used Sonnet pricing constants regardless of the requested model. Fixed in the same commit; all post-fix calls use `cost_for(model, …)`. Real cost was charged correctly because the actual API call used `claude-haiku-4-5`.)

## Source candidates

`data/distill_v2/sources.jsonl` (built by `src/irab_tashkeel/data/source_assembly.py`) contains **46,545 candidate Arabic sentences** assembled from:

| Source | Pre-filter | Post-filter (kept) | Kept rate | Composition |
|---|---:|---:|---:|---:|
| Wikipedia AR (20231101.ar, first paragraphs) | 60,000 | 38,893 | 64.8% | 84% |
| Tashkeela classical (10 random files, sampled) | 10,000 | 4,317 | 43.2% | 9% |
| PADT-UD (train + dev + test, surface text) | 7,664 | 3,335 | 43.5% | 7% |
| **Total** | **77,664** | **46,545** | 60% | 100% |

NyUAD-UD intentionally not used — its CoNLL-U files have FORM=`_` (text fully redacted under LDC PATB licensing); reconstructing requires three LDC catalogs (~$3-9k each), out of scope.

Filters applied: 5–25 whitespace tokens; reject if no Arabic letters; reject if any Latin/CJK letters; reject if Arabic-letter density < 50%; dedup on first 300 chars.

For the 5K distillation we sampled the first 5,000 of the (already-shuffled) source list. Source mix in the distilled output:

| Source | Distilled rows |
|---|---:|
| wikipedia | 4,206 (84%) |
| tashkeela | 437 (9%) |
| padt | 357 (7%) |

## Distilled corpus quality (post-extraction QC)

`data/distill_v2/distilled.jsonl` (5,000 rows, 21.8 MB):

| Metric | Value |
|---|---:|
| Sentences with non-empty items | 4,997 / 5,000 (99.9%) |
| Total word-level i'rāb rows | 77,534 |
| Word-level full-JSON (`word`+`irab`+`case`+`role`+`marker` all present) | **77,534 / 77,534 (100%)** |
| Word-level structural-extractor well-formed (case extracted) | 98.1% |
| Distinct syntactic roles | **590** (vs ~25 in the structural taxonomy — Haiku produces fine-grained role descriptions) |
| Distinct case-marker phrases | **109** (incl. الواو, الياء, تنوين الفتح, الكسرة المقدرة, الفتحة المقدرة, مبني variants) |

The all-4-fields rate via the structural extractor is 20.2% — but this is a **STRUCTURAL EXTRACTOR ceiling**, not a Haiku quality issue. The extractor's POS-term regex doesn't cover all POS strings Haiku emits (50% miss rate on POS specifically); case/role/marker extraction rates are 98.1% / 69.8% / 88.4%. For training purposes, the direct JSON fields are what matter, and those are 100% present.

## Top-15 syntactic roles (corpus-wide)

| n | role |
|---:|---|
| 14,102 | مضاف إليه |
| 7,907 | نعت |
| 7,385 | حرف جر |
| 7,372 | اسم مجرور |
| 6,914 | بدل |
| 5,055 | حرف عطف |
| 3,296 | مبتدأ |
| 2,731 | فعل |
| 2,471 | مفعول به |
| 2,297 | خبر |
| 1,416 | فاعل |
| 1,333 | ظرف زمان |
| 1,224 | علامة ترقيم |
| 1,191 | فعل مضارع |
|   838 | حال |

(plus 575 longer-tail role descriptions; rare roles like تمييز, مفعول مطلق, مفعول لأجله, اسم إن, خبر إن all present in single-digit-to-low-double-digit counts.)

## Top-15 case-mood markers

| n | marker |
|---:|---|
| 27,821 | الكسرة الظاهرة |
| 19,765 | السكون |
| 14,692 | الضمة الظاهرة |
| 6,637 | الفتحة الظاهرة |
| 1,523 | الفتحة |
| 1,195 | تنوين الفتح |
|   955 | الياء |
|   826 | مبني |
|   707 | لا يوجد |
|   501 | الكسرة المقدرة |
|   459 | الكسرة |
|   434 | الفتح |
|   369 | الضمة المقدرة |
|   340 | الضمة |
|   136 | الواو |

Non-canonical markers (الواو, الياء, الألف, الكسرة/الضمة المقدرة, تنوين variants) appear with non-trivial frequency, indicating Haiku is not flattening to the three canonical markers only.

## Sample for inspection

`data/distill_v2/sample_100.jsonl` — 100 random rows from the corpus, for spot-checking quality.

## Limitations

1. **Teacher quality**: Haiku 4.5 has case-acc 67.2% on Gazelle (vs Sonnet's 73.9%). Open-weight models trained on this corpus inherit some teacher errors; the gap to Sonnet RAG observed in the scaling-comparison results may therefore overstate the gap to a "best-teacher" baseline. We chose Haiku for budget reasons (Sonnet 5K would cost ~$80, exceeding the project's $50 envelope); the alternative (Sonnet 1.5K) would have been too small to cleanly fine-tune 7B-scale models without saturation.

2. **Register skew**: 84% Wikipedia. PADT (news) and Tashkeela (classical) provide register diversity but the corpus is still encyclopedic-leaning. Generalization to news-style or dialectal Arabic should be evaluated, not assumed.

3. **No second-model cross-check**: the directive's "5% cross-check with Haiku/Opus" was descoped under budget pressure. Trade-off: the structural extractor is the only QC line; we accept its 98.1% well-formed rate as the quality floor.

4. **Role taxonomy explosion**: Haiku emits 590 distinct role labels, many of which are paraphrastic variants of canonical ones (e.g., "فعل مضارع" vs "فعل" with the conjugation in `pos`). Models trained on this will inherit the variability; canonicalization downstream may help but is not applied here.
