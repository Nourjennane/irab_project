# Loader Interface — Adding a New Source Corpus

This document is the operational contract for anyone adding a
new corpus to the data_v2 engine.

## 1. Subclass `BaseLoader`

```python
# src/irab_tashkeel/data_v2/loaders/<my_corpus>.py
from typing import Any, Dict, Iterator, Optional
from ..schema_v2 import (
    AnnotationQuality, Domain, LabelTag, Sentence, Token,
    AnnotationCompleteness,
)
from .base import BaseLoader, register_loader


@register_loader
class MyCorpusLoader(BaseLoader):
    source_id          = "my_corpus_v1"
    domain             = Domain.MSA_NEWS.value
    annotation_quality = AnnotationQuality.SILVER_LLM_DISTILL.value
    parser_origin      = "stanza_ud"
    license            = "MIT"

    def iter_raw(self) -> Iterator[Dict[str, Any]]:
        # yield raw records from your source
        ...

    def normalize_row(self, raw: Dict[str, Any], idx: int) -> Optional[Sentence]:
        # construct a Sentence from one raw record
        ...
```

## 2. Loader responsibilities

Every loader **must**:

1. Set the four class attributes (`source_id`, `domain`,
   `annotation_quality`, `parser_origin`).
2. Use `arabic_normalize` from `data_v2.normalization` for
   `Token.normalized` and `Sentence.normalized_text`.
3. Use `LabelTag(value=None)` for unannotated fields — never
   `LabelTag(value="")`.
4. Set `LabelTag.confidence` to a real value, not the default 1.0,
   when the source is parser-derived.
5. Populate `AnnotationCompleteness` accurately (which layers
   actually have content).
6. Set `Sentence.metadata` via `self._make_metadata(source_id_within=...)`
   so the timestamp + provenance are uniform.

Loaders **must not**:

- Compute curriculum metadata (that's a separate post-pass in
  `data_v2.metadata.difficulty.populate_metadata`).
- Generate UUIDs for sentences manually (use `new_id`).
- Mutate global state.

## 3. Provenance source strings

Use only canonical strings from `docs/data_v2/SCHEMA_V2.md` §6.
If your source needs a new provenance, add it to that table first.

## 4. Quality tiers

Pick the most-conservative tier that still describes the source:

- Hand-annotated by an expert → `GOLD_HUMAN`
- Published treebank → `GOLD_TREEBANK`
- Distilled from teacher LLM → `SILVER_LLM_DISTILL`
- Parser at ≥ 0.8 confidence → `SILVER_PARSER_HIGH_CONF`
- Parser at < 0.8 confidence → `BRONZE_PARSER_LOW_CONF`
- Pure rule-based detector → `BRONZE_HEURISTIC`

Mixed-quality corpora should split into multiple loaders or use
per-token `LabelTag.source` values to capture the variability.

## 5. Test contract

Every loader must add a smoke test in
`tests/test_data_v2_loaders.py` that:

- imports the loader (so it registers)
- calls `load_all()` on a small subset
- asserts `len(sentences) > 0`
- asserts `sentences[0].metadata.source == loader.source_id`
- asserts `sentences[0].schema_version == SCHEMA_VERSION`
- asserts `sentences[0].metadata.annotation_quality` is in the
  defined tier set

## 6. Pipeline expected by callers

```python
from data_v2.loaders.my_corpus import MyCorpusLoader
from data_v2.metadata import difficulty

loader = MyCorpusLoader(root="/path/to/repo")
sentences = loader.load_all()                  # raw → schema_v2
difficulty.populate_all(sentences)             # add curriculum metadata
# → write to data_v2/annotated/<source>/{train,val,test}.jsonl
```

Loaders themselves never split into train/val/test — that's a
later pipeline pass that uses metadata + difficulty for stratified
sampling.
