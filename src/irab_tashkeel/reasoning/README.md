# Explanation Supervision (Step 9 of next-gen branch)

**Status:** scaffolded only. No implementation yet.

The model should eventually emit explicit grammatical reasoning
traces alongside its predictions. This module owns the schema,
training data, and decoder for those traces.

## Reasoning trace schema

For each word + construction, the trace contains:

- the canonical grammatical *justification* (e.g. "ism kāna
  because it follows the kāna-family particle ʿasā with marfūʿ
  case")
- a *derivation chain* (which rule fired, in which order)
- *ambiguity discussion* (which alternative analyses were
  considered and why each was rejected or kept)
- *alternative interpretations* (top-2 or top-3 parses, ranked)
- *confidence rationale* (why is the model confident or
  uncertain)
- *transformation logic* for constructions (e.g. "khabar of
  kāna shifts case from rafʿ-of-mubtadaʾ to naṣb")

## Examples (target output)

```
هذا اسم كان مرفوع لأنه:
  - يلي الفعل الناقص (كان)
  - مفرد، مذكر
  - معرف بـ"ال"
  - علامة الرفع: الضمة الظاهرة
الجملة الاسمية في محل نصب خبر كان لأنها:
  - تعقب اسم كان
  - تشكل وحدة دلالية متكاملة
  - جزء التركيب الإسنادي
الاستثناء منقطع لأن المستثنى ليس من جنس المستثنى منه
```

## Training data

- `data_v2/reasoning/` populated by the data engine (Step 2),
  with reasoning traces parsed from textbooks, exam solutions,
  and iʿrāb websites.
- Each row aligns to a sentence in `data_v2/annotated/` and
  carries the structured reasoning trace plus a free-form gloss.

## Decoder

Two design options:

1. **Structured decoder** — a separate seq2seq head conditioned
   on the predicted parse, generating the trace from a closed
   template vocabulary.
2. **Late-fusion LLM** — the parse + graph feeds an Arabic LLM
   (CAMeLBERT / AraBART) prompted to produce the trace.

The frozen baseline's lessons argue against introducing more
trainable downstream heads (Phase 5/6 pattern); option 2 is the
default starting point.

## Open design questions

- How is reasoning trace correctness *measured*? Token-level
  accuracy is too brittle. Likely a structured comparison
  (which justification rules fired vs which were expected).
- Should the reasoning trace influence the parse prediction at
  decode time (joint), or is it a post-hoc explanation only?
