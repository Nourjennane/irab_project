# Error Analysis v2

Tagged error logs from the frozen baseline and from each
next-gen experiment. Drives `docs/error_taxonomy.md` and
prioritises the data-engine annotation queue (Step 1).

## Initial population task

Walk the frozen-baseline Gazelle + MASAQ test errors, tag each
with categories from `docs/error_taxonomy.md`, save:

- `gazelle_errors_tagged.jsonl`
- `masaq_errors_tagged.jsonl`

This produces the empirical category histogram that decides
which next-gen modules (Step 3 / 4 / 5 / 10 / 11) deserve early
implementation.
