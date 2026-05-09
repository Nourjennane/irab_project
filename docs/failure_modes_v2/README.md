# Failure Modes v2

Structured per-experiment failure-mode catalogue. Each entry
documents *how* a next-gen experiment failed, not just *that* it
did. The frozen baseline's R/R2 cycle is the model: each
intervention (Phase 4a, 2, 5, 6, 3.1, 39, R-C, R2) produced a
characterised failure mode, not a vague regression.

## Required fields per failure-mode entry

```markdown
# <exp_id> failure mode

## What was attempted

## What we expected vs what happened

## Diagnosis (root cause)
- Mechanism: …
- Distribution: …
- Calibration: …
- Construction breakdown: …

## Lessons
What this rules out for future experiments.

## Tag
Category from docs/error_taxonomy.md.
```

The point of cataloguing failure modes explicitly is to ensure
the same mistake is not made on a different module.
