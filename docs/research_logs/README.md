# Research Logs (Step 15 of next-gen branch)

Every experiment on the `nextgen-grammatical-reasoning` branch
records here BEFORE running. The log entry is the contract between
hypothesis and execution; uncontrolled experimentation is not
allowed on this branch.

## File naming

`docs/research_logs/<exp_id>_<short_name>.md`

- `<exp_id>` — monotonic integer (start at 001 for the first
  experiment of this branch)
- `<short_name>` — kebab-case slug

## Required fields per entry

```markdown
# Experiment <exp_id> — <short_name>

**Date:** YYYY-MM-DD
**Branch state:** <git rev-parse HEAD>
**Stage:** <curriculum stage 1..7 or "infra">

## Hypothesis
What do we expect to learn / change?

## Expected mechanism
Why would this work? What signal does it add or remove?

## Setup
- Backbone:
- Corpus:
- Recipe:
- Eval:

## Metrics tracked
Specific cells we will compute, including per-construction
breakdowns and calibration.

## Pre-registered decision rule
What metric(s) and threshold(s) will trigger ship / archive /
iterate?

## Result
[filled after experiment]

## Error analysis
Tagged categories from `docs/error_taxonomy.md`. Per-construction
behaviour. Calibration changes. Distribution effects.

## Failure modes
[only if relevant]

## Verdict
ship / archive / iterate

## Pointer to artefact
- log file:
- checkpoint:
- per-construction summary:
```

## Discipline

- No experiment runs without a log entry committed.
- Decision rules are pre-registered, not chosen post-hoc.
- A failed experiment is a successful learning event when its
  failure mode is correctly tagged in the error taxonomy.
- Re-running the same experiment with different settings opens
  a new entry; log entries are immutable once committed.
