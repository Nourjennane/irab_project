# Curriculum Learning (Step 7 of next-gen branch)

**Status:** scaffolded only. No implementation yet.

Staged training schedule; the model learns Arabic grammar progressively
rather than receiving a single mixed-distribution shuffle.

## Stages

| Stage | Focus | Typical input |
|---|---|---|
| 1 | morphology | individual words with morph tags |
| 2 | local syntax | short sentences with simple S+V+O / mubtadaʾ+khabar |
| 3 | simple constructions | iḍāfa, single kāna sister, illa-istithnāʾ |
| 4 | nested syntax | iḍāfa chains, multi-level clauses |
| 5 | semantic interactions | predicate-argument, agreement consistency |
| 6 | discourse-sensitive structures | pronoun resolution, topic continuation |
| 7 | Quranic / classical complexity | omitted elements, archaic patterns, balāghah |

## Requirements

- Each training sample carries a difficulty label (1..7) derived
  from the data engine's metadata (Step 2): construction families,
  nested-depth score, discourse complexity.
- Stage transitions are checkpoint-driven, not epoch-driven —
  a stage exits when held-out per-construction metrics for that
  difficulty plateau.
- Earlier stages are *not* dropped at later stages; they remain in
  the training mix at decreasing proportion (rehearsal).

## Open design questions

- How are stage proportions scheduled? Linear ramp vs step-function
  vs adaptive based on evaluation?
- Does each stage warrant separate evaluation gates, or is the
  ship gate cumulative?
- Interaction with multi-task supervision (morph, dep, role, marker,
  reasoning trace) — should each stage emphasise different heads?
