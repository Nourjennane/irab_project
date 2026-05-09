# Semantic Reasoning (Step 10 of next-gen branch)

**Status:** scaffolded only. No implementation yet.

Many Arabic grammatical decisions hinge on semantics, not just
syntax — most visibly in *istithnāʾ munqaṭiʿ* (disconnected
exception) where the case of the mustathnā depends on whether the
exception is from the same semantic class as the mustathnā minhu.

## Capabilities to implement

- semantic-role interaction (event-level argument structure
  on top of the syntactic role taxonomy)
- predicate-argument structure (PropBank-style for Arabic, or
  a lightweight Arabic-specific semantic-role layer)
- clause meaning tracking (what a clause is *about*, not just
  its grammatical structure)
- semantic compatibility scoring (does word X plausibly have role
  Y given the predicate)
- event structure modelling (durative / punctual / perfective
  aspect interacting with the verbal forms)

## Why semantics matters here

- istithnāʾ munqaṭiʿ vs muttaṣil disambiguation
- distinguishing *ḥāl* (circumstantial accusative) from *naʿt*
  (descriptive adjective): morphologically identical, semantically
  different (state of action vs property of noun)
- omitted-mubtadaʾ reconstruction depends on what a clause is
  semantically about
- pronoun antecedent selection often requires semantic plausibility

## Required interfaces (TBD)

- `score_role_plausibility(predicate, arg, role)` — semantic
  compatibility score
- `event_structure(clause)` — durative / punctual classification
- `semantic_class(word)` — PropBank-style predicate frame or
  lightweight class tag

## Open design questions

- Use existing Arabic PropBank / SALMA-style resources, or
  bootstrap a lightweight class system from the dependency
  representation?
- How does semantic information feed back into the iʿrāb
  prediction — soft conditioning (which the frozen baseline
  showed to plateau under joint training) or hard rules (the
  classical iʿrāb literature)?
