# Backbone Upgrade — Comparison Matrix

**Step 6 of the next-generation branch. Design only.**

The frozen baseline used AraT5v2-base (296M, MSA-pretrained).
The §5.2(a) capacity null result showed scaling AraT5v2 to AraGPT2-large
or AceGPT-13B on the same Haiku-distilled corpus produces no Gazelle
improvement (every pairwise McNemar p=1.000 at n=134). That null is
**teacher-bound**, not capacity-bound — the corpus ceiling is the
binding constraint, not the model.

The next-gen branch must therefore decouple the backbone choice
from the corpus question. We benchmark backbones on:

1. **Pretraining coverage** — does the backbone know classical /
   Quranic Arabic?
2. **Syntax probing** — how cleanly does it represent dep relations
   and construction structure?
3. **Long-context ability** — can it handle multi-sentence input
   for Step 11 discourse reasoning?
4. **Nested-construction performance** — frozen-baseline blind spot
   (idafa chains, embedded khabar, ambiguous attachment).

## Backbones to benchmark

| Family | Variant | Params | Pretraining | Why benchmark |
|---|---|---:|---|---|
| AraT5 | AraT5v2-base | 296M | MSA | frozen-baseline reference |
| AraT5 | AraT5v2-large | 770M | MSA | scale within same family on a richer corpus |
| AraBART | AraBART-base | 139M | multi-dialect | encoder-decoder, multi-dialect breadth |
| CAMeLBERT | CAMeLBERT-CA | 135M | classical Arabic | classical / Quranic specialisation |
| CAMeLBERT | CAMeLBERT-MSA | 135M | MSA | head-to-head with AraT5v2 on MSA |
| AraBERT | AraBERT-large-v02 | ~370M | MSA | encoder-only; baseline for Stanza-style head |
| Multilingual | mT5-base / mT5-large | 580M / 1.2B | mT5 | for the Sonnet-distilled comparison |
| Long-context | LongFormer-Arabic / Nystromformer | ~340M | (limited Arabic) | for Step 5 long-range reasoning |
| Instruction-tuned | Jais-13B-chat / AceGPT-13B-chat | 13B | Arabic+code | for Step 9 reasoning-trace generation |

## Comparison matrix

For each backbone, score on:

| Axis | How measured |
|---|---|
| Phase 1 transfer | retrain Phase 1 morph heads only; compare UD-PADT macro to frozen-baseline 98.4% |
| Phase 3 transfer | retrain Phase 3 dep features; compare Gazelle case + marker to frozen-baseline 56.7 / 44.8 |
| Phase 3-A overall | full retrain on identical corpus + recipe; compare Gazelle / MASAQ headlines |
| Syntax probing | UD DEPREL classification accuracy on UD-PADT held-out |
| Construction probing | per-construction accuracy on the 6-family eval (frozen baseline structure) |
| Long-context | nested-clause case accuracy on synthetic 4+ sentence inputs |
| Reasoning trace generation | quality of generated derivation chain (paired human eval, 50 sentences) |

## Cost estimate

Each Phase 3-A-style retrain on Bocconi HPC: ~30 min on 1×A100 / job
within the 4-h SLURM cap. Comparison matrix budget: ~10 backbones ×
3 retrain configs (Phase 1 / Phase 3 / Phase 3-A) = ~30 jobs ≈
~15 GPU-hours. Reasoning trace human eval: ~6 person-hours.

## Decision rule

A backbone replaces AraT5v2-base as the next-gen production base
**only if**:

1. Phase 3-A overall *fully* on Gazelle ≥ 25.2 (no regression from
   frozen baseline at the same corpus + recipe), AND
2. *fully* on MASAQ ≥ 14.9 + 1.0 (meaningful classical/Quranic
   gain), OR
3. construction-probing on the catastrophic Gazelle subsets
   (kana / istithnāʾ / quranic_proxy) shows ≥ +5 pp on at least
   two of the three.

If no candidate clears, the comparison matrix is itself the
contribution: a publishable Arabic backbone benchmarking study
on iʿrāb generation.

## Open questions

- How is the Sonnet-distilled corpus (mentioned in `docs/REPORT.md`
  §8) integrated into this matrix? Likely a separate corpus axis
  multiplied against backbones.
- Does the data engine (Step 1) precede the backbone benchmark, or
  do we use the frozen-baseline corpus to keep the comparison
  apples-to-apples? Likely the latter for decision rule, the former
  for the publishable benchmark.
