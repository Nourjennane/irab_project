# Graph integration — documented negative result

## What we built

End-to-end wiring of the existing grammar graph + 2-layer edge-aware
graph refiner into the `DepAwareStructuredModel` forward path:

- Collator emits `(B, W, W)` dense `word_edge_index` matrix from
  dep heads + construction membership + overlap detection
- Forward applies `pooled = pooled + sigmoid(graph_gate) * delta`
  where `delta = refiner(pooled, edge_index, mask) − pooled`
- Gate logit initialised at −2.0 (sigmoid ≈ 0.119) so the graph
  signal starts weak; the model learns whether structure helps. Critical
  for avoiding catastrophic degradation and oversmoothing on small data.
- Encoder frozen for first 2,000 steps; refiner + gate train alone.
- Edge dropout 15% on dep + construction edges; per-stage edge-type
  curriculum (dep only → +construction → +overlap → all).
- Eval emits `fully_with_graph` / `fully_without_graph` /
  `graph_edge_ablation_delta` and the live `graph_gate_alpha`.

## What worked

- Refiner trained without instability (no NaN, no norm explosion).
- Gate moved 0.120 → 0.122 — small but real movement once the
  encoder unfroze and could co-train.
- Ablation delta was consistently positive after stage 3:
  +0.006 to +0.013 fully on the cap-100 eval slice.
- Stage transitions and the no-leakage assertions both held.

## What did not work

On the **full uncapped held-out sets**, the graph checkpoint did not
beat the regularization-only `validated_recovery`.

**Paper convention (denominator = n_words):**

| Dataset | metric | recovery | graph | Δ |
|---|---|---|---|---|
| Gazelle (n=134) | fully | 0.209 | 0.209 | +0.000 |
| Gazelle | role  | 0.366 | 0.366 | +0.000 |
| Gazelle | case  | 0.612 | 0.604 | −0.008 |
| MASAQ (n=5,007) | fully | 0.142 | 0.141 | −0.001 |
| MASAQ | role | 0.161 | 0.162 | +0.001 |
| MASAQ | case  | 0.845 | 0.843 | −0.002 |

**Fully-observable subset (n=61 / 999):**

| Dataset | metric | recovery | graph | Δ |
|---|---|---|---|---|
| Gazelle | fully | 0.459 | 0.459 | +0.000 |
| Gazelle | role  | 0.613 | 0.613 | +0.000 |
| MASAQ   | fully | 0.711 | 0.707 | −0.004 |
| MASAQ   | role  | 0.807 | 0.813 | +0.006 |

The training-time +0.013 ablation delta did not survive the
full-sample eval. Gains and regressions are all within the noise band
on either denominator.

## Interpretation

At ~20k training sentences, explicit graph reasoning on top of the
dependency-aware encoder features (Phase 3-A's input-side dep
augmentation) does not materially improve unseen generalization.
The encoder + dep-feature input augmentation already captures most
of the structural signal a downstream graph layer would provide;
adding a refiner on top therefore yields no measurable headroom at
this scale.

**This is a bottleneck-identification result.** The remaining gap to
higher unseen performance is not architectural — it is **supervision
density and semantic-ambiguity coverage**.

## Why this is documented, not deleted

The implementation is correct, the experiment is informative, and
the negative finding constrains the search space for future work.
Anyone who tries the same architectural tweak in this regime will
reproduce our result. Documenting the experiment keeps the project
honest and saves future research effort.

## Where to find the data

- Training: `runs/nextgen_graph/` on HPC (deleted from disk after
  freeze; eval traces preserved here)
- Independent eval shards: `docs/final_eval_graph/raw/`
- Aggregated report: `docs/final_eval_graph/final_eval_report.md`
- Frozen artefact on HPC: `runs/final_graph_negative_result/`
