# Governor head — documented negative result

## What we built

End-to-end wiring of the biaffine governor head + attachment-contrastive
triplet loss on top of `validated_nextgen_recovery`:

- Trainer flags: `--enable_governor_head`, `--attachment_contrastive_lambda 0.1`
- Collator emits `dep_head_labels` with self-loop filter (caught a real
  data bug: distill_v2 had spurious self-loops on tokens 0/1/8 of some
  sentences)
- Model: biaffine `query·W·key` over candidate head positions, masked
  diagonal + pad with `-1e9`, no label_smoothing on this head (caused
  +inf with masked logits)
- Loss: governor CE @ weight 0.5 + attachment contrastive @ weight 0.1
- Strict no-leakage assertions intact

## What worked

- Governor CE trained without instability (finite gradients throughout)
- Attachment contrastive spiked properly on nested-syntax data
  (1.0–3.0 in stage 4–5, 0.0 when negatives were already separated)
- Stage-7 checkpoint saved cleanly; full eval ran without error

## What did not work

On the full uncapped held-out sets vs `validated_recovery`:

| Dataset | metric | recovery | governor | Δ |
|---|---|---|---|---|
| Gazelle | fully | 0.459 | 0.459 | +0.000 |
| Gazelle | case  | 0.646 | 0.661 | +0.015 |
| Gazelle | role  | 0.613 | 0.600 | −0.013 |
| MASAQ   | fully | 0.711 | 0.714 | +0.003 |
| MASAQ   | role  | 0.807 | 0.805 | −0.002 |
| MASAQ   | case  | 0.848 | 0.844 | −0.004 |

The dominant idafa-attachment confusions are unchanged:

| Confusion | recovery | governor |
|---|---|---|
| mudaaf_ilayh → mafoul_bih | 32 | 32 |
| mudaaf_ilayh → mubtada | 29 | 29 |
| mudaaf_ilayh → ism_majrur | 13 | 13 |

## Interpretation

The auxiliary governor task does not transfer to the role-prediction
head at this data scale. The model learns to predict heads (attachment
loss approaches 0 on training-time batches) but the role head's
mudaaf_ilayh confusion is a **semantic** ambiguity, not a
**structural-attachment** ambiguity. Two adjacent nouns are nearly
always grammatically attached (the head sets case); deciding whether
the relation is *idafa* vs *mafoul_bih* requires semantic / lexical
knowledge of the verb's argument structure that the structural head
cannot supply.

Combined with the prior graph-integration negative result, the picture
is now consistent: at our data scale (~20k sentences), more
structural supervision does not reduce the idafa confusion. The
remaining bottleneck is **lexical-semantic supervision** (verb→argument
structure annotations) and **alternative-analysis annotations**
(genuinely ambiguous tokens marked as such, scored permissively).

## Why this is documented

The implementation is correct, the training is stable, the negative
finding constrains the search space. Future work proposing yet
another structural head should reproduce this result before claiming
gains.

The validated production checkpoint remains `runs/validated_nextgen_recovery/`.
