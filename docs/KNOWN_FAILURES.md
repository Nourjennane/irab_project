# Known Failures

Cases the validated nextgen recovery model
(`runs/validated_nextgen_recovery/`) is known to handle poorly.
Updated as new failures surface.

## 1. nasb / jarr ambiguity in idafa edge cases

When a noun is the *mudaaf ilayh* (second member of an idafa) but
appears at the start of a prepositional phrase, the model occasionally
predicts `nasb` instead of `jarr`. The marker head usually corrects to
kasra, leaving the case label inconsistent with the marker — a flag for
a downstream consistency check.

## 2. Construction nesting beyond depth 2

`kana` over `inna` over an idafa works. Add a fourth nested
construction (e.g. *masdar* of *inna* of *kana* of an idafa) and the
construction tagger drops the outermost frame. Symptom: the outermost
`ism_kana` token gets relabelled as `mubtada`.

## 3. Munada with implicit *yaa* particle

In *yaa rabbi* the *yaa* particle is optional in surface form. When
absent, the model under-applies the `munada` role. Workaround: include
the *yaa* explicitly in the input.

## 4. Quranic prose with archaic morphology

Stage 7 was trained on MASAQ Quranic, but archaic forms outside MASAQ's
coverage (e.g. some pre-Islamic poetic structures) collapse to the most
similar MSA pattern. Out-of-distribution by design.

## 5. Calibration on Quranic full predictions

quranic_fully=1.0 on a 100-sample eval slice during training does not
generalize: on the full MASAQ test, the number is lower (see
[`docs/final_eval/final_eval_report.md`](final_eval/final_eval_report.md)).
This is **not** a regression — it is the honest measurement at full
sample size.

## 6. Long sentences (≥ 40 tokens)

Performance degrades on sentences past ~40 tokens. The DepAware model's
word-pooling mask is fixed at training time and beyond ~40 tokens the
positional information from the encoder begins to wash out. For
production, segment longer inputs.

## 7. Free-form generation requested by user

The model has no generative head. If asked to "explain in Arabic prose
what each token does," the system uses the deterministic template
renderer in [`src/irab_tashkeel/reasoning/`](../src/irab_tashkeel/reasoning/),
which produces correct but stylistically rigid output. This is by
design — see [`docs/LIMITATIONS.md`](LIMITATIONS.md).

## 8. Construction detection for novel families

The construction tagger is trained on the seven canonical families in
[`src/irab_tashkeel/data_v2/constructions/`](../src/irab_tashkeel/data_v2/constructions/).
Novel construction families are silently mislabelled as the closest
known family. To extend, add a new family schema and retrain the
construction head only (cheap fine-tune, no full retrain).
