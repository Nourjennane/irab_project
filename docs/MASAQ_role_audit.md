# MASAQ role-F1 disagreement audit

**Question:** Is the apparent role-F1 collapse for AraT5v2-base on MASAQ (54.2 on Gazelle → 10.2 on MASAQ) a real cross-register finding, or a measurement artifact?

**Method:** Sampled 20 random word-level role disagreements between AraT5v2-base prediction and MASAQ gold (out of 302 cases where both gold and pred had a role extracted). Read each disagreement by hand and classified.

**Result: 14 of 20 (70%) are vocabulary artifacts. The role-F1 collapse is a measurement artifact, NOT a cross-register finding.**

## Disagreement type counts (across the entire MASAQ AraT5v2 prediction set)

- 1,815 — gold has a role extracted, pred extractor returns no role (pred uses paraphrases the extractor regex doesn't match)
- 302 — both have a role extracted, but they differ
- 38 — gold extractor returns no role, pred does

The first bucket (1,815, the largest) is mostly vocabulary mismatch too: predictions phrase the same role with words the extractor's ROLES list doesn't enumerate.

## Audit of the 302 "both extracted but differ" bucket — 20-row sample

The dominant pattern: MASAQ gold's templated form is `"اسم مجرور بحرف الجر وعلامة جره الكسرة الظاهرة"` (literally "noun in genitive by preposition"). Our trained model produces `"اسم مجرور بمن وعلامة جره الكسرة الظاهرة على آخره"` ("noun in genitive by-min …"), or splits the analysis into multiple morpheme-by-morpheme lines (e.g. `"الباء حرف جر مبني على الكسر، آيات اسم مجرور بالباء..."`).

Both descriptions encode the **same grammatical analysis**: the noun is genitive because it's the object of a preposition. But our `structural.extract` ROLES list contains both `"اسم مجرور بحرف الجر"` (long, specific) and `"اسم مجرور"` (shorter). The extractor returns the FIRST match in priority order:

- On the gold templated text: `"اسم مجرور بحرف الجر"` is present → extracts that
- On the pred text: `"اسم مجرور بحرف الجر"` is NOT present (model uses different wording) → falls back to `"اسم مجرور"`

Same grammatical content, two different role-string outputs from the same extractor. Counted as disagreement.

| # | Disagreement type | Vocabulary artifact? |
|---:|---|---|
| 1 | اسم مجرور بحرف الجر / اسم مجرور | YES (same case+marker, same analysis) |
| 2 | اسم مجرور / نعت — different case (majrur/marfu) | NO — real disagreement |
| 3 | اسم مجرور بحرف الجر / اسم مجرور | YES |
| 4 | اسم مجرور بحرف الجر / اسم مجرور | YES |
| 5 | مفعول به / خبر — both mansub, classical scholarly variation | BORDERLINE (treat as real) |
| 6 | اسم مجرور بحرف الجر / اسم مجرور | YES |
| 7 | اسم مجرور بحرف الجر / اسم مجرور | YES |
| 8 | اسم مجرور بحرف الجر / اسم مجرور | YES |
| 9 | مفعول به (mansub) / اسم مجرور (majrur) — model wrong | NO — real model error |
| 10 | اسم مجرور بحرف الجر / اسم مجرور | YES |
| 11 | مفعول به / فاعل — model wrong on Quranic | NO — real model error |
| 12 | اسم مجرور بحرف الجر / اسم مجرور | YES |
| 13 | اسم مجرور بحرف الجر / اسم مجرور | YES |
| 14 | اسم مجرور بحرف الجر / اسم مجرور | YES |
| 15 | اسم مجرور بحرف الجر / اسم مجرور | YES |
| 16 | اسم مجرور بحرف الجر / اسم مجرور | YES |
| 17 | اسم مجرور / مضاف إليه — extractor matched a suffix-pronoun's role | YES (extractor alignment bug) |
| 18 | اسم مجرور / بدل — different case, possibly real | BORDERLINE (treat as real) |
| 19 | مفعول به / بدل — model wrong on Quranic | NO — real model error |
| 20 | اسم مجرور بحرف الجر / اسم مجرور | YES |

**Classification:** 14/20 (70%) vocabulary artifact, 6/20 (30%) real (4 model errors + 2 borderline scholarly variation).

## Implication for what we report

- **DO NOT report** "AraT5v2-base role-F1 collapses on MASAQ" as a cross-register **finding**. It is dominated by a mismatch between the MASAQ templater's role vocabulary and what the model produces, both pipelined through our structural extractor.
- **DO report** as a transparent measurement limitation: "MASAQ role-F1 numbers should not be cross-register-compared to Gazelle role-F1 because the extractor's role term priority differs systematically between the two evaluation surfaces."
- Case-acc and marker-EM are not subject to this artifact (their atomic vocabulary is closed and small: rafʿ/naṣb/jarr/jazm/mabni for case; ~15 markers). Cross-register comparison on those two metrics is sound.
- The aggregate `fully_correct_word` metric inherits the role-F1 artifact, so cross-register fully comparison is also weakened. Report fully numbers, but caveat that role's artifact propagates.

## What this means for the larger story

- The genuine cross-register finding (~30% of disagreements) is small but real: the trained model occasionally mis-analyses Quranic constructions (mistaking object for prep-object, choosing fā'il vs مفعول به wrong on classical syntax). But this is a 6-out-of-20 effect, not a 44 pp role-F1 collapse.
- The original status-doc "role-F1 collapse 54.2 → 10.2" claim is **withdrawn**. Status doc has been updated; RESULTS.md when written should not feature this as a finding.
- The MASAQ eval surface is still useful for case-acc and marker-EM cross-register comparison and for tightening confidence intervals on the open-weight ranking.
