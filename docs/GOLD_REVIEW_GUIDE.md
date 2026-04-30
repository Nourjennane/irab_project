# Gold Benchmark Review Guide

You have **82 sentences / 571 word judgments** in `data/gold_seed.jsonl`, each seeded by Claude Sonnet 4.5. Your job: spot-check, fix where wrong, set `"verified": true`.

This is the single highest-leverage piece of human work for the project's grade. With this benchmark in the writeup, the smallest detectable difference drops from ±7 pp (Gazelle, n=134) to ~±3 pp (here, n=571). That converts the current "we cannot detect" findings into hard claims.

---

## File format

Each line is one sentence. Inside `items`, each entry is a per-word annotation:

```json
{
  "sentence": "ذهب الطالب إلى المدرسة",
  "items": [
    {"word": "ذهب",    "irab": "فعل ماضٍ مبني على الفتح",                    "verified": false},
    {"word": "الطالب", "irab": "فاعل مرفوع وعلامة رفعه الضمة الظاهرة",       "verified": false},
    {"word": "إلى",    "irab": "حرف جر مبني لا محل له من الإعراب",            "verified": false},
    {"word": "المدرسة","irab": "اسم مجرور بحرف الجر وعلامة جره الكسرة الظاهرة","verified": false}
  ],
  "padt_sent_id": "ar_padt-ud-train-AFP-20000715-XXX",
  "seed_model": "claude-sonnet-4-5"
}
```

---

## What to check, per word

For each `item`, decide if Sonnet got these three things right:

| Field | Look for |
|---|---|
| **case** (rafʿ/naṣb/jarr/jazm/mabni) | Reflected in the case word: مرفوع / منصوب / مجرور / مجزوم / مبني |
| **role** (فاعل / مفعول به / مضاف إليه / etc.) | The first noun phrase in the prose |
| **marker** (الضمة / الفتحة / الكسرة / السكون / الواو / etc.) | The phrase after وعلامة رفعه/نصبه/جره |

**Common Sonnet failure modes to watch for:**

1. **Wrong case after preposition.** If a noun follows إلى / في / من / على / ب / ل / عن — it must be **مجرور**. Sonnet sometimes outputs مرفوع.
2. **Iḍāfa case.** A noun whose head is another noun in possession (e.g., كتابُ الطالبِ — طالب is مجرور as مضاف إليه). Sonnet sometimes mislabels as مفعول به.
3. **Sisters of إنّ** (إن، أنّ، لكنّ، كأنّ، ليت، لعلّ) flip noun-marfu' to mansub for the *ism* role. Sonnet sometimes misses this.
4. **Sisters of كان** (كان، صار، أصبح، ليس، …) flip the *khabar* to mansub. Watch for these in nominal sentences.
5. **Mood-shifting particles:** لن makes verb mansub; لم makes it majzum; أن makes it mansub. Verify the verb's mood matches.
6. **Exception (istithnāʾ).** If you see سوى، إلا، عدا، خلا — the post-marker noun's case is *not* default. Verify carefully.

---

## How to edit

Open `data/gold_seed.jsonl` in any text editor that handles UTF-8 (VS Code, Sublime, vim, Streamlit — your call).

**For each row:**

1. Read the sentence.
2. For each `item`:
   - If `irab` is correct: change `"verified": false` → `"verified": true`. Done.
   - If wrong: edit the `irab` string to the correct prose, then set `"verified": true`.
3. Save the file.

**Format conventions:**
- Keep prose in the same canonical pattern: `<role> <case_word> وعلامة <case_genitive_form> <marker>`. Example: `فاعل مرفوع وعلامة رفعه الضمة الظاهرة`.
- For mabni: `<role> مبني على <vowel>` or `<pos> مبني لا محل له من الإعراب`.
- Don't change `word` or `sentence` fields.
- Keep the JSON valid (commas, quotes, no trailing commas).

---

## Pacing

- 82 sentences × ~2 min each = **~3 hours of focused work**.
- Most items will just need a glance + flip-the-flag. Sonnet at temperature=0 with RAG examples is reasonably accurate on case (~67% on Gazelle, similar expected here).
- The mistakes you'll find cluster around the failure modes above.

---

## Progress check

Run anytime:

```bash
.venv/bin/python -m irab_tashkeel.evaluation.gold_progress
```

It reports verified-row count, sentence count, and flags any items where the structured fields can't be cleanly extracted (likely formatting drift you'd want to fix).

---

## When you're done

Tell me. I'll re-run the three systems (per-word decoder, Claude RAG, Hybrid) against this benchmark, compute paired stats vs Gazelle, and populate §6.4 of the report with the comparison.

**This benchmark + the writeup polish is the path to 30.**
