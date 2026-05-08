# Qualitative trace: structured-v1-rebuild + 4 symbolic constraints

Sample sentences from Gazelle (n=134, MSA news). For each word the model emits four canonical labels (case / role / marker / POS) plus the constraint-fired log; the rendered Arabic prose is the deterministic template renderer's output.

### Kāna sister (ليس) + iḍāfa

**Sentence:** `ليس العلمُ بضار`

| # | word | case | role | marker | POS | constraints | rendered i'rāb | gold i'rāb |
|---|---|---|---|---|---|---|---|---|
| 1 | ليس | mabni | fil | sukun | verb |  | فعل مبني على السكون لا محل له من الإعراب | فعل ماض ناقص ( من أخوات كان ) مبني على الفتحة الظاهرة في آخره. |
| 2 | العلمُ | raf | fail | damma_visible | noun | kana ism→raf | فاعل مرفوع وعلامة رفعه الضمة الظاهرة على آخره | — |
| 3 | بضار | jarr | ism_majrur | kasra_visible | noun | kana khabar→nasb; iḍāfa | اسم مجرور وعلامة جره الكسرة الظاهرة على آخره | الباء حرف جر زائد ، ضار : اسم مجرور لفظا ، منصوب محلا على أنه خبر ( ليس ). |

### Preposition → jarr

**Sentence:** `أنت نشيط في المدرسة`

| # | word | case | role | marker | POS | constraints | rendered i'rāb | gold i'rāb |
|---|---|---|---|---|---|---|---|---|
| 1 | أنت | mabni | mubtada | sukun | pronoun |  | مبتدأ مبني على السكون لا محل له من الإعراب | ضمير منفصل مبني على الفتح في محل رفع مبتدأ. |
| 2 | نشيط | raf | khabar | damma_visible | noun |  | خبر مرفوع وعلامة رفعه الضمة الظاهرة على آخره | خبر مرفوع وعلامة رفعه الضمة الظاهرة على آخره. |
| 3 | في | mabni | harf_jarr | sukun | particle |  | حرف جر مبني على السكون لا محل له من الإعراب | حرف جر. |
| 4 | المدرسة | jarr | ism_majrur | kasra_visible | noun | prep→jarr | اسم مجرور وعلامة جره الكسرة الظاهرة على آخره | اسم مجرور بـ ( في ) وعلامة جره الكسرة الظاهرة على آخره والجار والمجرور متعلقان بالخبر ( نشيط ). |

### Iḍāfa stub

**Sentence:** `مررتُ بأخيكَ زيدٍ`

| # | word | case | role | marker | POS | constraints | rendered i'rāb | gold i'rāb |
|---|---|---|---|---|---|---|---|---|
| 1 | مررتُ | mabni | fil | sukun | verb |  | فعل مبني على السكون لا محل له من الإعراب | — |
| 2 | بأخيكَ | jarr | mudaaf_ilayh | kasra_visible | noun |  | مضاف إليه مجرور وعلامة جره الكسرة الظاهرة على آخره | — |
| 3 | زيدٍ | jarr | badal | damma_visible | noun | iḍāfa | بدل مجرور وعلامة جره الضمة الظاهرة على آخره | — |
