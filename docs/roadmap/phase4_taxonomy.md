# Phase 4 — Expanded Role Taxonomy (Design Doc)

> The roadmap is now: rev 2 → Phase 1 morphology → **Phase 4 syntax granularity** → Phase 3 dependency → Phase 2 soft conditioning. Phase 4 isolates the question *"does finer syntactic granularity alone improve role reasoning?"* without adding any new architectural mechanism. **No code is written before this design doc is approved.**

---

## 1. Why a taxonomy expansion now

**Empirical evidence that the schema is the ceiling.** Phase 1 closed
≈ 28% of the gap to seq2seq AraT5v2 FT on Gazelle role-F1
(rev 2 36.9 → Phase 1 42.3 → seq2seq 54.8). The remaining ~12 pp gap on
role-F1 happens with a model that already reads strong morphological
features and operates with the same encoder. The next-most-likely
bottleneck is **schema collapse**: 590 distinct gold role strings →
25 canonical labels.

**Three concrete patterns the data shows:**

1. The 25-label scheme over-collapses **adverbial + verbal** structure. Three labels (`mafoul_other`, `dharf`, `fil`, `harf_other`) each cover linguistically distinct sub-categories that the symbolic-constraint layer, the template renderer, and the eval extractor all treat differently downstream.
2. Model confusion analysis on Gazelle (table §3.4) shows `fail → fil` is the dominant single mismatch — disambiguating verb sub-types could give the role decoder more signal.
3. Schema clusters with one or two dominant raw strings (e.g. `mudaaf_ilayh` 99.4% one form) need NO splitting — finer granularity there would just add sparsity. **Splits should target only the heterogeneous canonical labels.**

**Hard constraints (re-stated from user, frozen):**

- 25 → ~35–40 first; **not** straight to 60+.
- Reversible — old↔new label conversion tables, grouped evaluation.
- Avoid ultra-rare labels (< 100–150 support).
- Symbolic-rule + renderer + interpretability compatible.
- Phase 4 isolates "finer granularity" — **no other architectural changes** simultaneously.
- Phase 1 + rev 2 stay frozen as comparison baselines.
- Success: improve role-F1 and/or fully without destabilising cross-register.

---

## 2. Method

### 2.1 Analysis sources

- **distill_v2/distilled.jsonl**, 5,000 sentences / 77,534 words / 590 distinct raw role strings — the training corpus for both rev 2 and Phase 1.
- **Gazelle predictions JSONL** for rev 2 (job 490894) and Phase 1 (job 490987) — for confusion analysis.
- **MASAQ predictions** for both — for cross-register sanity (we don't propose splits driven by MASAQ alone, but we sanity-check that proposed splits exist there too).

### 2.2 Pipeline (already executed for this doc)

1. Histogram raw role strings (frequencies).
2. For every current canonical label, list which raw strings get folded into it + their relative weight.
3. From Gazelle predictions, list (gold_canon → pred_canon) confusion pairs.
4. Score candidate splits: linguistic meaning × support count × extractor mismatch potential × symbolic-rule alignment.
5. Apply the support floor (≥ 100 raw examples for a new label, with one borderline exception flagged).

---

## 3. Findings (raw analysis)

### 3.1 Raw role distribution (top 25 of 590)

| Rank | Count | Share | Raw role string |
|---:|---:|---:|---|
| 1 | 14,102 | 18.19% | مضاف إليه |
| 2 |  7,907 | 10.20% | نعت |
| 3 |  7,385 |  9.52% | حرف جر |
| 4 |  7,372 |  9.51% | اسم مجرور |
| 5 |  6,914 |  8.92% | بدل |
| 6 |  5,055 |  6.52% | حرف عطف |
| 7 |  3,296 |  4.25% | مبتدأ |
| 8 |  2,731 |  3.52% | فعل (generic) |
| 9 |  2,471 |  3.19% | مفعول به |
| 10 |  2,297 |  2.96% | خبر |
| 11 |  1,416 |  1.83% | فاعل |
| 12 |  1,333 |  1.72% | ظرف زمان |
| 13 |  1,224 |  1.58% | علامة ترقيم |
| 14 |  1,191 |  1.54% | فعل مضارع |
| 15 |    838 |  1.08% | حال |
| 16 |    768 |  0.99% | ظرف مكان |
| 17 |    664 |  0.86% | معطوف |
| 18 |    655 |  0.84% | فعل ناقص |
| 19 |    528 |  0.68% | فعل ماضٍ |
| 20 |    444 |  0.57% | تمييز |
| 21 |    442 |  0.57% | مجرور بحرف الجر |
| 22 |    333 |  0.43% | ظرف (generic) |
| 23 |    326 |  0.42% | جار ومجرور |
| 24 |    322 |  0.42% | متعلق بالفعل |
| 25 |    303 |  0.39% | اسم أن |

**The top 25 raw strings cover ~91% of all word judgments**, but the 25-label canonical scheme groups them into 14 buckets. So the granularity loss happens at the most-frequent raw labels, not the long tail.

### 3.2 Heterogeneous canonical buckets (split candidates)

These four canonical labels each absorb sub-clusters with linguistically distinct functions and ≥ 100 support per sub-cluster:

| Current canonical | Total | # distinct raw | Sub-cluster (count, %) | Linguistic role |
|---|---:|---:|---|---|
| **dharf** | 2,444 | 9 | ظرف زمان (1,333, 54.5%) | adverbial of time |
|   | | | ظرف مكان (768, 31.4%) | adverbial of place |
|   | | | ظرف generic (333, 13.6%) | other adverbial |
| **fil** | 5,977 | 66 | فعل (2,731, 45.7%) | unspecified verb |
|   | | | فعل مضارع (1,191, 19.9%) | imperfect (present) verb |
|   | | | فعل ناقص (655, 11.0%) | defective verb (kāna family) |
|   | | | فعل ماضٍ (528 + 69 = ~597) | perfect (past) verb |
|   | | | متعلق بالفعل (322, 5.4%) | verb-attached prepositional phrase |
| **harf_other** | 1,197 | 50 | حرف توكيد ونصب (173, 14.5%) | emphasis-and-accusative (inna sister marker) |
|   | | | حرف تحقيق (156, 13.0%) | verification (e.g. قد) |
|   | | | حرف نفي (153, 12.8%) | negation (e.g. لا، لم) |
|   | | | حرف ناصب (105, 8.8%) | accusative-marker (e.g. أن، لن) |
|   | | | أداة تعريف (88, 7.4%) | definite article ال |
|   | | | حرف نصب (64, 5.3%) | accusative-marker (alt) |
|   | | | حرف نفي وجزم (60, 5.0%) | negation-and-jussive (لم) |
| **mafoul_other** | 334 | 6 | مفعول مطلق (278, 83.2%) | cognate accusative (verb-emphasizing) |
|   | | | مفعول فيه (26+8=34, 10%) | adverbial accusative — overlaps `dharf` |
|   | | | مفعول لأجله (21, 6.3%) | causative accusative |

The "other" canonical bucket also contains:

| | Total | # distinct | Sub-cluster | |
|---|---:|---:|---|---|
| **other** | 921 | 192 | اسم موصول (161, 17.5%) | relative pronoun (الذي، التي، …) |

### 3.3 Homogeneous canonical buckets (NO split — keep as-is)

These canonical labels are dominated (> 95%) by a single raw form. Splitting would add label-space without adding signal:

- `mubtada` (95.7% one form), `khabar` (93.3%), `fail` (98.0%), `mafoul_bih` (98.3%), `naat` (98.2%), `badal` (96.2%), `mudaaf_ilayh` (99.4%), `tamyeez` (100% one form), `harf_jarr` (96.7%), `harf_atf` (88.6%, runner-up "عاطف" is a paraphrase not a sub-class), `naib_fail` (100%), `punctuation` (100%).

### 3.4 Confusion pairs (Gazelle, n=58 mismatches across rev 2 + Phase 1)

| Frequency | Confusion (gold → pred) | Read |
|---:|---|---|
| 8 | `fail → fil` | Subject ↔ verb confusion. **Split candidate**: distinguishing verb sub-types (past/present/defective) gives the model more signal to push role away from "fil" when the word is a noun. |
| 4 | `ism_majrur ↔ mudaaf_ilayh` (symmetric) | Both genitive, both post-particle. Hard distinction. **Phase 3 (dependency-aware)**, not Phase 4. |
| 3 | `mafoul_bih → naat / fil / khabar` | Object ↔ adjective / verb / predicate confusion. Splits don't directly help. |
| 2 | `mudaaf_ilayh → khabar_kana / khabar_inna` | Mostly bias from upweighted *kāna*/*inna* classes. |
| 2 | `mubtada → ism_inna / harf_jarr` | Surface-position confusion. |
| Rest | 1-occurrence pairs | Spread out; no single dominant pattern. |

Phase 4 splits **directly target** the `fail → fil` confusion via `fil_madi / fil_mudari / fil_naqis`. The other top confusions are not addressed by granularity changes.

---

## 4. Proposed taxonomy: 25 → 35 labels

10 new labels (8 splits + 2 carve-outs from `other`). All ≥ 150 support except one borderline (flagged). All linguistically meaningful and rule-compatible.

### 4.1 Expansion table

| New label | Source canonical | Source raw strings | Support | Linguistic justification |
|---|---|---|---:|---|
| `dharf_zaman` | `dharf` (split) | ظرف زمان | 1,333 | Adverbial of time (e.g. اليوم، أمس). Symbolic-rule relevant: governs case differently from non-temporal nouns. |
| `dharf_makan` | `dharf` (split) | ظرف مكان | 768 | Adverbial of place (e.g. أمام، خلف، فوق). Governs an `i\d{d}\=afa` chain typically. |
| `dharf` | `dharf` (kept; generic + minor variants) | ظرف, ظرف شرط, … | 343 | Catch-all generic adverbial. |
| `fil_madi` | `fil` (split) | فعل ماضٍ + فعل ماض variants | 597 | Past verb. Carries Aspect=Perf morphology — Phase 1's morph head gives orthogonal signal. |
| `fil_mudari` | `fil` (split) | فعل مضارع | 1,191 | Present verb. Aspect=Imp morphology. |
| `fil_naqis` | `fil` (split) | فعل ناقص | 655 | Defective verb (the kāna family). Triggers the existing `kana_sisters` symbolic rule directly. |
| `fil` | `fil` (kept; generic + the rest) | فعل, متعلق بالفعل, فعل القول, … | ≈ 3,534 | Generic verb (no aspect/family info in source). |
| `harf_nafy` | `harf_other` (split) | حرف نفي, حرف نفي وجزم | 213 | Negation particle (لا، لم، ما). Affects mood (jussive) — already in Phase 1 morph head. |
| `harf_nasb` | `harf_other` (split) | حرف توكيد ونصب, حرف ناصب, حرف نصب | 342 | Accusative-marker. Triggers existing `inna_sisters` constraint when surface is إن/أن/لكن/ليت/لعل. |
| `harf_tahqiq` | `harf_other` (split) | حرف تحقيق | 156 | Verification particle (قد). Distinct because it doesn't change case but changes aspect/emphasis. |
| `harf_other` | `harf_other` (kept; the rest) | أداة تعريف, حرف ناسخ, حرف مصدري, … | ≈ 486 | Catch-all for the long tail. |
| `mafoul_mutlaq` | `mafoul_other` (split) | مفعول مطلق | 278 | Cognate accusative. Emphasizes a verb's action; case=nasb invariably. |
| `mafoul_other` | `mafoul_other` (kept; the rest) | مفعول فيه, مفعول لأجله, مفعول معه | ≈ 56 | Catch-all (mafoul fih overlaps dharf; mafoul lah is rare). |
| `mawsool` | `other` (carve-out) | اسم موصول | 161 | Relative pronoun (الذي، التي، الذين، …). Currently catch-all "other" — bad for interpretability. **Borderline** support; ship with monitoring. |

**Total new labels: 10.** Net taxonomy: 25 + 10 = **35 labels** + the catch-all `other` (now ≈ 760 since `mawsool` and `mafoul_mutlaq` carved out).

### 4.2 Support distribution after the expansion

```
Support range    # of canonical labels
≥ 5,000              5
1,000 – 4,999        7
  500 – 999          5
  200 – 499          5
  150 – 199          1   (mawsool — borderline)
  < 150              0   ← all expansion labels clear the floor
```

**No ultra-rare labels introduced.** The smallest new label (`mawsool`) has 161 raw examples — comfortably above the 100-floor stated as the user's threshold.

### 4.3 Splits explicitly *NOT* taken in Phase 4

| Considered | Reason rejected |
|---|---|
| `mafoul_lah` (cause) | only 21 raw examples — ultra-rare |
| `mafoul_fih` (place/time) | overlaps `dharf_zaman/makan`; 32 raw examples |
| `mubtada_thani` (second mubtada) | 36 + 16 + 10 = 62 raw — borderline rare |
| `khabar_jumla` (verbal khabar) | not extractable from raw cluster |
| `harf_tarif` (definite article ال) | 88 raw — below floor |
| `harf_masdari` (مصدري particle) | 39 raw — too rare |
| Splits inside `naat`, `badal`, `mudaaf_ilayh` | each is > 95% dominated by one form |
| `matuf_noun` vs `matuf_verb` | Phase 1's POS head already encodes this |
| `fil_amr` (imperative) | not present in distill_v2 (news register) |

These are **explicitly deferred** to a later (e.g. Phase 4b) expansion once we see how the 35-label scheme behaves.

---

## 5. Backward compatibility & reversibility

### 5.1 Old → new label mapping

For every Phase 4 retrain we ship a frozen `OLD_TO_NEW` table that lets us:

- Re-evaluate Phase 4 predictions under the rev 2 / Phase 1 25-label scheme (collapse-down).
- Compare apples-to-apples on the existing rev 2 + Phase 1 numbers.
- Swap back to the old taxonomy for any external consumer that depends on 25 labels.

```python
# Phase 4 expansion mapping (forward = split; reverse = collapse-back-to-25)
NEW_TO_OLD = {
    "dharf_zaman":    "dharf",
    "dharf_makan":    "dharf",
    "dharf":          "dharf",         # generic stays itself
    "fil_madi":       "fil",
    "fil_mudari":     "fil",
    "fil_naqis":      "fil",
    "fil":            "fil",
    "harf_nafy":      "harf_other",
    "harf_nasb":      "harf_other",
    "harf_tahqiq":    "harf_other",
    "harf_other":     "harf_other",
    "mafoul_mutlaq":  "mafoul_other",
    "mafoul_other":   "mafoul_other",
    "mawsool":        "other",
    # ...all other 24 labels unchanged...
}
```

Implementation: stored once in `src/irab_tashkeel/structured/taxonomy_v4.py` as a frozen dict; tests verify `OLD_TO_NEW(NEW_TO_OLD(x)) == x` for every old label.

### 5.2 Grouped evaluation

Two evaluation surfaces are produced for every Phase 4 retrain:

- **Native (35-label)**: the full new taxonomy. Reports per-class accuracy + macro F1 on the new 35 classes.
- **Grouped (25-label)**: predictions are collapsed via `NEW_TO_OLD` and scored against the rev 2 / Phase 1 gold canonicalisation. Apples-to-apples vs rev 2 / Phase 1 numbers.

Both are emitted by the eval pipeline; the paper writeup reports both.

---

## 6. Architecture impact

**Single change:** the role head's output dimension goes from 25 to 35.

```python
# Before (rev 2 / Phase 1):
self.role_head = nn.Linear(self.hidden_size, 25)

# After (Phase 4):
self.role_head = nn.Linear(self.hidden_size, 35)
```

That's it. ~7,680 extra parameters. Nothing else changes.

**Per design rule #6, the following stay compatible:**
- **Symbolic constraints**: existing 9 constraint families reference role names (e.g. `ROLE_TO_ID["ism_inna"]`). New labels don't break any existing rule. Two new rules become possible (e.g. `dharf_zaman → nasb` strong bias) but those are Phase 4b — for the first iteration, leave the constraint engine untouched to isolate the granularity effect.
- **Template renderer**: each new label gets one new entry in `ARABIC_ROLE_FORMS` and `template_renderer.render_word()`. Trivial 14-line patch.
- **Interpretability**: per-head confidence still reported. Per-class confusion now over 35 classes (richer interpretability surface, not poorer).
- **Modularity**: Phase 4 lives alongside (not on top of) Phase 1. Two configs:
  - `configs/structured_v1_rebuild.yaml` (rev 2, frozen, 25 labels)
  - `configs/phase1_morphology.yaml` (Phase 1, 25 labels + morph)
  - `configs/phase4_taxonomy.yaml` (Phase 4, 35 labels + morph) — NEW
  - `configs/phase4_taxonomy_no_morph.yaml` (Phase 4, 35 labels, no morph) — NEW, for ablation

---

## 7. Implementation plan (after this doc is approved)

### 7.1 File-level targets

```
src/irab_tashkeel/structured/
   taxonomy_v4.py                # NEW — frozen 35-label set + NEW_TO_OLD/OLD_TO_NEW + canonicalize_role_v4
   schema.py                     # MODIFY — re-export the v4 sets behind a flag; rev 2 path unchanged
   model.py                      # No code changes; head sizes parameterized via schema.N_ROLE
   word_irab.py                  # No code changes; role field stays free-string

src/irab_tashkeel/morphology/
   schema.py / dataset.py        # No changes (morph heads are independent)

src/irab_tashkeel/inference/
   template_renderer.py          # MODIFY — add 10 new entries to ARABIC_ROLE_FORMS + per-label rendering
   symbolic_constraints.py       # MODIFY — fall-through: any new label maps to closest old label for existing rules; no new rules in Phase 4
   structured_predictor.py       # No changes

src/irab_tashkeel/evaluation/
   structural.py                 # MODIFY — add new role surfaces to ROLES list (longest-first ordering preserved)
   v4_eval.py                    # NEW — produces both 35-label native + 25-label grouped metrics

scripts/structured/
   build_structured_corpus_v4.py # NEW — re-canonicalise distill_v2 against the v4 taxonomy
   eval_phase4.py                # NEW — runs both surfaces, emits per-class metrics + confusion + calibration

scripts/slurm/
   62_train_phase4_taxonomy.sbatch     # NEW
   63_eval_phase4_taxonomy.sbatch      # NEW

configs/
   phase4_taxonomy.yaml          # NEW — Phase 1 morph + Phase 4 35-label role
   phase4_taxonomy_no_morph.yaml # NEW — pure 35-label (no morph) for the granularity-only ablation

tests/
   test_taxonomy_v4.py           # round-trip: old↔new mapping is bijective on the 25 old labels
   test_taxonomy_v4_canonicalize.py  # raw → v4 canonical preserves at least the v3 cluster membership
```

### 7.2 Step order (one phase, three retrains for a clean ablation)

1. Write `taxonomy_v4.py` (label set + mappings + canonicalize_role_v4).
2. Re-canonicalise distill_v2 against v4 → `data/structured_v1_v4/{train,val}.jsonl`.
3. Re-merge with UD-PADT → `data/morph_v1_v4/{train,val}.jsonl`.
4. Update template_renderer (10 new role surfaces).
5. Update `structural.py` extractor: add the new role surfaces to the `ROLES` list (longest-first ordering preserved so `حرف نفي` doesn't accidentally match before `حرف`).
6. Unit tests on schema + renderer + extractor.
7. **Smoke test on HPC** (50–200 sentences, 1 epoch) — verify pipeline + masking + role-head dimensionality + no NaN.
8. **Three full retrains for the 2×2 ablation** (each ~7 min on the MIG slice):
   - `phase4_taxonomy.yaml`           (35 roles + morph)  — main ship candidate
   - `phase4_taxonomy_no_morph.yaml`  (35 roles, no morph) — granularity-only effect
   - (rev 2 + Phase 1 are already frozen → no retrain needed)
9. Eval all four surfaces (rev 2 / Phase 1 / Phase 4-no-morph / Phase 4) on Gazelle + MASAQ + UD-PADT (morph is unchanged for non-morph models).

### 7.3 Total compute

3 retrains × ~7 min + 4 evals × ~5 min = ~50 minutes wall-clock on HPC.

---

## 8. Ablation matrix

The 2×2 ablation isolates the contribution of granularity vs morphology vs their combination:

| Run | Roles | Morph heads | Source of gain |
|---|---:|---:|---|
| rev 2 (frozen) | 25 | off | encoder + 4 heads + label smoothing + role weights |
| Phase 1 (frozen) | 25 | on | + multi-task morph supervision |
| Phase 4-no-morph | 35 | off | + finer role granularity |
| **Phase 4 (full)** | 35 | on | + granularity + morph (the candidate) |

**Apples-to-apples comparison** uses **grouped (25-label) evaluation** so the role-F1 numbers are directly comparable across all four runs. Native (35-label) numbers report on the new richer taxonomy and are reported alongside (descriptive only).

### 8.1 Reported metrics per run

For every run × eval-surface (Gazelle, MASAQ):

- `well`, `case`, `role-F1` (macro on 25 labels grouped + 35 labels native), `marker`, `fully`
- Per-class precision / recall / F1 (35 native + 25 grouped)
- Confusion matrix (35 native; produces `confusion_role_v4.csv`)
- Calibration: mean confidence on correct vs wrong predictions per class
- Paired bootstrap CI on the deltas:
  - Phase 4 vs Phase 1 (granularity gain — main test)
  - Phase 4-no-morph vs rev 2 (granularity-only)
  - Phase 4 vs Phase 4-no-morph (morph contribution at 35 labels)

---

## 9. Success criteria

A simple decision rule, mirroring Phase 1's:

| Criterion | Threshold |
|---|---|
| Phase 4 grouped role-F1 (25-label) on Gazelle | ≥ Phase 1 role-F1 (42.3) within paired-bootstrap CI |
| Phase 4 native role-F1 (35-label) on Gazelle | informational only — no pass/fail |
| Phase 4 fully on Gazelle | ≥ Phase 1 fully (19.4) within bootstrap CI |
| Phase 4 case + marker on Gazelle | ≥ Phase 1 numbers within CI |
| MASAQ role-F1 + fully | within ±1 pp of Phase 1 (cross-register stability test) |
| Per-class support on the new 35 labels | none < 100 in training corpus (verified pre-retrain) |

**Decision tree:**

- ✅ **Ship as Phase 4 default** if Phase 4 (full) beats Phase 1 grouped role-F1 + fully on Gazelle within CI, AND MASAQ doesn't regress > 1 pp.
- ⚠ **Ship as opt-in only** if granularity helps native role-F1 but grouped grouped role-F1 is unchanged (the model learned finer distinctions but the rev 2 / Phase 1 view doesn't see it).
- ❌ **Don't ship**, debug if grouped role-F1 regresses by > 2 pp.

In all cases, rev 2 + Phase 1 stay as frozen comparison baselines. Phase 4 lives alongside, not on top of.

---

## 10. Risks + mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Per-class sparsity for the borderline `mawsool` (161) | M | Class-weighted CE (sqrt-inv-freq, same as rev 2). Monitor per-class F1; if mawsool < 50% F1, fall back to keeping it merged in `other`. |
| Extractor mismatches because new role strings don't appear in `structural.py`'s ROLES list | H | Update structural.py FIRST (add the new role surfaces, longest-first ordering preserved). Unit test extractor on 100 distill_v2 prose strings before retraining. |
| Symbolic constraints reference old role IDs | M | New labels' fallback maps to the closest old label for existing rules — code path unchanged. New rules deferred to Phase 4b. |
| Template renderer doesn't render the 10 new labels | M | Add explicit entries to `ARABIC_ROLE_FORMS`. Round-trip-test 100 Phase 4 prose → extractor → match. |
| Grouped evaluation maps inconsistently | M | `taxonomy_v4.py::collapse_to_v3` is a pure function. Unit tests verify identity on every old label. |
| Cross-register MASAQ regression because new labels are MSA-news-frequent | L | Same UD-PADT-induced register skew that's already documented in Phase 1. We monitor + report; no architectural mitigation possible at this phase. |
| Encoder representations destabilise from increased role-head dim | L | Same encoder, same morph heads, same training recipe (label smoothing 0.1, sqrt-inv-freq weights). Only role head dim changes. |

---

## 11. What this phase explicitly does *not* do

Per user constraint #8, Phase 4 isolates "does finer syntactic granularity alone improve role reasoning?" These items are explicitly **deferred**:

- **Phase 2 — soft morphology conditioning into role/case/marker logits.** Phase 1 morph predictions stay observational + auxiliary.
- **Phase 3 — dependency-aware reasoning (graph attention / biaffine).** The `ism_majrur ↔ mudaaf_ilayh` confusion (#2 in Gazelle) is a dependency problem, not a granularity problem; Phase 4 doesn't try to address it.
- **New symbolic constraints exploiting the finer labels** (e.g. `dharf_zaman → nasb` bias, `harf_nasb → next-noun nasb`). These become possible with the 35-label scheme but stay out of scope until the granularity effect is measured in isolation.
- **Expansion to 60+ labels.** Per user constraint #1, that's a future Phase 4b once 35 is stable.

---

## 12. Required outputs (per user spec)

| # | Output | Path / format | Status |
|---|---|---|---|
| 1 | Phase 4 design doc | `docs/roadmap/phase4_taxonomy.md` | this file ✅ |
| 2 | Mapping tables (NEW_TO_OLD, OLD_TO_NEW) | `src/irab_tashkeel/structured/taxonomy_v4.py` | TBD |
| 3 | Per-class metrics | `runs/phase4_eval_<JOBID>/per_class_metrics.json` | TBD |
| 4 | Macro / micro support | embedded in per-class metrics | TBD |
| 5 | Confusion matrix | `runs/phase4_eval_<JOBID>/confusion_role_v4.csv` | TBD |
| 6 | Calibration stats | `runs/phase4_eval_<JOBID>/calibration_role_v4.json` | TBD |
| 7 | Gazelle + MASAQ comparison | `runs/phase4_eval_<JOBID>/{gazelle,masaq}/summary.json` | TBD |
| 8 | Old-vs-new ablation | `docs/roadmap/phase4_taxonomy.md` §13 (filled after retrains) | TBD |

---

## 13. Findings (filled after the three Phase 4 retrains)

(To be populated.)

---

## 14. Approval gate

Per user constraint #3 (*"Before coding: produce a taxonomy-analysis document"*), this design doc is the gate. Once approved, the implementation plan in §7 starts at step 1 and runs end-to-end with one ablation matrix and one ship decision.

The frozen rev 2 + frozen Phase 1 comparison baselines remain untouched throughout.

---

## 15. Phase 4a per-split detail tables (post-approval)

User-approved set: 9 splits, `mawsool` deferred to Phase 4b. Final taxonomy = 25 + 9 = **34 labels**.

Train/val tally below comes from re-running the seed=42 / val_frac=0.05 split that produces `data/structured_v1/{train,val}.jsonl` from `data/distill_v2/distilled.jsonl`. **All 9 splits clear both the 100-train support floor and the ≥1-val floor**; the smallest is `harf_tahqiq` at 157 train / 6 val.

### 15.1 Per-split detail (9 splits)

| Split | Raw support (train) | (val) | Parent | Raw strings absorbed | Linguistic rationale | Confusion reduction target | Expected gain hypothesis |
|---|---:|---:|---|---|---|---|---|
| **dharf_zaman** | 1,276 | 60 | dharf | ظرف زمان (+ rare variants) | Adverbial of time. Distinct from spatial adverbials in Arabic syntax: *aṭ-ẓarf az-zamānī* governs an iḍāfa with a temporal noun (e.g. *يوم العيد*). | Indirect — gives encoder finer signal between adverbial sub-types so the upstream `dharf` slot becomes less of a pile. | Macro role-F1 +0.5 to +1.0 pp on Gazelle (small absolute gain because Gazelle has few adverbials). |
| **dharf_makan** | 733 | 35 | dharf | ظرف مكان | Adverbial of place. Governs a place-noun iḍāfa (e.g. *أمام المسجد*); morphologically often nasb with implied accusative-of-locative. | Indirect — same as dharf_zaman. | Macro role-F1 +0.3 to +0.7 pp. |
| **fil_madi** | 566 | 31 | fil | فعل ماضٍ, فعل ماض | Past (perfect) verb. Aspect=Perf in UD-PADT. Always carries explicit subject (faʿīl) — this is the key signal the model needs to push role away from the dominant `fil` class when the word's POS+Aspect pair clearly indicates "main verb of the clause". | **Top: addresses 8 `fail→fil` Gazelle mismatches**: a noun shouldn't get fil_madi if its POS=noun. | Macro role-F1 +1.5 to +3 pp on Gazelle (largest single hypothesised gain). |
| **fil_mudari** | 1,109 | 82 | fil | فعل مضارع | Present (imperfect) verb. Aspect=Imp; mood-bearing (jussive/subjunctive). Distinct from past + defective in the constraint engine's perspective: present verbs trigger the `harf_nasb` and `harf_jazm` constraints, past verbs don't. | Same as fil_madi (verb sub-type disambiguation). | +0.5 to +1.5 pp on role-F1; cleaner than fil_madi because the present/jussive interaction is more deterministic. |
| **fil_naqis** | 625 | 30 | fil | فعل ناقص | Defective (kāna-family) verb. The trigger for the existing `kana_sisters` symbolic constraint. Currently: the constraint sees the surface word, not the model's prediction. With this label, the constraint can fire OFF the model's role prediction — closing the loop. | Indirect — but creates a pathway for Phase 4b symbolic-constraint refinements (rule-fired iff model also predicts fil_naqis). | +0.3 to +0.8 pp; mainly improves `khabar_kana` recall. |
| **harf_nafy** | 205 | 20 | harf_other | حرف نفي, حرف نفي وجزم, … | Negation particle (لا، لم، ما، لن). Affects the next verb's mood (jussive for لم/لا of jazm; subjunctive for لن). Currently buried in `harf_other` so the model can't learn the negation→mood interaction directly. | Limited Gazelle (small construction n) but expected to help on MASAQ Quranic verses where negation is more frequent. | +0.2 pp Gazelle role-F1, **+0.5–1 pp MASAQ marker-EM** via mood interaction. |
| **harf_nasb** | 330 | 12 | harf_other | حرف توكيد ونصب, حرف ناصب, حرف نصب | Accusative-marker particle (إن، أن، لكن، ليت، لعل، لن). Already implicit in the `inna_sisters` constraint trigger list — this label EXPOSES the constraint's trigger to the encoder, letting the encoder learn the trigger surface forms directly. | **Targets `mubtada → ism_inna` confusions** (1+ occurrences in Gazelle). Improves `ism_inna` recall by giving the encoder a label to attach to the trigger word itself. | +0.5 to +1.5 pp role-F1; +0.5 pp on `ism_inna` per-class F1 specifically. |
| **harf_tahqiq** | 157 | 6 | harf_other | حرف تحقيق, حرف عطف وتحقيق | Verification particle (قد). Doesn't change case but changes aspectual reading (e.g. *قد ضرب* = "indeed struck"). Borderline support; 6 val examples is the smallest in this set — flagged for the auto-fallback policy in §16. | None on Gazelle (no قد in held-out). | Negligible (≤+0.1 pp); ships if val accuracy ≥ 80%, otherwise auto-collapsed back to `harf_other`. |
| **mafoul_mutlaq** | 265 | 13 | mafoul_other | مفعول مطلق | Cognate accusative — emphasises a verb's action (e.g. *سار سيرا حثيثا*). Always nasb; unusual in that the word repeats the verb's root. Currently in `mafoul_other` (a 56-example bucket of paraphrastic variants). Splitting it reduces label-space noise. | Indirect; reduces noise on `mafoul_other`. | +0.1 to +0.3 pp role-F1. |

### 15.2 Parent residual after splitting

| Parent canonical | Total raw mass | Consumed by splits | Residual after split | Distinct raw strings remaining |
|---|---:|---:|---:|---:|
| dharf | 2,333 | 2,009 | 324 | ≈6 |
| fil | 5,653 | 2,300 | 3,353 | 60 |
| harf_other | 1,144 | 674 | 470 | 45 |
| mafoul_other | 316 | 265 | 51 | 5 |

The parent residuals (`dharf`, `fil`, `harf_other`, `mafoul_other`) all retain ≥ 50 examples after splitting, so they are not over-thinned. `fil` retains 3,353 generic-verb examples — the largest residual; this is intentional because `فعل` (generic, Aspect=und in source) is itself the biggest sub-cluster and should remain a label rather than be force-classified into one of the three new sub-types.

### 15.3 Aggregate support distribution after the expansion

```
Support range    # of canonical labels
≥ 5,000              4   (mudaaf_ilayh, ism_majrur, naat, badal)
1,000 – 4,999        7   (harf_jarr, mubtada, fil residual, dharf_zaman, fil_mudari, harf_atf, …)
  500 – 999          7
  200 – 499          7   (incl. harf_nasb, mafoul_mutlaq, ism_kana, …)
  150 – 199          1   (harf_tahqiq — borderline; auto-fallback safeguarded)
  < 150              0   ← all expansion labels except harf_tahqiq cleanly above the floor
```

`harf_tahqiq` is the only borderline. Per req #8 it is auto-collapsed to `harf_other` at evaluation if its native macro-F1 stays below 60% — guarding against poisoning the macro.

---

## 16. Auto-fallback policy (req #8)

Each new Phase 4a label has a frozen parent fallback target (already in `NEW_TO_OLD`). At training time we don't override anything — every label is supervised normally.

At evaluation time we additionally compute:

- **Native macro-F1** (over 34 native labels)
- **Native macro-F1 with auto-fallback** — any new label whose support is < 50 in the held-out set OR whose native F1 < 60% gets collapsed to its parent BEFORE macro-F1 is computed; this prevents one rare class from dragging macro-F1 down

Both numbers are reported. The auto-fallback view is the "stable" macro for ship-decision purposes; the raw view is informational. Practically only `harf_tahqiq` is at risk of triggering auto-fallback in Phase 4a.

A label that auto-falls-back at eval time is not removed from training — the encoder still gets supervision on the discrimination. We just don't penalise the macro-F1 reporting if the held-out tail is too thin to score reliably.

---

## 17. Four metric streams (req #6) — separating genuine reasoning from surface matching

Each Phase 4a eval surface (Gazelle, MASAQ, UD-PADT) reports four orthogonal metric streams so the source of any gain is identifiable:

| Stream | What it measures | Whose role-F1 are we matching against? |
|---|---|---|
| **A — Native canonical (34-label)** | Phase 4a's full label set. Predicted role argmax → 34-label native space, scored against gold canonicalised to v4. | New richer view; shows internal granularity quality. |
| **B — Grouped canonical (25-label)** | Phase 4a's predictions COLLAPSED via `NEW_TO_OLD` to the 25-label rev2/Phase1 space, scored against the same gold canonicalised to v3. | **Apples-to-apples vs rev 2 + Phase 1.** This is the headline metric for ship decisions. |
| **C — Raw-string overlap** | For every prediction, render to prose via the deterministic template renderer; ignore the model's structured field; check whether the rendered prose contains any of the gold's raw role-string substrings. Reports the % of words where the rendered surface is recognisable as the gold's raw category. | Tests whether gains come from the model learning genuine grammatical reasoning (would predict the right role even if its surface form differs from the gold's exact wording) vs gains coming from the model copying surface patterns. |
| **D — Extractor-surface match** | For every prediction's rendered prose, run the existing `evaluation/structural.py` extractor and compare the extracted role to the gold's extracted role. This is the metric the original paper reports. | Tests whether the renderer + extractor pipeline preserves the model's structured prediction faithfully. |

**Genuine-reasoning vs surface-matching diagnostic:** if Stream B (grouped) and Stream A (native) BOTH improve over Phase 1, the model learned finer reasoning. If Stream A improves but Stream B doesn't, the granularity helped internal predictions but the rev 2 / Phase 1 view doesn't see the gain — ship-as-opt-in territory. If Stream A improves but Stream C (raw-string overlap) is FLAT, the model is internally over-fitting to canonical IDs without changing rendered surfaces — interpretability concern, ship-as-opt-in.

---

## 18. Taxonomy stress table (req #7)

A separate table reported alongside the headline numbers, computed on the held-out Phase 4a val + Gazelle:

| Stress dimension | Definition | Phase 1 baseline | Phase 4a target |
|---|---|---:|---:|
| **Rare-role macro-F1** | Macro-F1 over the 12 lowest-support canonical labels (those with < 500 train support: naib_fail, mafoul_other, ism_inna, khabar_inna, ism_kana, khabar_kana, mafoul_mutlaq, harf_tahqiq, harf_nafy, hal, tamyeez, munada). Stress-tests the long-tail. | Phase 1 macro on the rev2 12 rare classes | ≥ Phase 1 (within bootstrap CI) |
| **Head-role stability** | Macro-F1 over the 8 highest-support canonical labels (mudaaf_ilayh, ism_majrur, naat, badal, harf_jarr, mubtada, harf_atf, mafoul_bih). Asks: did expansion destabilise the well-supported core? | Phase 1 macro on those 8 | ≥ Phase 1 -1 pp |
| **Long-tail collapse** | Number of distinct labels whose held-out F1 < 50% after expansion vs before. | Phase 1 count | ≤ Phase 1 + 1 |
| **Calibration drift** | Mean confidence on correct role predictions − mean on wrong, on Gazelle. Asks: did the larger label space inflate uncertainty? | Phase 1 calibration gap (typically ~0.15) | ≥ Phase 1 -0.05 |

If any of {rare-role macro, head-role stability, long-tail collapse, calibration} crosses its red line, the Phase 4a result is downgraded to opt-in regardless of the headline grouped role-F1.

---

## 19. Re-stated success rule (with the 4 metric streams + stress table)

**Ship Phase 4a as a default candidate** if ALL of:

1. Stream B (grouped, 25-label) role-F1 on Gazelle ≥ Phase 1 (42.3) within paired-bootstrap CI.
2. Stream B fully on Gazelle ≥ Phase 1 (19.4) within CI.
3. Stream B case + marker on Gazelle ≥ Phase 1 within CI.
4. Stream A (native, 34-label) role-F1 ≥ Stream B role-F1 (i.e., the finer view at least matches the coarser).
5. Stream C (raw-string overlap) ≥ Phase 1 within CI (the model's gain isn't a surface artifact).
6. Stress table: rare-role macro within CI of Phase 1; head-role within −1 pp; long-tail collapse Δ ≤ +1; calibration drift ≥ −0.05.
7. MASAQ grouped role-F1 within ±1 pp of Phase 1 (cross-register stability).

**Ship as opt-in only** if 1–4 hold but 5 OR the stress table fails one criterion.
**Don't ship, debug + retry** if 1–3 fail or stress table fails > 1 criterion.

---

## 20. Step order for the 2×2 ablation matrix

| Step | What runs | Job | Wall clock |
|---|---|---|---|
| Frozen | rev 2 (already trained, no retrain) | 490946 (existing) | 0 |
| Frozen | Phase 1 (already trained, no retrain) | 490987 (existing) | 0 |
| Retrain 1 | **Phase 4a-no-morph** (taxonomy_v4 + no morph heads) | new sbatch 62a | ~7 min |
| Retrain 2 | **Phase 4a-morph** (taxonomy_v4 + morph heads on) | new sbatch 62b | ~7 min |
| Eval all 4 | Streams A/B/C/D × Gazelle/MASAQ + UD-PADT-morph for the two morph runs + stress table | new sbatch 63 | ~25 min |

Total Phase 4a HPC time: ~40 min.

For the retrain commits, the seed is fixed at 42 (matches rev 2 / Phase 1 for reproducibility). Each retrain's `runs/phase4a_*/<JOBID>/provenance.txt` records: git_commit, seed, config, started_at, completed_at — same provenance discipline as Phase 1.

---

## 21. Findings (filled after the three Phase 4a retrains)

(To be populated.)
