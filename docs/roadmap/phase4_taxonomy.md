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
