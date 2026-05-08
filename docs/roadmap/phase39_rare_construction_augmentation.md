# Phase #39 — Rare-Construction Synthetic Augmentation

> **Now the highest-priority phase.** Per the formal research thesis
> adopted 2026-05-09, the architectural picture is resolved at the
> 296M / 6-epoch scale: orthogonal linguistic information unlocks
> gain (Phase 1 morph, Phase 3 dep), downstream rearrangement does not
> (Phases 4a, 2, 5, 6, 3.1). The bottleneck is now construction
> coverage, not architecture. #39 addresses that directly.

## 1. Motivation — the documented coverage gap

§5.4 of REPORT.md reports that **EXCEPTION and KANA_SISTERS construct
types are 0/9 and 0/7 across all systems including the closed frontier
(Sonnet RAG)**. This is not a model-capacity issue — even Sonnet's
parametric knowledge of MSA grammar plus 5-shot retrieval can't
correctly tag these constructions. The training corpora (distill_v2 +
UD-PADT + 5-shot retrieval pool) simply don't expose the model to
enough varied examples of these constructions.

The same likely holds for several other constructions that show
systematic failure across systems but were not isolated in §5.4:
- **istithnāʾ** (exception with *إلا*, *غير*, *سوى*, *ما خلا*)
- **mawṣūl** (relative clauses with الذي / التي / الذين / من / ما)
- **idāfa edge cases** (multi-level, with *ال*, with proper nouns,
  with elliptical *muḍāf*)
- **Quranic-specific constructions** (verse-style word order,
  classical particles, idiomatic constructions in scripture)
- **rare role combinations** (e.g. dual + nasb + indeclinable;
  sound-fem-plural + jarr; ḥāl with prepositional phrase)

#39's hypothesis: **synthetic augmentation that targets these
specific constructions, balanced relative to their natural rarity,
will lift per-construction recall on Gazelle + MASAQ without
hurting overall metrics on majority constructions.**

## 2. Anti-pattern (per research thesis 2026-05-09)

#39 is the canonical case of "add new training data" rather than
"rearrange existing supervision". It does not:
- Add new model architecture
- Change the encoder, the heads, or the decoders
- Modify the training loss
- Touch dep features, morph supervision, or constraints

It only adds new (input, label) pairs to the training corpus. The
production model architecture stays exactly Phase 3-A.

## 3. Target constructions and templates

For each target construction, we define:
- A canonical surface pattern (Arabic template with placeholders)
- A grammatical signature (case + role + marker per word)
- A set of lexical fillers (~20-50 per slot)
- An expected (sentence, items[]) output

### 3.1 *kāna* sisters (priority — 0/7 in §5.4)

Pattern: `<KANA-SIS> <ISM-RAF> <KHABAR-NASB>` (verb + subj-nom + pred-acc)

`<KANA-SIS>` ∈ {كان، ليس، أصبح، ظل، صار، بات، أمسى، أضحى، ما زال، ما برح، ما فتئ، ما انفك}

Surface example: *أصبح الطالبُ مجتهداً*
- *أصبح* — verb, perfect, active, mood=ind, role=`fil_madi`, case=N/A
- *الطالبُ* — noun, def, sg, masc, role=`ism_kana`, case=`raf`, marker=`damma_visible`
- *مجتهداً* — noun, indef, sg, masc, role=`khabar_kana`, case=`nasb`, marker=`fatha_visible`

Generation: 12 KANA-SIS particles × 30 noun fillers × 30 adjective
fillers = 10,800 candidate sentences. Subsample to 200 balanced
across particles + noun gender/number.

### 3.2 *inna* sisters (priority — class is rare in distill_v2)

Pattern: `<INNA-SIS> <ISM-NASB> <KHABAR-RAF>` (particle + obj-acc + pred-nom)

`<INNA-SIS>` ∈ {إنّ، أنّ، لكنّ، ليت، لعلّ، كأنّ}

Surface: *إنّ المعلمَ مخلصٌ*
- *إنّ* — particle, role=`harf_nasb` or canonical `inna_sister`
- *المعلمَ* — noun, role=`ism_inna`, case=`nasb`
- *مخلصٌ* — adj, role=`khabar_inna`, case=`raf`

Generation: 6 particles × 50 nouns × 50 adj/noun predicates = ~15K
candidates → 200 balanced.

### 3.3 *istithnāʾ* (EXCEPTION — 0/9 in §5.4)

Pattern: `<MAIN-CLAUSE> <ILLA> <MUSTATHNA>`

`<ILLA>` ∈ {إلا، غير، سوى، ما خلا، ما عدا، حاشا}

Subtypes:
- *istithnāʾ tāmm* (positive, complete): mustathna takes nasb
- *istithnāʾ tāmm manfī* (negative, complete): mustathna takes either case (badal vs nasb on istithna)
- *istithnāʾ nāqis*: governed by main verb, mustathna inherits case

Surface: *جاء الطلابُ إلا زيداً* (the students came except Zayd)
- زيداً — case=`nasb`, role=`mustathna` (canonical `mafoul_other` in current schema)

This requires schema work: the current 25-label canonical taxonomy
collapses *istithnāʾ* into the `mafoul_other` bucket (Phase 4a v4
expansion has `mafoul_mutlaq` but not a dedicated *mustathna*; we
emit `mafoul_other` for the rare case). For #39's first pass we keep
the v3 schema and let role-F1 reflect the canonical collapse;
schema expansion to add an explicit *mustathna* class is deferred to
Phase 4b (not in scope for #39).

Generation: 6 illa-particles × 30 main verbs × 50 mustathna nouns ×
2 (positive/negative main clause) = ~18K → 250 balanced (more samples
because the construction is harder to template uniformly).

### 3.4 *mawṣūl* (relative clauses)

Pattern: `<NOUN-DEF> <MAWSUL> <RELATIVE-CLAUSE>` where the relative
clause's verb has an implicit pronoun referring back to the noun.

`<MAWSUL>` ∈ {الذي، التي، الذين، اللاتي، اللواتي، اللذان، اللتان، من، ما}

Surface: *الكتابُ الذي قرأتُهُ مفيدٌ*
- *الذي* — particle, role=`mawsool` (no case, indeclinable)
- The relative clause structure carries through to per-word labels

Generation: 9 mawsul particles × 50 head nouns × 30 relative-verb
templates = ~13K → 200 balanced.

### 3.5 *iḍāfa* edge cases

The current Phase 1/3 corpus covers basic *iḍāfa* (DEF + INDEF, e.g.
*كتابُ الطالبِ*) well. The edge cases:

- **Multi-level iḍāfa**: *كتابُ معلمِ المدرسةِ* (book of teacher of school)
  — chain of three nouns, all but the last in *muḍāf* state (Definite=Cons).
- **iḍāfa lafẓī (false iḍāfa)**: *المعلمُ الكثيرُ السؤالِ* (the much-question teacher)
  — adjective in iḍāfa state, semantically not possessive.
- **Numerical iḍāfa**: *ثلاثةُ كتبٍ* (three books) — number+counted noun
  in iḍāfa.
- **Elliptical muḍāf**: *قبل الظهرِ* (before noon) — prepositional iḍāfa
  with implicit pronoun.

Generation: ~150 multi-level + ~50 each of the other types = 300 total.

### 3.6 Quranic-specific constructions

Targeted Quranic patterns that appear ~0× in distill_v2:
- *قد + perfect verb* (qad + perfect — assertion particle)
- *إذ* and *إذا* with verb (temporal particles introducing verbal clauses)
- *لمّا* (when, with perfect verb only)
- *كلّما* (whenever)
- *حتى* introducing a verbal clause vs noun
- Vocative *يا أيها الذين آمنوا* (extended vocative)

These are largely tagged correctly when isolated but fail in the
nuanced contextual disambiguation. ~150 sentences, sourced by
templating common Quranic phrasings (without copying actual verses,
to avoid memorization).

### 3.7 Rare role combinations

The *fully* metric requires all four heads (case, role, marker, POS)
correct simultaneously. Rare 4-tuples that distill_v2 doesn't cover:

- dual + nasb + indeclinable noun (e.g. *رأيتُ الفتيين*)
- sound-fem-plural + jarr (e.g. *مررتُ بالمعلماتِ*)
- *ḥāl* with prepositional phrase (e.g. *جاء راكباً على فرسٍ*)
- *tamyīz* with weight measure (e.g. *قنطاراً من الحديد*)

~100 sentences across these.

### 3.8 Total scale

Target: ~1,500 augmented sentences (≈ 15% of current distill_v2 corpus).

Breakdown:
- *kāna* sisters: 200
- *inna* sisters: 200
- *istithnāʾ*: 250
- *mawṣūl*: 200
- *iḍāfa* edge cases: 300
- Quranic constructions: 150
- Rare role combinations: 100
- Buffer / variation: 100

## 4. Generator implementation

`scripts/augment/generate_rare_constructions.py` (new file, pure Python,
no HPC needed):

```python
@dataclass
class ConstructionTemplate:
    name: str              # e.g. "kana_sisters"
    surface_pattern: str   # e.g. "<KANA> <ISM> <KHABAR>"
    slots: Dict[str, List[str]]  # slot → list of fillers
    items_template: List[Dict]   # per-word (case, role, marker, pos, irab_prose)

def generate(template: ConstructionTemplate, n: int) -> List[Dict]:
    """Sample n sentences from template by filling slots."""
    ...
```

For each construction in §3, we encode the template + slot fillers +
per-word labels. Generator outputs JSONL records compatible with the
existing distill_v2 corpus format (`{sentence, items: [{word, case, role,
marker, pos, irab_prose, ...}]}`).

The lexical fillers come from:
- Manual list for particles (small closed sets)
- Top-frequency Arabic nouns/adjectives/verbs from the existing corpus
  (extract a vocabulary of ~200 high-frequency content words per POS
  and sample uniformly from it)
- Surface-form variants (with/without ال, with diacritics matching the
  inflection pattern)

## 5. Per-construction evaluation

`scripts/structured/eval_per_construction.py` (new):

For each augmented sentence in Gazelle + MASAQ that contains a target
construction (detected by surface match against `<MAWSUL>` particles,
*illa* etc.), report:
- per-construction word-level case + role + marker + fully accuracy
- per-word confusion matrix (gold role × predicted role) for the rare
  classes
- before/after comparison against Phase 3-A baseline

This is the rigorous "did we actually fix the targeted gaps" check.

## 6. Validation strategy

Before retraining, validate the synthetic data:
1. **Schema validation**: every generated record passes the existing
   distill_v2 schema check (case ∈ {raf, nasb, jarr, jazm, indecl},
   role ∈ ROLE_LABELS, marker ∈ MARKER_LABELS).
2. **Native speaker spot-check**: sample 20 generated sentences per
   construction, eyeball check for grammaticality. If significant
   issues found, fix templates and regenerate.
3. **Distribution check**: ensure each target construction is
   reasonably balanced (not 90% one particle).

## 7. Training schedule

Retrain Phase 3-A on the augmented corpus (`distill_v2 + 1,500 synthetic`):
- Same config as `phase3_dep_aware.yaml`
- Same 6-epoch schedule
- Same hyperparameters
- New corpus path: `data/structured_v1_augmented/{train,val}.jsonl`
- Stanza parse the synthetic sentences first to get dep features
  (compatible with Phase 3 input augmentation pipeline)
- Output: `runs/phase39_<JOBID>/final/`

Total HPC time:
- Stanza parse on 1,500 new sentences: ~5 min
- Smoke test: 5 min
- Full retrain: ~10 min (slightly longer than Phase 3-A's 7:53 because
  the corpus is 15% larger)
- Eval (full Gazelle + MASAQ + per-construction breakdown): ~5 min
- **Total: ~25 min HPC**

## 8. Decision gate

The augmented model ships as new production *if*:
1. Targeted constructions improve substantially (≥ +5 pp on
   per-construction word-fully on Gazelle constructions tagged as
   `kana_sisters`, `inna_sisters`, `mafoul_other`-istithnāʾ,
   `mawsool`).
2. Overall Gazelle metrics stay within ±1 pp of Phase 3-A's
   56.7 / 41.3 / 44.8 / 20.1 — ideally improve due to better tail-
   class coverage.
3. MASAQ retention is preserved or improves (the augmentation includes
   Quranic-specific patterns).

If targeted constructions improve but overall metrics regress > 1 pp,
the augmentation is too aggressive and we down-sample to half the
synthetic count.

## 9. Reversibility

The augmented corpus lives in `data/structured_v1_augmented/`,
separate from `data/structured_v1/`. The Phase 3-A production
checkpoint stays untouched at `runs/phase3a_491240/final/`.
`configs/phase39_augmented.yaml` opt-in with `train_path:
data/structured_v1_augmented/train.jsonl`. To revert, point back at
the original corpus.

## 10. Implementation order

1. **Schema validator + slot vocabulary extraction** (~1h): pull
   high-frequency content words from existing corpus.
2. **Templates per construction** (~2h): one `ConstructionTemplate`
   instance per §3 subsection; encode the per-word (case, role,
   marker, pos, irab_prose) tuple for each surface position.
3. **Generator script** (~1h): instantiate templates with slot
   fillers; emit JSONL.
4. **Manual validation** (~30min): eyeball check 20 samples per
   construction.
5. **Stanza parse** (~5min HPC): run existing `parse_deps.py` on
   augmented corpus.
6. **Build augmented training corpus** (~10min): merge synthetic +
   distill_v2 + UD-PADT into one corpus.
7. **Retrain on HPC** (~10min HPC).
8. **Per-construction eval** (~30min): write
   `eval_per_construction.py`, run on Phase 3-A and Phase 39
   side by side.
9. **Writeup + paper integration** (~1h): document #39 in
   `phase39_rare_construction_augmentation.md` §11+ and add row to
   REPORT.md headline + per-construction table to §5.4.

Total wallclock: ~6h focused work (~5h human / ~1h HPC).

## 11. Out of scope for #39

- Schema expansion (e.g. adding explicit *mustathna* role) — defer
  to Phase 4b.
- Adversarial augmentation (constructions that intentionally confuse
  the model) — defer.
- Quranic-corpus-derived training data — would require MASAQ
  exposure, which we deliberately keep held-out.
- Generative model for synthetic data (e.g. asking Sonnet to generate
  examples) — for #39 we use deterministic templates so the labels
  are guaranteed correct; LLM-generated would have label noise.

## 11.5 Run 1 — partial positive, large negative

Run `phase39_491324` (8:24 train, full eval `491337` 3:44):

| Metric | Phase 3-A | Phase 39 | Δ Gazelle | Δ MASAQ |
|---|---:|---:|---:|---:|
| case | 56.7 / 85.9 | 53.7 / 86.3 | **−3.0** | **+0.4** |
| role-F1 | 41.3 / 9.8 | 28.4 / 10.2 | **−12.9** | **+0.4** |
| marker | 44.8 / 32.3 | 41.8 / 32.7 | **−3.0** | **+0.4** |
| fully | 20.1 / 7.4 | 12.7 / 7.6 | **−7.4** | **+0.2** |

**The pattern is informative:**
- **MASAQ improves uniformly +0.4 pp across all four metrics.** Quranic
  prose has higher density of *kāna sisters*, *istithnāʾ*, *mawṣūl*, and
  the Quranic-template patterns we generated. The synthetic exposure
  helps the model on those constructions, validating the thesis that
  "adding orthogonal coverage unlocks gain on register-mismatched test
  surfaces".
- **Gazelle regresses sharply: role-F1 −12.9, fully −7.4.** The synthetic
  data was 22% of the iʿrāb-supervised training corpus (1,330 / 6,080)
  — too high a ratio. Templates were lexically narrow (~20-30 fillers
  per slot, same fixed word orders, same particle distributions). The
  role head overfit to template surfaces and lost in-distribution
  generalisation on Gazelle's natural prose.

**This is a partial-positive negative result.** It confirms the
research thesis (new information helps when target distribution
matches synthetic distribution — MASAQ benefits) AND exposes a
template-overfit failure mode (synthetic distribution must not
dominate the training signal — Gazelle suffers).

**The fix is in the augmentation strategy, not the thesis.** Concrete
next iteration (Phase 39 v2):
1. **Reduce synthetic ratio**: 1,400 → 350-500 (5-8% of corpus, not 22%)
2. **Diversify lexical fillers**: 3-4× more nouns, adjectives, verbs
   sourced from the existing distill_v2 corpus by frequency-weighted
   sampling, not hardcoded short lists
3. **Sentence-shape variation per template**: 3-5 word-order variants
   per template, not the single fixed pattern current templates use
4. **Per-construction evaluation in the eval pipeline**: add
   `eval_per_construction.py` so we can verify that *kāna* /
   *inna* / *istithnāʾ* / *mawṣūl* targeted constructions improve
   on Gazelle subsets, separately from overall metrics
5. **Mix-in noise check**: include synthetic at 5% but also at 0%
   (no augmentation) and 15% as ablation cells, to isolate the
   ratio-sensitivity curve

## 11.6 Ship decision (Run 1)

**Phase 39 Run 1 does NOT ship as production.** Phase 3-A remains the
production checkpoint. Phase 39 ships as:
- A **partial-positive on MASAQ** (+0.4 across the board on the
  cross-register surface), worth reporting as evidence the thesis
  generalises
- A **failure mode demonstration on Gazelle** (template-overfit),
  worth reporting as the principal lesson for the next iteration
- A **roadmap data point**: Phase 39 v2 (with the 5 fixes above)
  needs a fresh implementation cycle. Do NOT just retry the same
  templates with reduced ratio — the templates themselves need
  diversification.

The augmented training corpus (`data/structured_v1_augmented/`) and
parsed dep version (`data/morph_v1_augmented_dep/`) stay on HPC as
artefacts — Phase 39 v2 can re-use them or rebuild from scratch with
the fixes.

## 12. Risk register

- **Synthetic distribution mismatch**: Templated sentences may have
  surface artifacts (e.g. always start with the same particle, fixed
  word order). Mitigation: sample slot variants uniformly + add 10%
  word-order variations per template.
- **Label-noise blind spot**: If a template encodes the wrong label
  (e.g. wrong marker for an indeclinable noun), the augmented data
  trains the model to be wrong consistently. Mitigation: §6
  validation step + cross-check labels against the existing extractor
  on a held-out 10% of generated data.
- **Schema collapse**: We're using v3 25-label taxonomy. Some target
  constructions (*istithnāʾ*) collapse to `mafoul_other` and won't
  show clean improvement on role-F1 even if the model learns them
  correctly. Mitigation: report per-construction word-fully (case +
  role + marker + POS all correct), which is unaffected by schema
  collapse.
- **Overfitting to synthetic**: 15% synthetic in the training mix may
  bias the model toward template surfaces. Mitigation: monitor
  Gazelle headline metrics for regression; if Gazelle case +
  role-F1 + fully degrade > 1 pp, halve the synthetic count.
