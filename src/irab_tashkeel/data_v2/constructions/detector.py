"""Construction-detector pass for schema_v2.

Populates ``Sentence.constructions`` from the canonical particle
taxonomy + per-token role labels. Replaces the flat
``grammar_memory/signature.py::detect_constructions_in_record`` of
the frozen baseline with a schema_v2-native version that:

- Produces full :class:`Construction` objects with
  ``token_indices`` / ``head_idx`` / ``children_indices`` /
  ``particle_surface`` / ``subgroup``.
- Computes overlap diagnostics (how many tokens are covered by
  ≥ 2 constructions).
- Assigns ``ambiguity_score`` based on detected overlap and parser
  confidence.
- Marks ``alternative_analyses`` when multiple families could fit
  the same span.
- Populates ``agreement_relations`` when both tokens have morph.

Called as a post-loader pass; loaders themselves do not run
construction detection (keeps loaders deterministic + reusable).

Usage::

    from data_v2.loaders.distill2 import Distill2Loader
    from data_v2.constructions.detector import detect_constructions_pass
    sents = Distill2Loader(root=".").load_all()
    detect_constructions_pass(sents)             # in-place population

Detection sources
-----------------

The detector uses the following signals (in order):

1. **Particle surface** (kana / inna / istithna / mawsool / quranic)
   — string match on the normalised surface form.
2. **Multi-word particles** (``ما عدا``, ``ما خلا``) — bigram match.
3. **Per-token role label** (idafa) — gold or predicted ``role``
   value of ``mudaaf_ilayh`` triggers iḍāfa span detection.

If a token has no role label set, role-based detection is skipped
for that token. Surface-particle detection still works.

Provenance
----------

All detector-emitted constructions carry
``source="bronze_heuristic"`` and the appropriate
``confidence`` per family. Higher confidence is assigned for
unambiguous surface-particle hits (kana ``كان``, inna ``إن``) than
for ambiguous tokens (``إن`` can also be a conditional particle,
``ما`` can be a relative pronoun OR a negation OR ``ما عدا``).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

from ..schema_v2 import Construction, Sentence


# ===========================================================================
# Particle taxonomy (frozen-baseline canonical, extensible)
# ===========================================================================

FAMILIES: Dict[str, Dict[str, List[str]]] = {
    "kana_sisters": {
        "kana_completion":  ["كان", "صار", "أصبح", "أمسى", "أضحى", "بات", "ظل",
                              "كانت", "صارت", "أصبحت", "أمست", "باتت", "ظلت"],
        "kana_negation":    ["ليس", "ليست", "زال", "برح", "فتئ", "انفك"],
    },
    "inna_sisters": {
        "inna_assertion":   ["إن", "أن", "إنّ", "أنّ"],
        "inna_modal":       ["ليت", "لعل", "كأن", "لكن", "كأنّ", "لكنّ", "لعلّ"],
    },
    "istithna": {
        "illa":             ["إلا", "إلّا"],
        "istithna_noun":    ["غير", "سوى"],
        "ma_3ada_phrase":   ["ما عدا", "ما خلا"],     # multi-word
        "hasha":            ["حاشا"],
    },
    "mawsool": {
        "definite_relative":   ["الذي", "التي", "الذين", "اللاتي", "اللواتي",
                                 "اللذان", "اللتان", "اللذين", "اللتين"],
        "indefinite_relative": ["من", "ما"],
    },
    "idafa":         {"any": []},
    "idafa_multi":   {"any": []},
    "quranic_proxy": {
        "qad_idh":  ["قد", "إذ", "إذا"],
        "lamma":    ["لما", "لمّا"],
        "kullama":  ["كلما", "كلّما"],
        "hatta":    ["حتى"],
    },
}

# Per-family base confidence — surface-particle detection has these
# defaults; ambiguous particles like "إن" / "من" / "ما" get penalised
# below.
BASE_CONFIDENCE: Dict[str, float] = {
    "kana_sisters":  0.90,    # surface particles unambiguous mostly
    "inna_sisters":  0.75,    # إن also conditional; lowered
    "istithna":      0.85,
    "mawsool":       0.70,    # من/ما heavily ambiguous
    "quranic_proxy": 0.65,    # qad/idh polysemous
    "idafa":         0.85,    # role-based detection is reliable
    "idafa_multi":   0.85,
}

# Ambiguous particles → confidence penalty
AMBIGUOUS_PARTICLES = {
    "إن", "أن",          # conditional vs assertion
    "من", "ما",         # relative vs interrogative vs negation
    "إذا",              # condition vs time
    "حتى",              # exception vs purpose vs goal
    "لما", "لمّا",
}


# ===========================================================================
# Surface normalisation for particle lookup
# ===========================================================================

def _norm_ar(s: str) -> str:
    """Lookup-friendly Arabic normalisation: strip diacritics + non-Arabic."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[ً-ٰٟ]", "", s)
    s = re.sub(r"[^ء-ي ]+", "", s)
    return s


# Pre-compute normalised lookup: norm(surface) → (family, subgroup)
_PARTICLE_LOOKUP: Dict[str, Tuple[str, str]] = {}
for fam, groups in FAMILIES.items():
    for grp, particles in groups.items():
        for p in particles:
            np_ = _norm_ar(p)
            if " " in p:
                continue          # multi-word handled separately
            if np_ and np_ not in _PARTICLE_LOOKUP:
                _PARTICLE_LOOKUP[np_] = (fam, grp)


# ===========================================================================
# Construction-emitter helpers
# ===========================================================================

def _make_construction(
    family: str, subgroup: str, token_indices: List[int],
    head_idx: int, particle_surface: str = "",
    children_indices: Optional[List[int]] = None,
    confidence: float = 0.85,
) -> Construction:
    return Construction(
        family=family,
        subgroup=subgroup,
        token_indices=list(token_indices),
        head_idx=head_idx,
        children_indices=list(children_indices or []),
        particle_surface=particle_surface,
        source="bronze_heuristic",
        confidence=confidence,
    )


# ===========================================================================
# Per-family detectors
# ===========================================================================

def _detect_particle_constructions(sentence: Sentence) -> List[Construction]:
    """Detect particle-based constructions (kana, inna, istithna, mawsool, quranic).

    Span: [particle_idx, particle_idx + 3] capped at sentence length.
    Children: positions inside the span (excluding the particle).
    """
    out: List[Construction] = []
    n = sentence.n_tokens
    for i, tok in enumerate(sentence.tokens):
        norm = _norm_ar(tok.surface)
        match = _PARTICLE_LOOKUP.get(norm)
        if match is None:
            continue
        family, subgroup = match
        end = min(n, i + 3)
        token_indices = list(range(i, end))
        children = list(range(i + 1, end))
        conf = BASE_CONFIDENCE.get(family, 0.7)
        if norm in AMBIGUOUS_PARTICLES:
            conf *= 0.85
        out.append(_make_construction(
            family=family, subgroup=subgroup,
            token_indices=token_indices, head_idx=i,
            particle_surface=tok.surface,
            children_indices=children, confidence=conf,
        ))
    return out


def _detect_multiword_particles(sentence: Sentence) -> List[Construction]:
    """Detect multi-word particles (ما عدا, ما خلا)."""
    out: List[Construction] = []
    n = sentence.n_tokens
    if n < 2:
        return out
    for i in range(n - 1):
        w1 = _norm_ar(sentence.tokens[i].surface)
        w2 = _norm_ar(sentence.tokens[i + 1].surface)
        joined = f"{w1} {w2}"
        for fam, groups in FAMILIES.items():
            for grp, particles in groups.items():
                for p in particles:
                    if " " not in p:
                        continue
                    if _norm_ar(p) == joined:
                        end = min(n, i + 4)
                        out.append(_make_construction(
                            family=fam, subgroup=grp,
                            token_indices=list(range(i, end)),
                            head_idx=i,
                            particle_surface=p,
                            children_indices=list(range(i + 1, end)),
                            confidence=BASE_CONFIDENCE.get(fam, 0.7),
                        ))
    return out


def _detect_idafa(sentence: Sentence) -> List[Construction]:
    """Detect iḍāfa (single + multi) from per-token role=mudaaf_ilayh.

    Skips silently if role labels aren't populated.
    """
    out: List[Construction] = []
    consecutive: List[int] = []

    def _flush(cons: List[int]):
        if not cons:
            return
        if len(cons) >= 2:
            span_start = max(0, cons[0] - 1)
            out.append(_make_construction(
                family="idafa_multi", subgroup="any",
                token_indices=list(range(span_start, cons[-1] + 1)),
                head_idx=span_start,
                children_indices=list(cons),
                confidence=BASE_CONFIDENCE["idafa_multi"],
            ))
        for idx in cons:
            span_start = max(0, idx - 1)
            out.append(_make_construction(
                family="idafa", subgroup="any",
                token_indices=list(range(span_start, idx + 1)),
                head_idx=span_start,
                children_indices=[idx],
                confidence=BASE_CONFIDENCE["idafa"],
            ))

    for i, tok in enumerate(sentence.tokens):
        role = tok.role.value
        if role == "mudaaf_ilayh":
            consecutive.append(i)
        else:
            _flush(consecutive)
            consecutive = []
    _flush(consecutive)
    return out


# ===========================================================================
# Overlap + ambiguity diagnostics
# ===========================================================================

def _compute_overlap_diagnostics(
    constructions: List[Construction], n_tokens: int,
) -> Dict[int, List[Construction]]:
    """For each token index, list of constructions that cover it."""
    cover: Dict[int, List[Construction]] = {}
    for c in constructions:
        for i in c.token_indices:
            cover.setdefault(i, []).append(c)
    return cover


def _bump_ambiguity(constructions: List[Construction], cover: Dict[int, List[Construction]]) -> None:
    """When ≥ 2 constructions cover the same token, raise their
    ambiguity_score and add them as alternative_analyses to each
    other (mutually).
    """
    seen_pairs: Set[Tuple[str, str]] = set()
    for i, cs in cover.items():
        if len(cs) < 2:
            continue
        for a in cs:
            other_summaries = []
            for b in cs:
                if a is b:
                    continue
                pair = tuple(sorted([a.construction_id, b.construction_id]))
                if pair in seen_pairs:
                    continue
                # Mutually note as alternative analysis
                a.ambiguity_score = max(a.ambiguity_score, 0.3)
                other_summaries.append({
                    "construction_id": b.construction_id,
                    "family": b.family, "subgroup": b.subgroup,
                    "head_idx": b.head_idx,
                    "token_indices": list(b.token_indices),
                })
                seen_pairs.add(pair)
            for s in other_summaries:
                if s not in a.alternative_analyses:
                    a.alternative_analyses.append(s)


def _populate_agreement_relations(sentence: Sentence, constructions: List[Construction]) -> None:
    """When both tokens of a kana / inna construction have morph.gender
    or morph.number populated, record an agreement relation between
    the ism and khabar position (positions 1 and 2 of the span).
    """
    for c in constructions:
        if c.family not in ("kana_sisters", "inna_sisters"):
            continue
        if len(c.token_indices) < 3:
            continue
        ism_idx = c.token_indices[1]
        khabar_idx = c.token_indices[2]
        if ism_idx >= sentence.n_tokens or khabar_idx >= sentence.n_tokens:
            continue
        ism_morph = sentence.tokens[ism_idx].morph
        khabar_morph = sentence.tokens[khabar_idx].morph
        axes: List[str] = []
        for axis_name in ("gender", "number"):
            ism_v = getattr(ism_morph, axis_name).value
            kh_v = getattr(khabar_morph, axis_name).value
            if ism_v and kh_v and ism_v == kh_v:
                axes.append(axis_name)
        if axes:
            c.agreement_relations.append((ism_idx, khabar_idx, axes))


# ===========================================================================
# Public API
# ===========================================================================

def detect_constructions(sentence: Sentence) -> List[Construction]:
    """Return the list of constructions for a single sentence.

    Does NOT mutate ``sentence``. Caller is responsible for assigning
    the result to ``sentence.constructions``.
    """
    constructions: List[Construction] = []
    constructions += _detect_particle_constructions(sentence)
    constructions += _detect_multiword_particles(sentence)
    constructions += _detect_idafa(sentence)

    if constructions:
        cover = _compute_overlap_diagnostics(constructions, sentence.n_tokens)
        _bump_ambiguity(constructions, cover)
        _populate_agreement_relations(sentence, constructions)

    return constructions


def detect_constructions_pass(sentences: List[Sentence]) -> int:
    """In-place: run construction detection over a list of sentences.

    Updates ``sentence.constructions`` and ``sentence.completeness.
    has_constructions``. Returns count of sentences with ≥ 1 detection.
    """
    n_with = 0
    for s in sentences:
        cs = detect_constructions(s)
        s.constructions = cs
        s.completeness.has_constructions = bool(cs)
        if cs:
            n_with += 1
    return n_with


def overlap_summary(sentence: Sentence) -> Dict[str, int]:
    """Diagnostic: count of tokens covered by 0 / 1 / 2 / 3+ constructions."""
    cover = _compute_overlap_diagnostics(sentence.constructions, sentence.n_tokens)
    out = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for i in range(sentence.n_tokens):
        c = len(cover.get(i, []))
        if c == 0:
            out["0"] += 1
        elif c == 1:
            out["1"] += 1
        elif c == 2:
            out["2"] += 1
        else:
            out["3+"] += 1
    return out


def clause_consistency_check(sentence: Sentence) -> List[str]:
    """Return list of consistency-violation messages (empty when OK).

    Checks:
      - construction.token_indices are within [0, n_tokens)
      - construction.head_idx is in token_indices
      - children_indices ⊆ token_indices
      - clause_id (when set) refers to an existing clause
    """
    issues: List[str] = []
    n = sentence.n_tokens
    clause_ids = {c.clause_id for c in sentence.clauses}
    for c in sentence.constructions:
        for i in c.token_indices:
            if not (0 <= i < n):
                issues.append(f"{c.construction_id}: token_idx {i} out of range")
        if c.head_idx is not None and c.head_idx not in c.token_indices:
            issues.append(f"{c.construction_id}: head_idx {c.head_idx} not in token_indices")
        for i in c.children_indices:
            if i not in c.token_indices:
                issues.append(f"{c.construction_id}: child {i} not in token_indices")
        if c.clause_id and c.clause_id not in clause_ids:
            issues.append(f"{c.construction_id}: clause_id {c.clause_id} unknown")
    return issues
