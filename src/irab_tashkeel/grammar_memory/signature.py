"""Phase R — construction signature for retrieval.

Defines:
- :class:`ConstructionInstance` — one record per construction occurrence
  in the training corpus.
- Construction family + particle group taxonomy.
- :func:`detect_constructions_in_record` — find construction spans in
  a training record (reuses the surface + role logic from
  ``scripts/structured/eval_per_construction.py``).
- :func:`build_signature` — produce a ConstructionInstance from a span +
  encoder span embedding.

The signature is the unit indexed by :class:`GrammarMemory`. At inference
time, query signatures are built the same way and matched against the
indexed pool.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Construction families and particle groups (frozen)
# ---------------------------------------------------------------------------

# Particle groups are subgroups within a family — a kana-completion sentence
# (أصبح) shouldn't retrieve a kana-negation sentence (ليس) because their
# syntactic behaviour differs. Particle group is part of the symbolic filter.
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
        "ma_3ada_phrase":   ["ما عدا", "ما خلا"],
        "hasha":            ["حاشا"],
    },
    "mawsool": {
        "definite_relative":   ["الذي", "التي", "الذين", "اللاتي", "اللواتي",
                                "اللذان", "اللتان", "اللذين", "اللتين"],
        "indefinite_relative": ["من", "ما"],
    },
    "idafa":         {"any": []},        # detected by role==mudaaf_ilayh, no particle
    "idafa_multi":   {"any": []},
    "quranic_proxy": {
        "qad_idh":  ["قد", "إذ", "إذا"],
        "lamma":    ["لما", "لمّا"],
        "kullama":  ["كلما", "كلّما"],
        "hatta":    ["حتى"],
    },
}

ALL_FAMILIES: List[str] = list(FAMILIES.keys())


def _norm_ar(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[ً-ٰٟ]", "", s)
    s = re.sub(r"[^ء-ي]+", "", s)
    return s


# Pre-compute normalised lookup tables: norm_surface -> (family, particle_group)
_PARTICLE_LOOKUP: Dict[str, Tuple[str, str]] = {}
for fam, groups in FAMILIES.items():
    for grp, particles in groups.items():
        for p in particles:
            np_ = _norm_ar(p)
            if np_ and np_ not in _PARTICLE_LOOKUP:
                _PARTICLE_LOOKUP[np_] = (fam, grp)


def lookup_particle(surface: str) -> Optional[Tuple[str, str]]:
    """Return (family, particle_group) for a particle surface, or None."""
    return _PARTICLE_LOOKUP.get(_norm_ar(surface))


# ---------------------------------------------------------------------------
# ConstructionInstance — the record stored in GrammarMemory
# ---------------------------------------------------------------------------

@dataclass
class ConstructionInstance:
    """One construction occurrence indexed in the grammar memory.

    Stored as JSONL. The ``embedding`` field is stored separately in a
    parallel FAISS index; in the JSONL we keep ``embedding_idx`` (the row
    index into the FAISS index) for round-trip lookup.
    """
    instance_id: str                    # e.g. "distill_v2_train_03421_2_4"
    sentence: str                        # full source sentence
    sentence_idx: int                    # row index in source corpus
    construction: str                    # one of ALL_FAMILIES
    particle_group: str                  # subgroup within family ("any" if none)
    span: Tuple[int, int]                # word-level (start, end+1) of construction
    particle_surface: str                # e.g. "أصبح" or "" for particle-less constructions
    head_morph: Dict[str, str] = field(default_factory=dict)
    head_deprel: str = ""
    head_governor_upos: str = ""
    sentence_length: int = 0
    items: List[Dict] = field(default_factory=list)   # full per-word labels for the span
    embedding_idx: int = -1              # row in the FAISS index
    confidence: float = 1.0              # quality flag (gold completeness 0..1)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["span"] = list(d["span"])
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ConstructionInstance":
        d = dict(d)
        d["span"] = tuple(d["span"])
        return cls(**d)


# ---------------------------------------------------------------------------
# Construction detection
# ---------------------------------------------------------------------------

def detect_constructions_in_record(rec: Dict) -> List[Dict]:
    """Return a list of construction span descriptors for a corpus record.

    Each descriptor:
        {
          "construction": str,
          "particle_group": str,
          "particle_surface": str,
          "span": (start, end_excl),
          "head_idx": int,                  # which word in span is the syntactic head
        }

    Detection rules (matching ``eval_per_construction.py``):
      - Particle-based families (kana, inna, istithna, mawsool, quranic_proxy):
        find the particle in the sentence, define span as
        [particle_idx, particle_idx + 3] (capped to sentence length) — the
        construction template is typically particle + ism + khabar.
      - iḍāfa: find tokens with role == 'mudaaf_ilayh'; span is
        [prev_token_idx, this_token_idx + 1].
      - idafa_multi: ≥2 consecutive 'mudaaf_ilayh' tokens; span covers them.
    """
    items = rec.get("items", [])
    if not items:
        return []
    out: List[Dict] = []

    # Particle-based detection
    for i, it in enumerate(items):
        word = it.get("word", "") or ""
        match = lookup_particle(word)
        if match is None:
            continue
        fam, grp = match
        end_excl = min(len(items), i + 3)   # cap at 3-word window
        out.append({
            "construction": fam,
            "particle_group": grp,
            "particle_surface": word,
            "span": (i, end_excl),
            "head_idx": i,
        })

    # Multi-word phrases (ma 3ada / ma khala) — need 2-token surface match
    for i in range(len(items) - 1):
        w1 = _norm_ar(items[i].get("word", "") or "")
        w2 = _norm_ar(items[i + 1].get("word", "") or "")
        joined = f"{w1} {w2}"
        for fam_name, groups in FAMILIES.items():
            for grp_name, particles in groups.items():
                for p in particles:
                    if " " in p and _norm_ar(p) == joined:
                        end_excl = min(len(items), i + 4)
                        out.append({
                            "construction": fam_name,
                            "particle_group": grp_name,
                            "particle_surface": p,
                            "span": (i, end_excl),
                            "head_idx": i,
                        })

    # iḍāfa and idafa_multi (role-based, no particle)
    consecutive: List[int] = []
    for i, it in enumerate(items):
        if it.get("role") == "mudaaf_ilayh":
            consecutive.append(i)
        else:
            if consecutive:
                if len(consecutive) >= 2:
                    span_start = max(0, consecutive[0] - 1)
                    out.append({
                        "construction": "idafa_multi",
                        "particle_group": "any",
                        "particle_surface": "",
                        "span": (span_start, consecutive[-1] + 1),
                        "head_idx": span_start,
                    })
                # Single iḍāfa pairs (one mudaaf + one mudaaf_ilayh)
                for idx in consecutive:
                    span_start = max(0, idx - 1)
                    out.append({
                        "construction": "idafa",
                        "particle_group": "any",
                        "particle_surface": "",
                        "span": (span_start, idx + 1),
                        "head_idx": span_start,
                    })
                consecutive = []
    # Tail flush
    if consecutive:
        if len(consecutive) >= 2:
            span_start = max(0, consecutive[0] - 1)
            out.append({
                "construction": "idafa_multi",
                "particle_group": "any",
                "particle_surface": "",
                "span": (span_start, consecutive[-1] + 1),
                "head_idx": span_start,
            })
        for idx in consecutive:
            span_start = max(0, idx - 1)
            out.append({
                "construction": "idafa",
                "particle_group": "any",
                "particle_surface": "",
                "span": (span_start, idx + 1),
                "head_idx": span_start,
            })

    return out


# ---------------------------------------------------------------------------
# Build a ConstructionInstance from a detected span
# ---------------------------------------------------------------------------

def build_signature(
    rec: Dict,
    span_desc: Dict,
    sentence_idx: int,
    embedding_idx: int = -1,
) -> ConstructionInstance:
    """Build a ConstructionInstance from a corpus record + a detected span."""
    items = rec.get("items", [])
    start, end = span_desc["span"]
    head_idx = span_desc["head_idx"]
    head_item = items[head_idx] if 0 <= head_idx < len(items) else {}

    head_morph = {
        "gender":   head_item.get("gender", "und"),
        "number":   head_item.get("number", "und"),
        "definite": head_item.get("definite", "und"),
    }
    head_deprel = head_item.get("deprel", "<unk>") or "<unk>"
    head_governor_upos = head_item.get("governor_upos", "") or ""

    instance_id = (
        f"{rec.get('source', 'unknown')}_{sentence_idx}_{start}_{end}_"
        f"{span_desc['construction']}"
    )
    return ConstructionInstance(
        instance_id=instance_id,
        sentence=rec.get("sentence", ""),
        sentence_idx=sentence_idx,
        construction=span_desc["construction"],
        particle_group=span_desc["particle_group"],
        span=(start, end),
        particle_surface=span_desc.get("particle_surface", ""),
        head_morph=head_morph,
        head_deprel=head_deprel,
        head_governor_upos=head_governor_upos,
        sentence_length=len(items),
        items=[items[i] for i in range(start, end) if i < len(items)],
        embedding_idx=embedding_idx,
        confidence=1.0 if rec.get("has_irab") else 0.5,
    )


# ---------------------------------------------------------------------------
# Symbolic similarity (used in retrieval scoring)
# ---------------------------------------------------------------------------

def symbolic_overlap(q: ConstructionInstance, c: ConstructionInstance) -> float:
    """Fraction of matching categorical fields between query and candidate.

    Returns a value in [0, 1]. Used as part of the hybrid retrieval score
    after the binary symbolic filter (family + particle_group must match).
    """
    matches = 0
    total = 0
    for k in ("gender", "number", "definite"):
        total += 1
        if q.head_morph.get(k, "und") == c.head_morph.get(k, "und"):
            matches += 1
    total += 1
    if q.head_deprel == c.head_deprel:
        matches += 1
    total += 1
    if q.head_governor_upos == c.head_governor_upos:
        matches += 1
    return matches / total if total else 0.0
