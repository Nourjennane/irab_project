"""Canonical UD dependency-feature schema for Phase 3.

Per-word dep features fed as input augmentation to the iʿrāb decoders:

  - DEPREL          : 37-way Universal Dependencies relation label
  - HEAD direction  : 3-way (root / left / right)
  - HEAD distance   : 5-way log bucket (0=root, 1, 2-3, 4-7, ≥8)
  - Governor UPOS   : 6-way canonical POS of the head word

These are computed offline by Stanza's Arabic UD parser (or read from
UD-PADT directly). The resulting features feed
:class:`irab_tashkeel.morphology.dep_aware_model.DepAwareStructuredModel`
through a small per-feature embedding table + concat-then-project to 768.

Why static input augmentation, not a learned conditioning module:
Phase 2 documented a joint-training-dynamics regression when iʿrāb-side
gradients flowed into morph heads through a conditioning module. Phase 3
sidesteps that by making the dep signal a *static input* — gradients can
flow through the embedding tables (small, ~30K params total) but the
*source* of the dep signal is offline parsing, not a head we train. No
moving target.

Kept frozen across Phase 3 retrains so the v3 (25-label iʿrāb) corpus
+ v4 (34-label) corpus can both be enriched with the same dep schema.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# DEPREL — UD canonical relation labels
# ---------------------------------------------------------------------------
# Source: https://universaldependencies.org/u/dep/index.html
# We keep the standard 37 labels + an explicit `<unk>` bucket for parser
# misses. We do NOT collapse compositional relations (e.g. ``nmod:poss``)
# — Stanza outputs the head-form ``nmod`` for those, which is what the
# UD-PADT corpus also uses (compositional sub-types are in DEPS, not DEPREL).
DEPREL_LABELS: List[str] = [
    "<unk>",     # 0 — parser miss / unknown
    "root",      # 1 — sentence root
    "nsubj",     # 2 — nominal subject
    "obj",       # 3 — direct object
    "iobj",      # 4 — indirect object
    "csubj",     # 5 — clausal subject
    "ccomp",     # 6 — clausal complement
    "xcomp",     # 7 — open clausal complement
    "obl",       # 8 — oblique nominal (most prepositional phrases)
    "vocative",  # 9 — vocative noun
    "expl",      # 10 — expletive
    "dislocated",# 11 — dislocated argument
    "advcl",     # 12 — adverbial clause modifier
    "advmod",    # 13 — adverbial modifier
    "discourse", # 14 — discourse marker
    "aux",       # 15 — auxiliary verb
    "cop",       # 16 — copula (often null in Arabic)
    "mark",      # 17 — subordinating particle
    "nmod",      # 18 — noun modifier (idafa head)
    "appos",     # 19 — appositional noun
    "nummod",    # 20 — numeric modifier
    "acl",       # 21 — adjectival clause
    "amod",      # 22 — adjectival modifier (naat)
    "det",       # 23 — determiner
    "clf",       # 24 — classifier (rare in Arabic)
    "case",      # 25 — preposition (harf jarr) / case marker
    "conj",      # 26 — conjoined element
    "cc",        # 27 — coordinating conjunction (harf atf)
    "fixed",     # 28 — fixed multi-word expression
    "flat",      # 29 — flat structure
    "compound",  # 30 — compound
    "list",      # 31 — list element
    "parataxis", # 32 — parataxis
    "orphan",    # 33 — orphan
    "goeswith",  # 34 — goes-with
    "reparandum",# 35 — reparandum (speech repair)
    "punct",     # 36 — punctuation
    "dep",       # 37 — generic / unspecified
]
DEPREL_TO_ID: Dict[str, int] = {d: i for i, d in enumerate(DEPREL_LABELS)}
ID_TO_DEPREL: Dict[int, str] = {i: d for d, i in DEPREL_TO_ID.items()}
N_DEPREL = len(DEPREL_LABELS)


# ---------------------------------------------------------------------------
# HEAD direction (3-way)
# ---------------------------------------------------------------------------
# 0 = root (no governor)
# 1 = left (governor index < self index)
# 2 = right (governor index > self index)
HEAD_DIR_LABELS: List[str] = ["root", "left", "right"]
HEAD_DIR_TO_ID: Dict[str, int] = {d: i for i, d in enumerate(HEAD_DIR_LABELS)}
N_HEAD_DIR = len(HEAD_DIR_LABELS)


# ---------------------------------------------------------------------------
# HEAD distance (5-way log bucket)
# ---------------------------------------------------------------------------
# 0 = root  (no governor)
# 1 = adjacent (|HEAD - self| == 1)
# 2 = near    (2 ≤ |HEAD - self| ≤ 3)
# 3 = mid     (4 ≤ |HEAD - self| ≤ 7)
# 4 = far     (|HEAD - self| ≥ 8)
HEAD_DIST_LABELS: List[str] = ["root", "adj", "near", "mid", "far"]
HEAD_DIST_TO_ID: Dict[str, int] = {d: i for i, d in enumerate(HEAD_DIST_LABELS)}
N_HEAD_DIST = len(HEAD_DIST_LABELS)


def head_distance_bucket(distance: int) -> int:
    """Map an absolute head-self index distance to a bucket id.

    Args:
        distance: ``HEAD - self`` (absolute value). 0 means root / unbound.
    """
    d = abs(int(distance))
    if d == 0:
        return HEAD_DIST_TO_ID["root"]
    if d == 1:
        return HEAD_DIST_TO_ID["adj"]
    if d <= 3:
        return HEAD_DIST_TO_ID["near"]
    if d <= 7:
        return HEAD_DIST_TO_ID["mid"]
    return HEAD_DIST_TO_ID["far"]


# ---------------------------------------------------------------------------
# Governor UPOS (6-way canonical POS, matches existing rev-2 POS schema)
# ---------------------------------------------------------------------------
# Reuse the 6-class POS schema already used by the iʿrāb POS head.
# Imported lazily to avoid circular imports between phase modules.
def _canonical_pos_id_for(upos: str) -> int:
    """Map UD UPOS → canonical 6-class POS id used by the rev-2 POS head.

    Falls back to ``punctuation`` for empty/unknown UPOS (e.g. when the
    word is the sentence root and there is no governor) — using
    ``punctuation`` here is a deliberate "non-content" sentinel since the
    canonical 6-class schema does not have an explicit ``<unk>`` slot.
    """
    from .schema import UPOS_TO_CANONICAL_POS
    from ..structured.schema import POS_TO_ID
    if not upos:
        return POS_TO_ID["punctuation"]
    canon = UPOS_TO_CANONICAL_POS.get(upos, "punctuation")
    return POS_TO_ID[canon]


# ---------------------------------------------------------------------------
# Per-word feature dim summary
# ---------------------------------------------------------------------------
# Default embedding dims (sized so the total dep embedding is small
# relative to the 768-dim encoder feature):
DEPREL_EMB_DIM = 32
HEAD_DIR_EMB_DIM = 16
HEAD_DIST_EMB_DIM = 16
GOV_POS_EMB_DIM = 16

DEP_FEATURE_DIM_TOTAL = (
    DEPREL_EMB_DIM + HEAD_DIR_EMB_DIM + HEAD_DIST_EMB_DIM + GOV_POS_EMB_DIM
)


def deprel_to_id(deprel: str) -> int:
    """DEPREL string → id, with case-stripping and ``<unk>`` fallback.

    Stanza emits compositional labels like ``nmod:poss``; we keep the head
    form (``nmod``). UD-PADT uses the same convention.
    """
    if not deprel:
        return DEPREL_TO_ID["<unk>"]
    head = deprel.split(":", 1)[0]
    return DEPREL_TO_ID.get(head, DEPREL_TO_ID["<unk>"])


def build_dep_features(
    *,
    deprels: List[str],
    head_indices: List[int],
    governor_uposes: List[str],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Build the four per-word dep feature id lists for a single sentence.

    Args:
        deprels: per-word DEPREL strings (length W).
        head_indices: per-word HEAD index, 1-based UD convention. 0 = root.
        governor_uposes: per-word UPOS of the governor token (or "" if root).

    Returns:
        (deprel_ids, head_dir_ids, head_dist_ids, gov_upos_ids), each
        length W of int32-compatible integers.
    """
    n = len(deprels)
    if not (len(head_indices) == n and len(governor_uposes) == n):
        raise ValueError(
            f"per-word dep feature lists must match length: "
            f"{len(deprels)=} {len(head_indices)=} {len(governor_uposes)=}"
        )

    deprel_ids = [deprel_to_id(d) for d in deprels]
    head_dir_ids: List[int] = []
    head_dist_ids: List[int] = []
    gov_pos_ids: List[int] = []

    for self_idx, head_idx in enumerate(head_indices):
        # UD HEAD is 1-based; self_idx here is 0-based → self UD = self_idx+1.
        # head_idx == 0 means root (no governor).
        if head_idx == 0:
            head_dir_ids.append(HEAD_DIR_TO_ID["root"])
            head_dist_ids.append(HEAD_DIST_TO_ID["root"])
        else:
            ud_self = self_idx + 1
            if head_idx < ud_self:
                head_dir_ids.append(HEAD_DIR_TO_ID["left"])
            else:
                head_dir_ids.append(HEAD_DIR_TO_ID["right"])
            head_dist_ids.append(head_distance_bucket(head_idx - ud_self))
        gov_pos_ids.append(_canonical_pos_id_for(governor_uposes[self_idx]))

    return deprel_ids, head_dir_ids, head_dist_ids, gov_pos_ids
