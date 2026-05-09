"""Stage configuration for the 7-stage curriculum.

Each stage has explicit policy: which sources, which difficulty
range, which sentence filters, which graph-edge filters, what
proportion of earlier-stage sentences to *rehearse*, and what
validation gate must pass before advancing.

These configs are the contract between the corpus (data_v2) and
the training pipeline. Modifying them changes the curriculum
schedule directly without touching trainer code.

Strict-leakage policy
---------------------

The constants ``TEST_SOURCES`` and ``DEV_SOURCES`` define names
that may **never** appear in ``allowed_sources`` /
``preferred_sources`` of any training stage. Code that builds
training pools must assert against these.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Held-out source policy — sentences from these sources are FORBIDDEN
# in any training pool, rehearsal pool, hard-negative pool, or graph
# construction. They appear ONLY in the final eval set.
# ---------------------------------------------------------------------------

TEST_SOURCES = frozenset({"gazelle_test", "masaq_quranic", "ud_padt_test"})
DEV_SOURCES  = frozenset({"ud_padt_dev"})


def assert_no_test_sources(sources, where: str = ""):
    """Hard assertion that no source in ``sources`` is a held-out test source."""
    bad = [s for s in sources if s in TEST_SOURCES]
    if bad:
        raise AssertionError(
            f"FORBIDDEN test source(s) {bad} appearing in {where!r}. "
            f"TEST_SOURCES={sorted(TEST_SOURCES)} must NEVER enter the "
            f"training/rehearsal/hard-negative pools. "
            f"Re-check curriculum config and dataset-loading code."
        )


@dataclass
class StageConfig:
    """One curriculum stage's policy.

    Sample-selection filters
    ------------------------
    - ``difficulty_range``       — (min, max) inclusive on
                                    ``CurriculumMetadata.difficulty_level``
    - ``preferred_sources``      — sources to draw most heavily from
    - ``allowed_sources``        — full set of sources permitted
    - ``min_completeness``       — drop sentences with fewer fields
    - ``max_semantic_pressure``  — None means no ceiling
    - ``min_semantic_pressure``  — None means no floor
    - ``max_ambiguity``          — None means no ceiling
    - ``drop_ambiguous_constructions`` — for stages where overlap is noise

    Graph filters (used when emitting sparse graphs to a GNN)
    --------------------------------------------------------
    - ``keep_node_types``       — None means all NODE_TYPES
    - ``keep_edge_types``       — None means all EDGE_TYPES

    Rehearsal
    ---------
    - ``rehearsal_ratio``       — what fraction of each batch should
                                    come from earlier stages (0.0..1.0)

    Gate
    ----
    - ``gate_metric``           — which scalar metric on the eval
                                    set decides advancement
    - ``gate_threshold``        — minimum value of that metric
    """
    stage_id:                 int
    name:                     str
    description:              str

    # Sample selection
    difficulty_range:         tuple = (1, 7)
    preferred_sources:        List[str] = field(default_factory=list)
    allowed_sources:          List[str] = field(default_factory=list)
    min_completeness:         float = 0.0
    max_semantic_pressure:    Optional[int] = None
    min_semantic_pressure:    Optional[int] = None
    max_ambiguity:            Optional[float] = None
    drop_ambiguous_constructions: bool = False

    # Graph filters
    keep_node_types:          Optional[List[str]] = None
    keep_edge_types:          Optional[List[str]] = None

    # Rehearsal of earlier stages
    rehearsal_ratio:          float = 0.0

    # Gate
    gate_metric:              str = "fully"
    gate_threshold:           float = 0.0
    gate_eval_set:            str = "dev"           # which eval split to gate on

    # Training budget
    target_steps:             int = 1000
    max_steps:                int = 5000


# ---------------------------------------------------------------------------
# Default 7-stage schedule
# ---------------------------------------------------------------------------

DEFAULT_STAGES: Dict[int, StageConfig] = {
    1: StageConfig(
        stage_id=1, name="morphology_foundation",
        description="Pure morphology + POS supervision; no constructions, "
                    "shallow dep, single-clause sentences.",
        difficulty_range=(1, 1),
        preferred_sources=["ud_padt_train", "distill_v2"],
        allowed_sources=["ud_padt_train", "distill_v2", "ud_padt_dev"],
        keep_node_types=["token"],
        keep_edge_types=["dep"],
        rehearsal_ratio=0.0,
        gate_metric="morph_macro_f1", gate_threshold=0.90,
        target_steps=2000, max_steps=10000,
    ),
    2: StageConfig(
        stage_id=2, name="local_syntax",
        description="Local syntax — short sentences, dep depth ≤ 3, "
                    "no nested clauses, no construction overlap.",
        difficulty_range=(1, 2),
        preferred_sources=["ud_padt_train", "distill_v2"],
        allowed_sources=["ud_padt_train", "distill_v2", "ud_padt_dev"],
        max_semantic_pressure=1,
        keep_node_types=["token", "phrase"],
        keep_edge_types=["dep", "construction_member"],
        rehearsal_ratio=0.15,
        gate_metric="role_f1", gate_threshold=0.40,
        target_steps=3000, max_steps=15000,
    ),
    3: StageConfig(
        stage_id=3, name="simple_constructions",
        description="Single-construction sentences (single iḍāfa, "
                    "single kāna sister, single mawṣūl). Drop ambiguous.",
        difficulty_range=(1, 3),
        preferred_sources=["distill_v2", "ud_padt_train"],
        allowed_sources=["distill_v2", "ud_padt_train", "ud_padt_dev"],
        max_semantic_pressure=1,
        drop_ambiguous_constructions=True,
        keep_edge_types=["dep", "construction_member", "agreement"],
        rehearsal_ratio=0.20,
        gate_metric="construction_f1_macro", gate_threshold=0.60,
        target_steps=3000, max_steps=15000,
    ),
    4: StageConfig(
        stage_id=4, name="nested_syntax",
        description="Iḍāfa chains, embedded clauses, multi-construction "
                    "sentences with overlap permitted.",
        difficulty_range=(1, 4),
        preferred_sources=["ud_padt_train", "distill_v2"],
        allowed_sources=["ud_padt_train", "ud_padt_dev", "distill_v2"],
        max_semantic_pressure=2,
        keep_edge_types=["dep", "construction_member", "agreement",
                          "clause_member"],
        rehearsal_ratio=0.20,
        gate_metric="fully", gate_threshold=0.25,
        target_steps=4000, max_steps=20000,
    ),
    5: StageConfig(
        stage_id=5, name="semantic_interactions",
        description="Sentences requiring semantic disambiguation: hāl vs "
                    "naʿt, istithnāʾ munqaṭiʿ vs muttaṣil, ambiguous "
                    "attachment.",
        difficulty_range=(1, 5),
        preferred_sources=["distill_v2", "ud_padt_train"],
        allowed_sources=["distill_v2", "ud_padt_train", "ud_padt_dev"],
        min_semantic_pressure=2,
        keep_edge_types=["dep", "construction_member", "agreement",
                          "clause_member", "semantic_link"],
        rehearsal_ratio=0.25,
        gate_metric="fully", gate_threshold=0.30,
        target_steps=5000, max_steps=25000,
    ),
    6: StageConfig(
        stage_id=6, name="discourse_sensitive",
        description="Sentences requiring cross-sentence context — pronoun "
                    "antecedents, topic continuation, rhetorical relations.",
        difficulty_range=(1, 6),
        preferred_sources=["distill_v2", "ud_padt_train"],
        allowed_sources=["distill_v2", "ud_padt_train", "ud_padt_dev"],
        min_semantic_pressure=2,
        keep_edge_types=["dep", "construction_member", "agreement",
                          "clause_member", "semantic_link", "discourse_link",
                          "coref"],
        rehearsal_ratio=0.30,
        gate_metric="fully", gate_threshold=0.30,
        target_steps=4000, max_steps=20000,
    ),
    7: StageConfig(
        stage_id=7, name="quranic_classical",
        description="Hardest classical / nested / overlapping cases drawn "
                    "from distill_v2 + UD train; the *evaluation* set "
                    "carries the held-out Quranic split.",
        difficulty_range=(1, 7),
        preferred_sources=["distill_v2", "ud_padt_train"],
        allowed_sources=["distill_v2", "ud_padt_train", "ud_padt_dev"],
        keep_edge_types=None,                      # all edge types
        rehearsal_ratio=0.35,
        gate_metric="fully", gate_threshold=0.30,
        target_steps=4000, max_steps=20000,
    ),
}


# Sanity check at module load: no stage may carry a TEST_SOURCES name.
for _sid, _cfg in DEFAULT_STAGES.items():
    assert_no_test_sources(_cfg.allowed_sources,
                           where=f"DEFAULT_STAGES[{_sid}].allowed_sources")
    assert_no_test_sources(_cfg.preferred_sources,
                           where=f"DEFAULT_STAGES[{_sid}].preferred_sources")


def get_stage(stage_id: int) -> StageConfig:
    """Look up a stage config by id (1..7)."""
    if stage_id not in DEFAULT_STAGES:
        raise KeyError(f"unknown stage_id {stage_id}; valid 1..7")
    return DEFAULT_STAGES[stage_id]


def all_stages() -> List[StageConfig]:
    """Return stages in order (1..7)."""
    return [DEFAULT_STAGES[i] for i in sorted(DEFAULT_STAGES.keys())]
