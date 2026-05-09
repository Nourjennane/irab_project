"""Stratified mini-batch sampler with rehearsal.

The sampler draws sentences for a single curriculum stage. Within
each batch:

  - ``(1 - rehearsal_ratio)`` fraction comes from the *current*
    stage's eligible pool (filtered by stage policy).
  - ``rehearsal_ratio`` fraction comes from earlier stages, sampled
    proportionally to each earlier stage's size.

Within the current-stage portion we additionally domain-stratify:
each domain seen in ``stage.preferred_sources`` is allocated
proportional batch capacity, ensuring no single source dominates.

The sampler is deterministic given a seed — same seed + same data
+ same stage config = same batch sequence. Useful for reproducible
training runs and for debugging stage transitions.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from ..data_v2.schema_v2 import Sentence
from .config import StageConfig


# ===========================================================================
# Filter helpers
# ===========================================================================

def stage_eligibility(s: Sentence, cfg: StageConfig) -> bool:
    """Return True iff ``s`` is eligible for ``cfg`` per all filters."""
    d = s.curriculum.difficulty_level
    if d < cfg.difficulty_range[0] or d > cfg.difficulty_range[1]:
        return False

    if cfg.allowed_sources and s.metadata.source not in cfg.allowed_sources:
        return False

    if s.completeness.fields_complete_pct < cfg.min_completeness:
        return False

    sp = s.curriculum.semantic_pressure_score
    if cfg.max_semantic_pressure is not None and sp > cfg.max_semantic_pressure:
        return False
    if cfg.min_semantic_pressure is not None and sp < cfg.min_semantic_pressure:
        return False

    if (cfg.max_ambiguity is not None
        and s.curriculum.ambiguity_score > cfg.max_ambiguity):
        return False

    if cfg.drop_ambiguous_constructions:
        for c in s.constructions:
            if c.ambiguity_score >= 0.3:
                return False

    return True


# ===========================================================================
# Stratified sampler
# ===========================================================================

@dataclass
class StagePool:
    """Filtered sentence pool for one stage, with per-source buckets."""
    stage_id: int
    sentences: List[Sentence] = field(default_factory=list)
    by_source: Dict[str, List[Sentence]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.sentences)


def build_stage_pool(sentences: List[Sentence], cfg: StageConfig) -> StagePool:
    """Filter sentences to those eligible for the stage; bucket by source."""
    eligible: List[Sentence] = []
    by_source: Dict[str, List[Sentence]] = {}
    for s in sentences:
        if not stage_eligibility(s, cfg):
            continue
        eligible.append(s)
        by_source.setdefault(s.metadata.source, []).append(s)
    return StagePool(stage_id=cfg.stage_id, sentences=eligible,
                     by_source=by_source)


# ===========================================================================
# Sampler
# ===========================================================================

class StratifiedSampler:
    """Yields batches of sentences for one curriculum stage.

    Args
    ----
    current_pool : :class:`StagePool` for the current stage
    earlier_pools : list of :class:`StagePool` for stages < current_stage
    config : :class:`StageConfig`
    batch_size : sentences per batch
    seed : RNG seed
    """

    def __init__(
        self,
        current_pool: StagePool,
        earlier_pools: List[StagePool],
        config: StageConfig,
        batch_size: int = 32,
        seed: int = 0,
    ):
        self.current = current_pool
        self.earlier = earlier_pools
        self.cfg = config
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self._step = 0

    def n_current_per_batch(self) -> int:
        return max(1, int(self.batch_size * (1.0 - self.cfg.rehearsal_ratio)))

    def n_rehearsal_per_batch(self) -> int:
        return self.batch_size - self.n_current_per_batch()

    # -------------------------------------------------------------------
    # Domain-stratified draw from a pool's by_source buckets
    # -------------------------------------------------------------------

    def _draw_stratified(self, pool: StagePool, k: int,
                          preferred: List[str]) -> List[Sentence]:
        """Draw k sentences from pool, preferring sources in ``preferred``."""
        if not pool.sentences or k == 0:
            return []
        if not preferred:
            preferred = list(pool.by_source.keys())

        # Allocate per-preferred-source budget proportional to availability
        avail = [(src, len(pool.by_source.get(src, [])))
                 for src in preferred if src in pool.by_source]
        if not avail:
            return self.rng.sample(pool.sentences, min(k, len(pool.sentences)))

        total_avail = sum(n for _, n in avail) or 1
        # Round-robin within budget
        out: List[Sentence] = []
        per_source = {src: max(1, k * n // total_avail) for src, n in avail}
        for src in preferred:
            if src not in pool.by_source:
                continue
            n_take = min(per_source.get(src, 0), len(pool.by_source[src]))
            out.extend(self.rng.sample(pool.by_source[src], n_take))

        # Top up if rounding undershot
        while len(out) < k and len(out) < pool.n:
            extra = self.rng.choice(pool.sentences)
            if extra not in out:
                out.append(extra)
        return out[:k]

    def _draw_rehearsal(self, k: int) -> List[Sentence]:
        """Draw rehearsal samples from earlier stages, proportional to size."""
        if not self.earlier or k == 0:
            return []
        sizes = [p.n for p in self.earlier]
        total = sum(sizes) or 1
        out: List[Sentence] = []
        for pool, size in zip(self.earlier, sizes):
            if size == 0: continue
            n_take = max(1, k * size // total)
            n_take = min(n_take, size)
            if n_take == 0: continue
            out.extend(self.rng.sample(pool.sentences, n_take))
        # Top up if rounding undershot; trim if overshot
        if len(out) > k:
            out = out[:k]
        elif len(out) < k:
            all_earlier = [s for p in self.earlier for s in p.sentences]
            while len(out) < k and len(out) < len(all_earlier):
                cand = self.rng.choice(all_earlier)
                if cand not in out:
                    out.append(cand)
        return out

    # -------------------------------------------------------------------
    # Public iteration interface
    # -------------------------------------------------------------------

    def sample_batch(self) -> List[Sentence]:
        """Return one batch of size ``batch_size``."""
        cur = self._draw_stratified(
            self.current, self.n_current_per_batch(),
            self.cfg.preferred_sources,
        )
        reh = self._draw_rehearsal(self.n_rehearsal_per_batch())
        batch = cur + reh
        self._step += 1
        return batch[:self.batch_size]

    def __iter__(self) -> Iterator[List[Sentence]]:
        return self

    def __next__(self) -> List[Sentence]:
        return self.sample_batch()
