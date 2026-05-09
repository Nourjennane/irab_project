"""Hard-failure-weighted batch sampler.

Wraps the existing ``StratifiedSampler`` to upweight sentences carrying
high-priority T-codes (long-range, nested clause, semantic ambiguity,
construction overlap, …) per ``failure_taxonomy.DEFAULT_TAG_WEIGHTS``.

The wrapper still respects:

  - stage eligibility (filtered by ``stage_eligibility``)
  - source policy (TEST_SOURCES forbidden — assertion)
  - rehearsal ratio
  - source-stratified draws

It changes ONLY the within-pool draw to be weighted by hard-failure
score instead of uniform random.

Usage::

    sampler = HardFailureSampler(current_pool, earlier_pools, cfg,
                                   batch_size=16, seed=0,
                                   tag_weights=stage_tag_weights(cfg.stage_id))
    for batch in sampler:
        ...

This is a drop-in replacement for ``StratifiedSampler``.
"""
from __future__ import annotations

import random
from typing import Dict, List

from ..curriculum.config import StageConfig, assert_no_test_sources
from ..curriculum.sampler import StagePool, StratifiedSampler
from ..data_v2.schema_v2 import Sentence
from .failure_taxonomy import (
    DEFAULT_TAG_WEIGHTS, sentence_weight, stage_tag_weights,
)


class HardFailureSampler(StratifiedSampler):
    """Stratified + per-sentence hard-failure-weighted draws."""

    def __init__(
        self,
        current_pool: StagePool,
        earlier_pools: List[StagePool],
        config: StageConfig,
        batch_size: int = 32,
        seed: int = 0,
        tag_weights: Dict[str, float] = None,
    ):
        super().__init__(current_pool, earlier_pools, config, batch_size, seed)
        self.tag_weights = tag_weights or stage_tag_weights(config.stage_id)
        # Defence in depth: no TEST_SOURCES sentence may be reachable
        assert_no_test_sources(
            list(self.current.by_source.keys())
            + [src for p in self.earlier for src in p.by_source.keys()],
            where=f"HardFailureSampler(stage={config.stage_id})",
        )
        # Pre-compute weights per pool
        self._cur_weights = self._weight_list(self.current.sentences)
        self._earlier_weights = [self._weight_list(p.sentences)
                                  for p in self.earlier]

    def _weight_list(self, sentences: List[Sentence]) -> List[float]:
        return [sentence_weight(s, self.tag_weights) for s in sentences]

    def _weighted_choice(self, sentences: List[Sentence],
                         weights: List[float], k: int) -> List[Sentence]:
        if not sentences or k == 0:
            return []
        # random.choices with weights handles the weighted draw
        # without requiring numpy.
        return self.rng.choices(sentences, weights=weights, k=k)

    # Override _draw_stratified to use weighted picks within each source
    def _draw_stratified(self, pool: StagePool, k: int,
                          preferred: List[str]) -> List[Sentence]:
        if not pool.sentences or k == 0:
            return []
        if not preferred:
            preferred = list(pool.by_source.keys())

        avail = [(src, len(pool.by_source.get(src, [])))
                 for src in preferred if src in pool.by_source]
        if not avail:
            # Fallback to weighted pick from full pool
            return self._weighted_choice(
                pool.sentences,
                [sentence_weight(s, self.tag_weights) for s in pool.sentences],
                k,
            )

        total_avail = sum(n for _, n in avail) or 1
        out: List[Sentence] = []
        per_source = {src: max(1, k * n // total_avail) for src, n in avail}
        for src in preferred:
            if src not in pool.by_source:
                continue
            sents = pool.by_source[src]
            ws = [sentence_weight(s, self.tag_weights) for s in sents]
            n_take = min(per_source.get(src, 0), len(sents))
            out.extend(self._weighted_choice(sents, ws, n_take))

        # Top up if rounding undershot
        while len(out) < k and len(out) < pool.n:
            extra = self._weighted_choice(
                pool.sentences,
                [sentence_weight(s, self.tag_weights) for s in pool.sentences],
                1,
            )[0]
            out.append(extra)
        return out[:k]

    def _draw_rehearsal(self, k: int) -> List[Sentence]:
        if not self.earlier or k == 0:
            return []
        sizes = [p.n for p in self.earlier]
        total = sum(sizes) or 1
        out: List[Sentence] = []
        for pool, size in zip(self.earlier, sizes):
            if size == 0:
                continue
            n_take = max(1, k * size // total)
            n_take = min(n_take, size)
            ws = [sentence_weight(s, self.tag_weights) for s in pool.sentences]
            out.extend(self._weighted_choice(pool.sentences, ws, n_take))
        if len(out) > k:
            out = out[:k]
        return out
