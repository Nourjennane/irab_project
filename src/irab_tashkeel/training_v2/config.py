"""Training configuration for the next-gen curriculum trainer.

The trainer is intentionally separate from the curriculum scheduler:
the scheduler decides *which* sentences to train on per stage; the
trainer decides *how* to train (optimiser, head weighting, batch
size, etc).

A single :class:`TrainerConfig` carries the full training setup
across all 7 stages. Stage-specific overrides are derived from
:class:`StageConfig` (curriculum.config) — e.g., the trainer
emphasises the morph head at stage 1, the role head at stage 5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HeadLossWeights:
    """Per-head loss weights. Stage-specific overrides land here."""
    case:    float = 1.0
    role:    float = 1.0
    marker:  float = 1.0
    pos:     float = 0.5
    morph:   float = 0.5      # combined weight for the 7 morph heads
    dep:     float = 0.0      # 0 unless a dep prediction head is added later

    def as_dict(self) -> Dict[str, float]:
        return {"case": self.case, "role": self.role, "marker": self.marker,
                "pos": self.pos, "morph": self.morph, "dep": self.dep}


@dataclass
class TrainerConfig:
    """Top-level training configuration."""

    # Backbone
    encoder_name:           str = "UBC-NLP/AraT5v2-base-1024"
    warm_start_checkpoint:  Optional[str] = "runs/phase3a_491240/final"
    """Path to a checkpoint to warm-start from. Default Phase 3-A."""

    # Optimisation
    learning_rate:          float = 5e-5
    weight_decay:           float = 0.01
    warmup_steps:           int   = 500
    max_grad_norm:          float = 1.0
    batch_size:             int   = 32
    gradient_accumulation:  int   = 1
    seed:                   int   = 0
    fp16:                   bool  = False
    bf16:                   bool  = True

    # Heads
    head_loss_weights:      HeadLossWeights = field(default_factory=HeadLossWeights)
    """Default head weights; per-stage overrides set in stage_overrides."""

    # Stage-specific head weight overrides (stage_id → HeadLossWeights)
    stage_overrides:        Dict[int, HeadLossWeights] = field(default_factory=lambda: {
        # Stage 1: emphasise morph; case/role/marker still get gradient
        1: HeadLossWeights(case=0.5, role=0.5, marker=0.5, pos=0.5, morph=2.0),
        # Stage 2: ramp up syntax
        2: HeadLossWeights(case=1.0, role=1.0, marker=1.0, pos=0.5, morph=1.5),
        # Stage 3: emphasise role + construction-relevant heads
        3: HeadLossWeights(case=1.0, role=1.5, marker=1.0, pos=0.5, morph=1.0),
        # Stage 4: full multi-head
        4: HeadLossWeights(case=1.0, role=1.0, marker=1.0, pos=0.5, morph=1.0),
        # Stage 5: full + slight role bias
        5: HeadLossWeights(case=1.0, role=1.2, marker=1.0, pos=0.5, morph=1.0),
        # Stage 6: discourse-aware (no separate weight, but sentence
        # selection in the sampler handles it)
        6: HeadLossWeights(case=1.0, role=1.2, marker=1.0, pos=0.5, morph=1.0),
        # Stage 7: full (no further bias)
        7: HeadLossWeights(case=1.0, role=1.0, marker=1.0, pos=0.5, morph=1.0),
    })

    # Eval / gate
    eval_every_steps:       int   = 200
    eval_batch_size:        int   = 64
    gate_check_every:       int   = 1     # every eval

    # Output
    output_root:            str   = "runs/nextgen"
    save_per_stage:         bool  = True

    # Reproducibility
    deterministic:          bool  = True

    # Device
    device:                 str   = "auto"

    def head_weights_for_stage(self, stage_id: int) -> HeadLossWeights:
        """Return the head weights for a stage, falling back to default."""
        return self.stage_overrides.get(stage_id, self.head_loss_weights)
