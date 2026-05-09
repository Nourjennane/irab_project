"""StageTrainer — main training loop driven by CurriculumScheduler.

Glues:

  CurriculumScheduler (curriculum) →
  SchemaV2Dataset + SchemaV2Collator (training_v2) →
  DepAwareStructuredModel (frozen-baseline model) →
  multi-head loss (training_v2.loss) →
  eval_v2 metrics →
  gate decision

into a single object the CLI invokes.

Design constraints
------------------

- The trainer never inspects the model's internal architecture; it
  only knows the (input_ids, attention_mask, word_*) input
  signature and the multi-head ``logits`` dict output. This
  keeps it backbone-agnostic for the future Step 6 backbone
  benchmark.

- All stage transitions are checkpoint-driven: when
  ``CurriculumScheduler.advance_or_continue`` returns ADVANCE, the
  trainer saves the per-stage checkpoint and rebuilds the sampler.

- Resumable: ``save_checkpoint(path)`` persists the model state +
  optimizer state + scheduler state; ``resume(path)`` loads them
  and continues from the saved step.

This file is *thin* — the heavy lifting happens in the model
class and the eval_v2 module. The trainer is mostly orchestration.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..curriculum import (
    CurriculumScheduler, GateDecision, GateResult, ScheduleState,
)
from ..data_v2.schema_v2 import Sentence
from .config import HeadLossWeights, TrainerConfig
from .dataset import SchemaV2Dataset


@dataclass
class TrainerState:
    """Persisted training state — checkpointed alongside model weights."""
    global_step:       int = 0
    stage_id:          int = 1
    steps_in_stage:    int = 0
    epoch_in_stage:    int = 0
    history:           List[Dict] = field(default_factory=list)


class StageTrainer:
    """Curriculum-driven multi-task trainer.

    The trainer's run loop:

    .. code-block:: python

        trainer.train()
            ┝→ for each stage in scheduler:
            │      sampler = scheduler.make_sampler(...)
            │      for step in range(stage.target_steps):
            │          batch = next(sampler)
            │          loss = train_step(batch)
            │          if step % eval_every == 0:
            │              metrics = eval_step()
            │              gate = scheduler.advance_or_continue(metrics)
            │              if gate.decision in {ADVANCE, TIMEOUT_ADVANCE}:
            │                  save_checkpoint(stage_dir)
            │                  break  # rebuild sampler with new stage
            │      else:
            │          save_checkpoint(timeout_dir)
    """

    def __init__(
        self,
        config: TrainerConfig,
        scheduler: CurriculumScheduler,
        model,                          # any torch.nn.Module with forward(input_ids, ...) → logits dict
        tokenizer,                      # HF tokenizer
        eval_sentences: List[Sentence],
        eval_predictor=None,            # callable: (sentence) → SentencePrediction
    ):
        self.config = config
        self.scheduler = scheduler
        self.model = model
        self.tokenizer = tokenizer
        self.eval_sentences = eval_sentences
        self.eval_predictor = eval_predictor
        self.state = TrainerState(stage_id=scheduler.current_stage_id)

    # -- training loop -------------------------------------------------------

    def _train_step(self, batch) -> Dict[str, float]:
        """Single optimisation step. Returns scalar metrics for logging.

        Subclasses or trainer wrappers may override this; the default
        does the standard:
          forward → multi-head loss → backward → optimizer.step
        """
        import torch
        from .loss import compute_multi_head_loss

        device = next(self.model.parameters()).device
        ws = self.config.head_weights_for_stage(self.state.stage_id)

        batch_t = {k: (v.to(device) if hasattr(v, "to") else v)
                    for k, v in batch.items()}

        # Forward — adapt to the existing DepAwareStructuredModel API
        out = self.model(
            input_ids=batch_t["input_ids"],
            attention_mask=batch_t["attention_mask"],
            word_starts=batch_t["word_starts"],
            word_ends=batch_t["word_ends"],
            word_mask=batch_t["word_mask"],
            return_dict=True,
        )

        # Build logits dict in the shape compute_multi_head_loss expects
        logits = {
            "case":   out["case_logits"],
            "role":   out["role_logits"],
            "marker": out["marker_logits"],
            "pos":    out["pos_logits"],
        }
        # morph heads (if model exposes them under morph_logits)
        for axis in ("gender", "number", "definite", "person",
                     "aspect", "mood", "voice"):
            key = f"{axis}_logits"
            if key in out:
                logits[f"morph_{axis}"] = out[key]

        labels = {
            "case":   batch_t["case_labels"],
            "role":   batch_t["role_labels"],
            "marker": batch_t["marker_labels"],
            "pos":    batch_t["pos_labels"],
        }
        for axis in ("gender", "number", "definite", "person",
                     "aspect", "mood", "voice"):
            key = f"morph_{axis}_labels"
            if key in batch_t:
                labels[f"morph_{axis}"] = batch_t[key]

        result = compute_multi_head_loss(logits, labels, ws)
        loss = result["loss"]

        return {
            "loss": float(loss.item()),
            **{f"loss_{k}": float(v.item()) for k, v in result["per_head"].items()},
            "logits": logits, "labels": labels, "loss_tensor": loss,
        }

    def train(self) -> None:
        """Run the full multi-stage curriculum.

        Concrete training requires torch + an optimiser, which lands
        in the CLI entry point (``scripts/training_v2/train_curriculum.py``)
        rather than here — keeping this module testable without a GPU.
        Subclasses override :func:`_train_step` to add the
        backward + optimiser update.
        """
        raise NotImplementedError(
            "StageTrainer.train() is invoked from the CLI entry point "
            "which wires in the optimiser; this base class provides the "
            "stage-aware sampling + gating + checkpointing scaffolding."
        )

    # -- checkpoint ----------------------------------------------------------

    def save_checkpoint(self, path: str | Path,
                         optimizer_state=None,
                         scheduler_state_dict=None) -> None:
        """Save trainer state + scheduler state + (optional) optimiser state."""
        try:
            import torch
        except ImportError:
            raise ImportError("save_checkpoint requires torch")
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "pytorch_model.bin")
        if optimizer_state is not None:
            torch.save(optimizer_state, out_dir / "optimizer.bin")
        sched_state = self.scheduler.to_dict()
        (out_dir / "scheduler_state.json").write_text(
            json.dumps(sched_state, indent=2, ensure_ascii=False)
        )
        (out_dir / "trainer_state.json").write_text(
            json.dumps({
                "global_step": self.state.global_step,
                "stage_id": self.state.stage_id,
                "steps_in_stage": self.state.steps_in_stage,
                "epoch_in_stage": self.state.epoch_in_stage,
                "history": self.state.history,
            }, indent=2, ensure_ascii=False)
        )

    def resume(self, path: str | Path) -> None:
        """Restore trainer + scheduler state from a checkpoint dir."""
        try:
            import torch
        except ImportError:
            raise ImportError("resume requires torch")
        in_dir = Path(path)
        sd_path = in_dir / "pytorch_model.bin"
        if sd_path.exists():
            self.model.load_state_dict(
                torch.load(sd_path, map_location="cpu", weights_only=True)
            )
        ts_path = in_dir / "trainer_state.json"
        if ts_path.exists():
            d = json.loads(ts_path.read_text())
            self.state = TrainerState(
                global_step=int(d.get("global_step", 0)),
                stage_id=int(d.get("stage_id", 1)),
                steps_in_stage=int(d.get("steps_in_stage", 0)),
                epoch_in_stage=int(d.get("epoch_in_stage", 0)),
                history=list(d.get("history", [])),
            )
        sc_path = in_dir / "scheduler_state.json"
        if sc_path.exists():
            self.scheduler.load_state_dict(json.loads(sc_path.read_text()))
