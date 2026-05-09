"""CLI entry point for curriculum-driven multi-task training.

Wires the StageTrainer to a torch optimiser + DataLoader + the
existing :class:`DepAwareStructuredModel` (Phase 3-A architecture).

Usage::

    PYTHONPATH=src python scripts/training_v2/train_curriculum.py \\
        --output_root runs/nextgen \\
        --warm_start runs/phase3a_491240/final \\
        --batch_size 32 \\
        --bf16

Pre-requisites
--------------

- ``data_v2/annotated/<source>/all.jsonl`` must exist (run
  ``scripts/data_v2/build_schema_v2_corpus.py`` first).
- A torch + transformers + (optionally) bitsandbytes environment.
- The Phase 3-A checkpoint at ``--warm_start``.

Stage transition policy
-----------------------

The CLI advances stages automatically when
:func:`CurriculumScheduler.advance_or_continue` returns ADVANCE
or TIMEOUT_ADVANCE. Each stage saves a checkpoint to
``<output_root>/stage_<id>/final/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=str(ROOT / "runs" / "nextgen"))
    ap.add_argument("--warm_start", default=str(ROOT / "runs" / "phase3a_491240" / "final"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_total_steps", type=int, default=80_000,
                    help="Hard upper bound across all stages")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--limit_corpus", type=int, default=None,
                    help="Limit total sentences (debug)")
    ap.add_argument("--eval_max_sentences", type=int, default=200,
                    help="Cap eval set to keep periodic gate-checks cheap")
    ap.add_argument("--encoder_name", default="UBC-NLP/AraT5v2-base-1024")
    args = ap.parse_args()

    print("=" * 70)
    print("Next-gen curriculum training")
    print("=" * 70)
    print(f"  output_root:      {args.output_root}")
    print(f"  warm_start:       {args.warm_start}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  lr:               {args.lr}")
    print(f"  bf16/fp16:        {args.bf16}/{args.fp16}")

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Lazy imports (torch only when training) -----
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from irab_tashkeel.curriculum import CurriculumScheduler, GateDecision
    from irab_tashkeel.data_v2.schema_v2 import read_jsonl
    from irab_tashkeel.training_v2 import (
        SchemaV2Collator, SchemaV2Dataset, StageTrainer, TrainerConfig,
        compute_multi_head_loss, gate_metrics_for_stage,
    )
    from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel

    # ----- Load corpus -----
    print("\n[1/5] Loading corpus...")
    sentences = []
    sources = ["distill_v2", "ud_padt_train", "ud_padt_dev",
               "masaq_quranic", "gazelle_test"]
    for src in sources:
        p = ROOT / "data_v2" / "annotated" / src / "all.jsonl"
        if p.exists():
            sentences.extend(list(read_jsonl(str(p))))
        else:
            print(f"  [warn] missing {p}")
    if args.limit_corpus:
        sentences = sentences[: args.limit_corpus]
    print(f"  Loaded {len(sentences)} sentences")

    # ----- Curriculum scheduler -----
    print("\n[2/5] Building curriculum scheduler...")
    sched = CurriculumScheduler.from_corpus(sentences, seed=args.seed)
    for sid, info in sched.stage_summary().items():
        print(f"  stage {sid} ({info['name']:25}): n={info['n_eligible']:5d}")

    # Held-out eval set: pick gazelle_test (gold MSA) by default; for
    # late stages MASAQ is also relevant. We use both for a single
    # eval-set; the per-stage metric breakdown distinguishes them.
    eval_sentences = [s for s in sentences
                      if s.metadata.source in ("gazelle_test", "masaq_quranic")]
    if args.eval_max_sentences and len(eval_sentences) > args.eval_max_sentences:
        # Deterministic: sort by id, take prefix. Keeps gazelle_test
        # entirely (only 30 sentences) and samples MASAQ.
        eval_sentences.sort(key=lambda s: s.sentence_id)
        eval_sentences = eval_sentences[: args.eval_max_sentences]
    print(f"  eval set: {len(eval_sentences)} sentences "
          f"(gazelle_test + masaq_quranic, capped)")

    # ----- Tokeniser + model -----
    print("\n[3/5] Loading tokeniser + warm-start checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.warm_start if Path(args.warm_start).exists() else args.encoder_name,
    )
    # Rebuild the DepAwareStructuredModel with the same flags as Phase 3-A
    model = DepAwareStructuredModel(
        encoder_name=args.encoder_name,
        enable_morph_heads=True, morph_heads_enabled=None,
        enable_dep_features=True,
    )
    if Path(args.warm_start).exists():
        sd = torch.load(Path(args.warm_start) / "pytorch_model.bin",
                        map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # ----- DataLoader machinery -----
    collator = SchemaV2Collator(tokenizer)
    cfg = TrainerConfig(
        encoder_name=args.encoder_name,
        warm_start_checkpoint=args.warm_start,
        learning_rate=args.lr, batch_size=args.batch_size,
        bf16=args.bf16, fp16=args.fp16, seed=args.seed,
        output_root=args.output_root,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    trainer = StageTrainer(cfg, sched, model, tokenizer, eval_sentences=[])

    # ----- Training loop -----
    print("\n[4/5] Starting curriculum training loop")
    global_step = 0
    while not sched.is_done() and global_step < args.max_total_steps:
        stage_cfg = sched.current_config
        stage_id = sched.current_stage_id
        print(f"\n--- Stage {stage_id}: {stage_cfg.name} ---")
        print(f"  eligible: {sched.current_pool.n}")
        if sched.current_pool.n == 0:
            print(f"  [skip] empty pool — advancing")
            sched.state.steps_in_stage = stage_cfg.target_steps
            sched.advance_or_continue({stage_cfg.gate_metric: 1.0})
            continue

        sampler = sched.make_sampler(batch_size=args.batch_size)

        # Stage-internal step loop
        stage_start_step = global_step
        last_log = time.time()
        while True:
            batch_sents = sampler.sample_batch()
            if not batch_sents:
                break
            # Convert via dataset wrapper + collator
            ds = SchemaV2Dataset(batch_sents)
            collated = collator([ds[i] for i in range(len(batch_sents))])
            collated = {k: (v.to(device) if hasattr(v, "to") else v)
                        for k, v in collated.items()}

            # Forward + loss
            out = model(
                input_ids=collated["input_ids"],
                attention_mask=collated["attention_mask"],
                word_starts=collated["word_starts"],
                word_ends=collated["word_ends"],
                word_mask=collated["word_mask"],
                return_dict=True,
            )
            logits = {
                "case":   out["case_logits"], "role":   out["role_logits"],
                "marker": out["marker_logits"], "pos":   out["pos_logits"],
            }
            for axis in ("gender", "number", "definite", "person",
                         "aspect", "mood", "voice"):
                key = f"{axis}_logits"
                if key in out: logits[f"morph_{axis}"] = out[key]
            labels = {
                "case": collated["case_labels"], "role": collated["role_labels"],
                "marker": collated["marker_labels"], "pos": collated["pos_labels"],
            }
            for axis in ("gender", "number", "definite", "person",
                         "aspect", "mood", "voice"):
                k = f"morph_{axis}_labels"
                if k in collated: labels[f"morph_{axis}"] = collated[k]

            ws = cfg.head_weights_for_stage(stage_id)
            res = compute_multi_head_loss(logits, labels, ws)
            loss = res["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            global_step += 1
            sched.state.steps_in_stage += 1

            # Periodic logging
            if global_step % 50 == 0:
                elapsed = time.time() - last_log
                print(f"  step {global_step}/{args.max_total_steps} stage {stage_id} "
                      f"loss {float(loss.item()):.4f} ({elapsed:.1f}s/50)")
                last_log = time.time()

            # Periodic gate check — runs the model on the held-out
            # eval set and computes the gate metric for the current
            # stage. Returns a dict; CurriculumScheduler picks out the
            # one matching the current stage's gate_metric.
            if global_step % args.eval_every == 0:
                model.eval()
                metrics = gate_metrics_for_stage(
                    stage_id, model, tokenizer, eval_sentences,
                    batch_size=args.batch_size,
                )
                model.train()
                metric_summary = ", ".join(
                    f"{k}={v:.3f}" for k, v in sorted(metrics.items())
                    if isinstance(v, (int, float))
                )
                print(f"  [eval] {metric_summary}")
                gate = sched.advance_or_continue(metrics)
                print(f"  [gate] {gate.decision.value}: {gate.reason}")
                if gate.decision in (GateDecision.ADVANCE, GateDecision.TIMEOUT_ADVANCE):
                    stage_dir = out_dir / f"stage_{stage_id}" / "final"
                    trainer.save_checkpoint(stage_dir,
                                              optimizer_state=optimizer.state_dict())
                    print(f"  [checkpoint] saved {stage_dir}")
                    break

            # Hard timeout per stage
            if sched.state.steps_in_stage >= stage_cfg.max_steps:
                print(f"  [stage timeout] {stage_cfg.max_steps} reached — advancing")
                stage_dir = out_dir / f"stage_{stage_id}" / "final"
                trainer.save_checkpoint(stage_dir,
                                          optimizer_state=optimizer.state_dict())
                # force-advance
                metrics = {}
                gate = sched.advance_or_continue(metrics)
                if gate.decision is GateDecision.CONTINUE:
                    # Rare: gate still says continue but we hit max_steps.
                    # Manually advance to avoid infinite loop.
                    sched.state.active_stage_id += 1
                    sched.state.steps_in_stage = 0
                break

    # ----- Final summary -----
    print("\n[5/5] Training complete")
    summary = {
        "global_step": global_step,
        "final_stage": sched.current_stage_id,
        "is_done": sched.is_done(),
        "history": sched.state.history,
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"Wrote {out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
