"""Phase R — build the grammar memory from a training corpus.

Iterates the iʿrāb-supervised training records, detects constructions,
encodes each construction span with the Phase 3-A encoder (mean-pooled
``pooled_irab`` features), and writes per-family JSONL + FAISS index
to ``data/grammar_memory/{family}/``.

The encoder is loaded from the production checkpoint
(``runs/phase3a_491240/final``). We do a single forward pass per
sentence and slice per-construction spans from the per-word features.

Usage:
    python scripts/grammar_memory/build_memory.py \\
        --model runs/phase3a_491240/final \\
        --in_corpus data/morph_v1_dep/train.jsonl \\
        --out_dir data/grammar_memory/

Compute: ~15 min on a single GPU for 4,750 distill_v2 sentences.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to Phase 3-A final dir")
    ap.add_argument("--in_corpus", required=True,
                    help="iʿrāb-supervised training corpus (data/morph_v1_dep/train.jsonl)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_records", type=int, default=None,
                    help="optional cap for testing")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import numpy as np
    import torch

    from irab_tashkeel.grammar_memory.signature import (
        detect_constructions_in_record, build_signature, ALL_FAMILIES,
    )
    from irab_tashkeel.grammar_memory.memory import (
        GrammarMemoryBuilder, save_build_summary,
    )
    from irab_tashkeel.inference.structured_predictor import (
        StructuredPredictor, StructuredPredictorConfig,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 3-A predictor
    print(f"[build_memory] loading Phase 3-A predictor from {args.model}")
    cfg = StructuredPredictorConfig(
        apply_constraints=False,
        apply_hierarchical=False,
        return_attention=False,
        render_prose=False,
        device="auto",
    )
    pred = StructuredPredictor(args.model, cfg=cfg)
    model = pred.model
    tokenizer = pred.tokenizer
    device = pred.device
    print(f"[build_memory] device={device}")

    # Iterate corpus
    in_path = Path(args.in_corpus)
    print(f"[build_memory] reading {in_path}")
    records: List[Dict] = []
    with in_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Only iʿrāb-supervised records
            if not rec.get("has_irab"):
                continue
            records.append(rec)
            if args.max_records and len(records) >= args.max_records:
                break
    print(f"[build_memory] {len(records)} iʿrāb-supervised records")

    builders: Dict[str, GrammarMemoryBuilder] = {
        f: GrammarMemoryBuilder(f) for f in ALL_FAMILIES
    }
    family_count: Counter = Counter()
    total_spans = 0
    n_no_construction = 0

    # For span embedding: do a forward pass on the encoder + dep_proj path,
    # then mean-pool over the construction-span words.
    def encode_sentence(sentence: str):
        """Run Phase 3-A encoder + dep_proj path; return per-word pooled_irab."""
        enc = pred._encode_sentence(sentence)
        if enc is None:
            return None, None
        # Forward through encoder + word-pool. We don't need the iʿrāb heads
        # for retrieval — we just want the encoder's per-word feature.
        with torch.no_grad():
            from irab_tashkeel.structured.model import _word_first_pool
            enc_out = model.encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            hidden = enc_out.last_hidden_state
            pooled = _word_first_pool(hidden, enc["word_starts"], enc["word_mask"])
            # If dep features are enabled in the model, we have a dep_proj layer;
            # apply it with zero dep_emb (matches inference path with no dep tensors).
            if getattr(model, "enable_dep_features", False):
                B, W = pooled.shape[:2]
                dep_emb = pooled.new_zeros(
                    B, W,
                    model.dep_feature_encoder.deprel_embed.embedding_dim
                    + model.dep_feature_encoder.head_dir_embed.embedding_dim
                    + model.dep_feature_encoder.head_dist_embed.embedding_dim
                    + model.dep_feature_encoder.gov_pos_embed.embedding_dim,
                )
                h_aug = torch.cat([pooled, dep_emb], dim=-1)
                pooled_irab = model.dep_proj(h_aug)
            else:
                pooled_irab = pooled
        return pooled_irab[0].cpu().numpy(), enc["words"]

    # Iterate records
    for sidx, rec in enumerate(records):
        sentence = rec.get("sentence", "")
        if not sentence:
            continue
        spans = detect_constructions_in_record(rec)
        if not spans:
            n_no_construction += 1
            continue

        # Encode sentence once
        per_word_emb, words = encode_sentence(sentence)
        if per_word_emb is None:
            continue
        n_words = per_word_emb.shape[0]

        for span_desc in spans:
            start, end = span_desc["span"]
            # Clamp to actual encoded words
            start = max(0, min(start, n_words))
            end = max(start, min(end, n_words))
            if end <= start:
                continue
            span_emb = per_word_emb[start:end].mean(axis=0)  # (768,)

            inst = build_signature(rec, span_desc, sentence_idx=sidx)
            family = inst.construction
            builders[family].add(inst, span_emb)
            family_count[family] += 1
            total_spans += 1

        if (sidx + 1) % 500 == 0:
            print(f"  ... processed {sidx + 1}/{len(records)} records "
                  f"({total_spans} construction spans indexed)")

    print(f"\n[build_memory] indexed {total_spans} construction spans across "
          f"{len(records) - n_no_construction} sentences "
          f"({n_no_construction} sentences had no detected construction)")
    for f, n in sorted(family_count.items(), key=lambda kv: -kv[1]):
        print(f"  {f}: {n}")

    # Save per-family
    summaries: List[Dict] = []
    for family, builder in builders.items():
        s = builder.save(out_dir / family)
        summaries.append(s)
    save_build_summary(out_dir, summaries)

    print(f"\n[build_memory] memory written to {out_dir}")


if __name__ == "__main__":
    main()
