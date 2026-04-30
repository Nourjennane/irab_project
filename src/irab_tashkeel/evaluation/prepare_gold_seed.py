"""Generate a starter manual-eval set from PADT + Claude-RAG.

Manual i'rāb annotation is slow if you write from scratch. This script:
  1. Samples N sentences from PADT (length-balanced)
  2. Runs Claude few-shot RAG on each to seed i'rāb
  3. Writes a JSONL ready for human review (one row per sentence)

Format per row:
    {"sentence": ..., "items": [{"word": w, "irab": "<seed>", "verified": false}, ...]}

You then open it in any editor (or build a tiny Streamlit reviewer), correct
each `irab` field by hand, set `verified` to true. The resulting set is the
gold benchmark for the project.

The plan says this manual set is "non-negotiable" — without it every metric
is unfalsifiable.

Usage:
    ANTHROPIC_API_KEY=... python -m irab_tashkeel.evaluation.prepare_gold_seed \\
        --n 200 --out data/gold_seed.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List


def main():
    p = argparse.ArgumentParser(description="Seed a manual i'rāb gold set with Claude RAG")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("data/gold_seed.jsonl"))
    p.add_argument("--padt_dir", type=Path, default=Path("data/ud_padt"))
    p.add_argument("--model", default="claude-sonnet-4-5",
                   help="prefer Sonnet over Haiku for the seed (higher quality, $$ more)")
    p.add_argument("--rag_k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--budget_usd", type=float, default=10.0)
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")

    from ..data.ud_arabic import load_padt_examples
    from ..inference.llm_baselines import claude_fewshot_rag, load_yarob_fewshots
    from ..data.distill import COSTS, estimate_cost

    examples = load_padt_examples(args.padt_dir, download_if_missing=False)
    if not examples:
        sys.exit("PADT not found — run setup first")

    # Length-balanced sampling: 5-8w / 9-15w / 16-25w buckets, equal share each.
    buckets = defaultdict(list)
    for e in examples:
        n = len(e.word_offsets)
        if 5 <= n <= 8: buckets["short"].append(e)
        elif 9 <= n <= 15: buckets["mid"].append(e)
        elif 16 <= n <= 25: buckets["long"].append(e)
    rng = random.Random(args.seed)
    per_bucket = max(1, args.n // 3)
    chosen = []
    for k in ("short", "mid", "long"):
        rng.shuffle(buckets[k])
        chosen.extend(buckets[k][:per_bucket])
    rng.shuffle(chosen)
    chosen = chosen[: args.n]
    print(f"sampled {len(chosen)} PADT sentences across length buckets")

    pool = load_yarob_fewshots()
    print(f"yarob retrieval pool: {len(pool)}")

    cost_key = f"anthropic:{args.model}"
    if cost_key not in COSTS:
        print(f"⚠ unknown cost for {cost_key}; refusing to run without estimate")
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    spent = 0.0
    print(f"streaming output to {args.out} (line-buffered)", flush=True)
    with open(args.out, "w", encoding="utf-8", buffering=1) as f:
        for ex in chosen:
            if spent >= args.budget_usd:
                print(f"  budget hit (${spent:.2f}); stopping at {n_done}")
                break
            try:
                items = claude_fewshot_rag(ex.bare_text, pool, k=args.rag_k, model=args.model)
            except Exception as e:
                print(f"  err {ex.bare_text[:30]}…: {e}")
                continue
            row = {
                "sentence": ex.bare_text,
                "items": [
                    {"word": it.word, "irab": it.irab, "verified": False}
                    for it in items
                ],
                "source": "padt",
                "padt_sent_id": ex.sent_id,
                "seed_model": args.model,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += 1
            # rough cost estimate: input ~ 700 tok with RAG context, output ~ 700 tok
            spent += estimate_cost(1, 700, 700, cost_key)
            if n_done % 5 == 0:
                print(f"  [{n_done}/{args.n}]  spent ≈ ${spent:.2f}", flush=True)

    print(f"\n✓ wrote {n_done} sentences to {args.out}  (~${spent:.2f})")
    print(f"\nNext step: review each row, correct the 'irab' field, set 'verified': true.")
    print(f"Use this set for your final reported numbers.")


if __name__ == "__main__":
    main()
