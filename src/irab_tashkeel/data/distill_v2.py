"""V2 distillation: Sonnet 4.5 + RAG via Anthropic Messages Batches API.

Reads candidate sentences from data/distill_v2/sources.jsonl (built by
`source_assembly.py`), runs each through the headline RAG pipeline (k=5
retrieval over Yarob+Distilled n=1060), submits as a Messages Batch for ~50%
discount + asynchronous polling, then reconstructs per-row JSON.

Output JSONL (compatible with FewShotExample.from_dict via the existing
`load_distilled_fewshots` loader):

    {"sentence": <str>,
     "items": [{"word", "diacritized", "irab", "pos", "case", "role", "marker"}, ...],
     "source": <str>,                   # provenance from sources.jsonl
     "well_formed_count": <int>,        # how many items pass structural extractor
     "n_words": <int>,
     "batch_id": <str>}

Usage:
    # 1. preview cost estimate before any spend
    python -m irab_tashkeel.data.distill_v2 --estimate --n 10000

    # 2. smoke test: 50 sentences, synchronous (immediate result)
    python -m irab_tashkeel.data.distill_v2 --smoke 50

    # 3. full batch run (~50% off):
    python -m irab_tashkeel.data.distill_v2 --batch --n 10000 \\
        --out data/distill_v2/distilled.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Costs (USD per 1M tokens) — model-keyed so --model selects the right pricing
# ---------------------------------------------------------------------------
COST_PER_M = {
    # Anthropic list prices (May 2026)
    "claude-sonnet-4-5": {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5":  {"in": 1.00, "out":  5.00},
    "claude-opus-4-7":   {"in": 15.0, "out": 75.00},
}

# Backward-compat aliases used by older code paths in this file
SONNET_INPUT_PER_M = COST_PER_M["claude-sonnet-4-5"]["in"]
SONNET_OUTPUT_PER_M = COST_PER_M["claude-sonnet-4-5"]["out"]

BATCH_DISCOUNT = 0.50         # Anthropic Messages Batches give 50% off

DEFAULT_MODEL = "claude-sonnet-4-5"


def cost_for(model: str, in_tokens: int, out_tokens: int, batch: bool = True) -> float:
    """Compute USD cost for a (in_tokens, out_tokens) pair under a given model."""
    if model not in COST_PER_M:
        raise ValueError(f"Unknown model {model!r}; add to COST_PER_M.")
    c = COST_PER_M[model]
    discount = BATCH_DISCOUNT if batch else 1.0
    return (in_tokens / 1e6 * c["in"] + out_tokens / 1e6 * c["out"]) * discount


# ---------------------------------------------------------------------------
# Source loading + RAG prompt construction
# ---------------------------------------------------------------------------
def load_sources(path: Path | str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def build_user_prompt(sentence: str, k: int = 5) -> str:
    from ..inference.llm_baselines import (
        load_combined_fewshots, retrieve_fewshots,
    )
    pool = build_user_prompt._pool  # cached on the function
    examples = retrieve_fewshots(sentence, pool, k=k)
    fewshot_block = ""
    for ex in examples:
        fewshot_block += f"\nمثال:\nالجملة: {ex.sentence}\n{ex.irab_lines}\n"
    return (
        "إليك أمثلة على الإعراب التقليدي:\n"
        f"{fewshot_block}\n"
        "والآن أعرب الجملة التالية بنفس الأسلوب، وأخرج النتيجة كمصفوفة JSON كما حُدد في النظام:\n"
        f"الجملة: {sentence}"
    )


def _ensure_pool_cached() -> None:
    if hasattr(build_user_prompt, "_pool"):
        return
    from ..inference.llm_baselines import load_combined_fewshots
    pool = load_combined_fewshots(include_yarob=True, include_distilled=True)
    build_user_prompt._pool = pool


# ---------------------------------------------------------------------------
# Cost estimation (no API spend)
# ---------------------------------------------------------------------------
def estimate_cost(sources: List[Dict[str, Any]], model: str = DEFAULT_MODEL,
                  use_batch: bool = True) -> Tuple[float, Dict[str, float]]:
    """Approximate cost based on per-sentence token estimates.

    Inputs: ~1300 tokens (5 fewshot Arabic blocks + sentence + system).
    Outputs: ~30 tokens × n_words + ~50 tokens JSON overhead.
    """
    in_per_m = SONNET_INPUT_PER_M
    out_per_m = SONNET_OUTPUT_PER_M
    if use_batch:
        in_per_m *= BATCH_DISCOUNT
        out_per_m *= BATCH_DISCOUNT

    avg_in_tokens = 1300
    avg_out_tokens_per_word = 30
    json_overhead_tokens = 50

    n = len(sources)
    in_tokens = avg_in_tokens * n
    out_tokens = sum(avg_out_tokens_per_word * (s.get("n_tokens") or 10)
                     + json_overhead_tokens for s in sources)

    in_cost = in_tokens / 1_000_000 * in_per_m
    out_cost = out_tokens / 1_000_000 * out_per_m
    total = in_cost + out_cost
    return total, {
        "n": n,
        "in_tokens_total": in_tokens,
        "out_tokens_total": out_tokens,
        "in_cost_usd": in_cost,
        "out_cost_usd": out_cost,
        "model": model,
        "batch": use_batch,
        "per_sentence_usd": total / max(1, n),
    }


# ---------------------------------------------------------------------------
# Synchronous smoke distillation (small N, immediate result)
# ---------------------------------------------------------------------------
def smoke_distill(sources: List[Dict[str, Any]], n: int,
                  out_path: Path, model: str = DEFAULT_MODEL,
                  k: int = 5) -> Path:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")
    from anthropic import Anthropic
    from ..inference.llm_baselines import SYSTEM, _parse_json_array
    from ..evaluation.structural import extract

    _ensure_pool_cached()
    client = Anthropic()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_actual = min(n, len(sources))
    in_total = out_total = 0
    n_well = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, src in enumerate(sources[:n_actual]):
            sentence = src["sentence"]
            user_msg = build_user_prompt(sentence, k=k)
            try:
                r = client.messages.create(
                    model=model, max_tokens=6144, temperature=0.0,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
            except Exception as e:
                print(f"  [{i+1}/{n_actual}] error: {e}")
                continue
            text = "".join(b.text for b in r.content if b.type == "text")
            items = _parse_json_array(text) or []
            in_total += r.usage.input_tokens
            out_total += r.usage.output_tokens
            for it in items:
                if isinstance(it, dict) and extract(it.get("irab", "")).well_formed:
                    n_well += 1
            row = {
                "sentence": sentence,
                "items": items,
                "source": src.get("source"),
                "id_in_source": src.get("id_in_source"),
                "n_words": len(sentence.split()),
                "well_formed_count": sum(1 for it in items if isinstance(it, dict)
                                          and extract(it.get("irab", "")).well_formed),
                "in_tokens": r.usage.input_tokens,
                "out_tokens": r.usage.output_tokens,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_actual}] avg in_tok={in_total/(i+1):.0f} out_tok={out_total/(i+1):.0f}", flush=True)
    cost = (in_total / 1_000_000 * SONNET_INPUT_PER_M
            + out_total / 1_000_000 * SONNET_OUTPUT_PER_M)
    print(f"\nSmoke complete: {n_actual} sentences")
    print(f"  total tokens: in={in_total}, out={out_total}")
    print(f"  cost (sync, no batch discount): ${cost:.3f}")
    print(f"  per-sentence (sync): ${cost / max(1, n_actual):.4f}")
    print(f"  per-sentence (batch -50%): ${cost / max(1, n_actual) * 0.5:.4f}")
    print(f"  written → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Batch distillation (Messages Batches API, 50% discount)
# ---------------------------------------------------------------------------
def batch_distill(sources: List[Dict[str, Any]], n: int,
                  out_path: Path, model: str = DEFAULT_MODEL,
                  k: int = 5) -> Path:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")
    from anthropic import Anthropic
    from anthropic.types.messages.batch_create_params import Request
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from ..inference.llm_baselines import SYSTEM, _parse_json_array
    from ..evaluation.structural import extract

    _ensure_pool_cached()
    client = Anthropic()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_actual = min(n, len(sources))
    requests: List[Request] = []
    src_by_id: Dict[str, Dict[str, Any]] = {}
    print(f"  building {n_actual} batch requests ...", flush=True)
    for i, src in enumerate(sources[:n_actual]):
        custom_id = f"row_{i:06d}"
        src_by_id[custom_id] = src
        user_msg = build_user_prompt(src["sentence"], k=k)
        requests.append(Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model=model, max_tokens=6144, temperature=0.0,
                system=SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            ),
        ))

    print(f"  submitting batch ({len(requests)} requests) ...", flush=True)
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch id: {batch.id}")
    print(f"  status: {batch.processing_status}")

    # Poll
    while True:
        b = client.messages.batches.retrieve(batch.id)
        rc = b.request_counts
        print(f"    [{b.processing_status}] processed={rc.processing} "
              f"succeeded={rc.succeeded} errored={rc.errored} "
              f"canceled={rc.canceled} expired={rc.expired}", flush=True)
        if b.processing_status == "ended":
            break
        time.sleep(30)

    # Collect results
    print(f"  retrieving results ...", flush=True)
    results = client.messages.batches.results(batch.id)
    in_total = out_total = 0
    n_ok = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in results:
            cid = entry.custom_id
            src = src_by_id.get(cid)
            if entry.result.type != "succeeded":
                continue
            msg = entry.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            items = _parse_json_array(text) or []
            in_total += msg.usage.input_tokens
            out_total += msg.usage.output_tokens
            wf = sum(1 for it in items if isinstance(it, dict)
                     and extract(it.get("irab", "")).well_formed)
            row = {
                "sentence": src["sentence"] if src else "",
                "items": items,
                "source": src.get("source") if src else "",
                "id_in_source": src.get("id_in_source") if src else "",
                "n_words": len(src["sentence"].split()) if src else 0,
                "well_formed_count": wf,
                "in_tokens": msg.usage.input_tokens,
                "out_tokens": msg.usage.output_tokens,
                "batch_id": batch.id,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1
    cost = (in_total / 1_000_000 * SONNET_INPUT_PER_M * BATCH_DISCOUNT
            + out_total / 1_000_000 * SONNET_OUTPUT_PER_M * BATCH_DISCOUNT)
    print(f"\nBatch complete: {n_ok}/{n_actual} succeeded")
    print(f"  total tokens: in={in_total}, out={out_total}")
    print(f"  cost (batch -50%): ${cost:.2f}")
    print(f"  written → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="V2 distillation (Sonnet RAG, batch API)")
    p.add_argument("--sources", type=Path, default=Path("data/distill_v2/sources.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/distill_v2/distilled.jsonl"))
    p.add_argument("--n", type=int, default=10_000, help="number of sentences to distill")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--k", type=int, default=5, help="RAG k")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--estimate", action="store_true",
                   help="print cost estimate; no API spend")
    g.add_argument("--smoke", type=int, default=0, metavar="N",
                   help="run synchronous smoke distillation on N sentences")
    g.add_argument("--batch", action="store_true",
                   help="submit a Messages Batches job (50% discount)")
    args = p.parse_args()

    if not args.sources.exists():
        print(f"sources file not found: {args.sources}\n"
              f"run `python -m irab_tashkeel.data.source_assembly` first")
        sys.exit(1)

    sources = load_sources(args.sources)
    print(f"loaded {len(sources)} candidate sentences from {args.sources}")

    if args.estimate:
        sub = sources[:args.n]
        total, breakdown = estimate_cost(sub, model=args.model, use_batch=True)
        print(json.dumps(breakdown, indent=2))
        # also report sync cost
        total_sync, _ = estimate_cost(sub, model=args.model, use_batch=False)
        print(f"\nsync cost (no batch discount): ${total_sync:.2f}")
        print(f"batch cost (-50% discount):    ${total:.2f}")
        return

    if args.smoke:
        out = args.out.parent / f"smoke_{args.smoke}.jsonl"
        smoke_distill(sources, n=args.smoke, out_path=out,
                      model=args.model, k=args.k)
        return

    if args.batch:
        batch_distill(sources, n=args.n, out_path=args.out,
                      model=args.model, k=args.k)
        return


if __name__ == "__main__":
    main()
