"""Quick CLI for i'rāb generation.

Three backends:
  - decoder (your trained per-word decoder; offline, no API)
  - claude_zero / claude_rag (Claude API, needs ANTHROPIC_API_KEY)

Usage:
    # via API (best quality today, no GPU needed):
    ANTHROPIC_API_KEY=... python -m irab_tashkeel.inference.cli \\
        --backend claude_rag --sentence "ذهب الطالب إلى المدرسة"

    # offline:
    python -m irab_tashkeel.inference.cli \\
        --backend decoder --ckpt runs/model_small/best.pt \\
        --sentence "ذهب الطالب إلى المدرسة"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _run_decoder(ckpt: str, sentence: str):
    from .predictor import Predictor
    pred = Predictor.from_checkpoint(ckpt)
    r = pred.predict(sentence)
    return [{"word": w["surface"], "irab": w.get("irab_text", "")} for w in r.words]


def _run_claude(method: str, model: str, sentence: str, k: int):
    from .llm_baselines import (
        claude_zero_shot, claude_fewshot_rag, load_yarob_fewshots,
    )
    if method == "claude_zero":
        items = claude_zero_shot(sentence, model=model)
    else:
        pool = load_yarob_fewshots()
        items = claude_fewshot_rag(sentence, pool, k=k, model=model)
    return [{"word": it.word, "irab": it.irab, "case": it.case, "role": it.role} for it in items]


def main():
    p = argparse.ArgumentParser(description="i'rāb CLI")
    p.add_argument("--backend", choices=["decoder", "claude_zero", "claude_rag"],
                   default="claude_rag")
    p.add_argument("--sentence", default=None,
                   help="Arabic sentence to analyze (otherwise reads stdin)")
    p.add_argument("--ckpt", default="runs/model_small/best.pt",
                   help="checkpoint for the decoder backend")
    p.add_argument("--model", default="claude-haiku-4-5",
                   help="Claude model id")
    p.add_argument("--rag_k", type=int, default=5)
    p.add_argument("--json", action="store_true", help="emit JSON, not pretty text")
    args = p.parse_args()

    sentence = args.sentence or sys.stdin.read().strip()
    if not sentence:
        sys.exit("no sentence given (pass --sentence or via stdin)")

    if args.backend == "decoder":
        items = _run_decoder(args.ckpt, sentence)
    else:
        items = _run_claude(args.backend, args.model, sentence, args.rag_k)

    if args.json:
        print(json.dumps({"sentence": sentence, "items": items}, ensure_ascii=False, indent=2))
        return

    print(f"\n  الجملة: {sentence}\n")
    print(f"  {'word':20} | الإعراب")
    print(f"  {'-'*20}-+-{'-'*60}")
    for it in items:
        w = it.get("word", "")
        irab = it.get("irab", "")
        # Pad with explicit width that handles RTL approximately
        print(f"  {w:20} | {irab}")
    print()


if __name__ == "__main__":
    main()
