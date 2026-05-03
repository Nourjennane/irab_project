"""Sonnet 4.5 zero-shot on MASAQ, batch API. First N verses only to fit budget.

Disentangles retrieval-pool-mismatch from model-register-effect:
- Sonnet RAG MASAQ (existing): retrieves from 100% MSA pool → register mismatch
- Sonnet ZERO-SHOT MASAQ (this script): no retrieval, model alone → isolates model
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

from irab_tashkeel.inference.llm_baselines import SYSTEM, USER_TEMPLATE, _parse_json_array
from irab_tashkeel.evaluation.run_baselines import _gold_pairs_from_jsonl

EVAL = Path("data/masaq_eval.jsonl")
OUT_DIR = Path("runs/baseline_eval_masaq_sonnet_zeroshot")
MODEL = "claude-sonnet-4-5"
N_VERSES = 400  # budget cap (~$5.5 at observed token rates)


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = _gold_pairs_from_jsonl(EVAL)[:N_VERSES]
    n_words = sum(len(p) for _, p in gold)
    print(f"Zero-shot eval on {len(gold)} verses ({n_words} word judgments)")

    client = Anthropic()
    by_id: Dict[str, dict] = {}
    requests: List[Request] = []
    for i, (sentence, gold_pairs) in enumerate(gold):
        cid = f"row_{i:05d}"
        by_id[cid] = {"sentence": sentence, "gold_pairs": gold_pairs}
        requests.append(Request(
            custom_id=cid,
            params=MessageCreateParamsNonStreaming(
                model=MODEL, max_tokens=4096, temperature=0.0,
                system=SYSTEM,
                messages=[{"role": "user", "content": USER_TEMPLATE.format(sentence=sentence)}],
            ),
        ))
    print(f"submitting batch …", flush=True)
    batch = client.messages.batches.create(requests=requests)
    print(f"batch id: {batch.id}", flush=True)

    while True:
        b = client.messages.batches.retrieve(batch.id)
        rc = b.request_counts
        print(f"  [{b.processing_status}] proc={rc.processing} ok={rc.succeeded} "
              f"err={rc.errored} cancel={rc.canceled} exp={rc.expired}", flush=True)
        if b.processing_status == "ended":
            break
        time.sleep(60)

    in_total = out_total = 0
    out_path = OUT_DIR / "claude_zero.predictions.jsonl"
    n_ok = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in client.messages.batches.results(batch.id):
            cid = entry.custom_id
            src = by_id.get(cid)
            if entry.result.type != "succeeded":
                continue
            msg = entry.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            items = _parse_json_array(text) or []
            in_total += msg.usage.input_tokens
            out_total += msg.usage.output_tokens
            f.write(json.dumps({
                "sentence": src["sentence"],
                "gold": [{"word": w, "irab": ir} for w, ir in src["gold_pairs"]],
                "pred": items,
            }, ensure_ascii=False) + "\n")
            n_ok += 1

    cost = in_total / 1e6 * 1.5 + out_total / 1e6 * 7.5
    print(f"\n{n_ok}/{len(gold)} succeeded; tokens in={in_total} out={out_total}")
    print(f"actual Sonnet 4.5 zero-shot batch cost: ${cost:.2f}")
    print(f"wrote → {out_path}")


if __name__ == "__main__":
    main()
