"""Run Sonnet RAG on the MASAQ eval set via Anthropic Messages Batches API.

Reuses the same prompt + retrieval code as the live Sonnet RAG baseline,
but submits all 624 verses as a single batch (50% off + ~30 min wait
instead of ~20 hr sync).

Output: runs/baseline_eval_masaq_sonnet/claude_rag.predictions.jsonl
in the same shape as the sync eval, so the existing scorer + subset
scoring just works.
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

from irab_tashkeel.inference.llm_baselines import (
    SYSTEM, _parse_json_array, load_combined_fewshots, retrieve_fewshots,
)
from irab_tashkeel.evaluation.run_baselines import _gold_pairs_from_jsonl

EVAL = Path("data/masaq_eval.jsonl")
OUT_DIR = Path("runs/baseline_eval_masaq_sonnet")
MODEL = "claude-sonnet-4-5"
RAG_K = 5


def build_prompt(sentence: str, pool, k: int = RAG_K) -> str:
    examples = retrieve_fewshots(sentence, pool, k=k)
    block = "".join(
        f"\nمثال:\nالجملة: {ex.sentence}\n{ex.irab_lines}\n" for ex in examples
    )
    return (
        "إليك أمثلة على الإعراب التقليدي:\n"
        f"{block}\n"
        "والآن أعرب الجملة التالية بنفس الأسلوب، وأخرج النتيجة كمصفوفة JSON كما حُدد في النظام:\n"
        f"الجملة: {sentence}"
    )


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = _gold_pairs_from_jsonl(EVAL)
    print(f"loaded {len(gold)} MASAQ verses ({sum(len(p) for _, p in gold)} word judgments)")

    pool = load_combined_fewshots(include_yarob=True, include_distilled=True)
    print(f"RAG pool size: {len(pool)}")

    client = Anthropic()

    print(f"Building {len(gold)} batch requests …", flush=True)
    requests: List[Request] = []
    by_id: Dict[str, dict] = {}
    for i, (sentence, gold_pairs) in enumerate(gold):
        cid = f"row_{i:05d}"
        by_id[cid] = {"sentence": sentence, "gold_pairs": gold_pairs}
        user = build_prompt(sentence, pool)
        requests.append(Request(
            custom_id=cid,
            params=MessageCreateParamsNonStreaming(
                model=MODEL, max_tokens=4096, temperature=0.0,
                system=SYSTEM,
                messages=[{"role": "user", "content": user}],
            ),
        ))
    print(f"submitting batch …", flush=True)
    batch = client.messages.batches.create(requests=requests)
    print(f"batch id: {batch.id}")

    while True:
        b = client.messages.batches.retrieve(batch.id)
        rc = b.request_counts
        print(f"  [{b.processing_status}] proc={rc.processing} ok={rc.succeeded} "
              f"err={rc.errored} cancel={rc.canceled} exp={rc.expired}", flush=True)
        if b.processing_status == "ended":
            break
        time.sleep(60)

    print(f"retrieving …", flush=True)
    in_total = out_total = 0
    out_path = OUT_DIR / "claude_rag.predictions.jsonl"
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
            row = {
                "sentence": src["sentence"],
                "gold": [{"word": w, "irab": ir} for w, ir in src["gold_pairs"]],
                "pred": items,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1

    cost = in_total / 1e6 * 1.5 + out_total / 1e6 * 7.5  # Sonnet 4.5 batch: $1.50/$7.50 per M
    print(f"\n{n_ok}/{len(gold)} succeeded; tokens in={in_total} out={out_total}")
    print(f"Sonnet 4.5 batch cost: ${cost:.2f}")
    print(f"wrote → {out_path}")


if __name__ == "__main__":
    main()
