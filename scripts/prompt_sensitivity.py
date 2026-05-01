#!/usr/bin/env python
"""Prompt-format sensitivity: re-run Sonnet RAG on Gazelle with an alternative
system prompt, then compare paired stats vs the headline.

Alt prompt: terser, English-instruction, same JSON output contract.
The point is to estimate robustness of the headline to prompt wording.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from irab_tashkeel.data.gazelle import load_gazelle_iraab
from irab_tashkeel.evaluation.run_baselines import _gold_pairs_from_gazelle, evaluate_baseline
from irab_tashkeel.inference.llm_baselines import (
    FewShotExample, _claude_call, _parse_json_array, WordIrab,
    load_combined_fewshots, retrieve_fewshots,
)


ALT_SYSTEM = """You are an expert Arabic syntactician (مدقّق نحوي). Given an MSA Arabic sentence, produce a per-word i'rāb analysis in the traditional grammatical-tradition style, with full diacritization.

Output rules (strict):
- Output ONLY a JSON array of objects, no surrounding text.
- Each object: word (undiacritized), diacritized, irab (full Arabic prose), pos, case, role, marker.
- case ∈ {rafʿ, naṣb, jarr, jazm, mabni}
- marker examples: الضمة الظاهرة، الفتحة الظاهرة، الكسرة الظاهرة، السكون، الواو، الياء، تنوين الفتح
- role examples: فاعل، مفعول به، مضاف إليه، اسم مجرور، حال، نعت، مبتدأ، خبر، اسم إن، خبر إن"""


def alt_rag(sentence: str, pool, k: int = 5, model: str = "claude-sonnet-4-5"):
    examples = retrieve_fewshots(sentence, pool, k=k)
    fewshot_block = ""
    for ex in examples:
        fewshot_block += f"\nExample:\nSentence: {ex.sentence}\n{ex.irab_lines}\n"
    user_msg = (
        f"Here are i'rāb examples:\n{fewshot_block}\n\n"
        f"Now produce the i'rāb for this sentence as a JSON array per the system spec:\n"
        f"Sentence: {sentence}"
    )
    raw, _, _ = _claude_call(
        [{"role": "system", "content": ALT_SYSTEM},
         {"role": "user", "content": user_msg}],
        model=model,
    )
    items = _parse_json_array(raw) or []
    return [WordIrab.from_dict(d) for d in items]


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")
    items = load_gazelle_iraab()
    gold = _gold_pairs_from_gazelle(items)
    sentences = [s for s, _ in gold]
    gold_pairs = [g for _, g in gold]
    pool = load_combined_fewshots(include_yarob=True, include_distilled=True)
    print(f"  RAG pool: {len(pool)}; n_sent={len(sentences)}")

    out_dir = Path("runs/baseline_eval_sonnet_altprompt")
    rep = evaluate_baseline(
        "claude_rag", sentences, gold_pairs,
        predict_fn=lambda s: alt_rag(s, pool, k=5),
        out_dir=out_dir,
    )
    print(json.dumps({k: v for k, v in rep["constrained"].items() if not k.startswith("_")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
