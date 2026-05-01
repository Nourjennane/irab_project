"""Open-weight Arabic LLM baseline (Qwen2.5-7B-Instruct + RAG).

Sanity-check counterpart to Claude RAG. Uses the same retrieval pool, the
same Jaccard-similarity retrieval, and the same prompt format — only the
LLM changes from `claude-sonnet-4-5` to a local open-weight model loaded
in 4-bit quantization (fits on a single 8 GB consumer GPU).

The point of this baseline (per Hovy & Spruit 2016) is NOT to beat Claude;
it is to demonstrate that the closed system is being honestly compared to
a freely-available alternative whose weights and code are inspectable.

Usage:
    python -m irab_tashkeel.evaluation.run_baselines \\
        --eval gazelle --baselines openweight \\
        --openweight_model Qwen/Qwen2.5-7B-Instruct \\
        --rag_k 5 \\
        --out runs/baseline_eval_openweight
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .llm_baselines import (
    FewShotExample, SYSTEM, _parse_json_array, retrieve_fewshots, WordIrab,
)


@dataclass
class _OpenLLMState:
    model: Any = None
    tokenizer: Any = None
    name: str = ""


_state = _OpenLLMState()


def _load_model(model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Lazy-load the open-weight model in 4-bit on the first call."""
    if _state.model is not None and _state.name == model_id:
        return _state.model, _state.tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"  [openweight] loading {model_id} in 4-bit ...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, torch_dtype="auto",
    )
    model.eval()
    _state.model = model
    _state.tokenizer = tok
    _state.name = model_id
    print(f"  [openweight] {model_id} loaded.", flush=True)
    return model, tok


def _format_prompt(sentence: str, examples: Sequence[FewShotExample]) -> str:
    fewshot_block = ""
    for ex in examples:
        fewshot_block += f"\nمثال:\nالجملة: {ex.sentence}\n{ex.irab_lines}\n"
    user_msg = (
        "إليك أمثلة على الإعراب التقليدي:\n"
        f"{fewshot_block}\n"
        "والآن أعرب الجملة التالية بنفس الأسلوب، وأخرج النتيجة كمصفوفة JSON كما حُدد في النظام:\n"
        f"الجملة: {sentence}"
    )
    return user_msg


def openweight_fewshot_rag(
    sentence: str,
    pool: Sequence[FewShotExample],
    k: int = 5,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens: int = 1024,
) -> List[WordIrab]:
    """Run open-weight LLM with RAG context. Returns parsed WordIrab list."""
    model, tok = _load_model(model_id)
    examples = retrieve_fewshots(sentence, pool, k=k)
    user_msg = _format_prompt(sentence, examples)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    chat_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(chat_text, return_tensors="pt").to(model.device)
    import torch
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    raw = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    items = _parse_json_array(raw) or []
    return [WordIrab.from_dict(d) for d in items]
