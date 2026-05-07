"""AceGPT-13B (Llama-2-13B Arabic-extended) + LoRA per-word i'rāb wrapper.

Same per-word call signature as `ArAT5IrabPredictor` and `AraGPT2IrabPredictor`,
so it slots into `run_baselines.py` and the structural-extraction metrics with
no special plumbing. Loads base in 4-bit NF4 (training was QLoRA), then attaches
the PEFT LoRA adapter.

Usage:
    from .acegpt_irab import AceGPTIrabPredictor
    pred = AceGPTIrabPredictor(
        "runs/irab_acegpt13b_distill_v2_<jobid>/final",
        base_model_id="/home/3415496/acegpt13b",  # local download path
    )
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .llm_baselines import WordIrab


PROMPT_TEMPLATE = (
    "أعرب الكلمة التالية في سياقها.\n"
    "الكلمة: {word}\n"
    "الجملة: {sentence}\n"
    "الإعراب: "
)


def _format_prompt(word: str, sentence: str, max_sent_len: int = 220) -> str:
    sent = (sentence or "").strip()
    if len(sent) > max_sent_len:
        sent = sent[:max_sent_len]
    return PROMPT_TEMPLATE.format(word=word.strip(), sentence=sent)


def _split_into_words(sentence: str) -> List[str]:
    return [t for t in re.split(r"\s+", (sentence or "").strip()) if t]


@dataclass
class AceGPTIrabPredictor:
    model_path: str                       # PEFT adapter dir
    base_model_id: Optional[str] = None   # base Llama-2 path / hub id (auto-resolved from adapter_config.json if None)
    max_input_length: int = 320
    # 192 was wasteful: measured p95 of MASAQ output is ~29 whitespace words
    # ≈ 60 subword tokens. 96 leaves ~40% headroom and saves ~30% wall time
    # on EOS-failure cases.
    max_new_tokens: int = 96
    device: Optional[str] = None
    load_in_4bit: bool = True             # match QLoRA training; loads weights in 4-bit NF4

    def __post_init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        adapter_cfg = Path(self.model_path) / "adapter_config.json"
        is_lora = adapter_cfg.exists()
        if is_lora and self.base_model_id is None:
            with open(adapter_cfg, encoding="utf-8") as f:
                self.base_model_id = json.load(f).get("base_model_name_or_path")
        if self.base_model_id is None and not is_lora:
            self.base_model_id = self.model_path

        print(f"  [acegpt_irab] tokenizer ← {self.model_path}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kwargs = {}
        if self.load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                # Required when device_map auto-spills to CPU (small consumer GPU).
                # Despite the int8-prefixed name, this flag also gates 4-bit
                # CPU offload in current bnb/transformers.
                llm_int8_enable_fp32_cpu_offload=True,
            )
            # Auto-device-map with CPU spillover for small consumer GPUs.
            # On 4060 (8 GB), 13B@4bit (~6.5 GB) + activations + KV cache spills
            # by ~1 GB; max_memory caps GPU at 6 GiB and lets CPU/RAM hold
            # the overflow layers. Slower than pure-GPU but actually runs.
            kwargs["device_map"] = "auto"
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_gb < 12:
                kwargs["max_memory"] = {0: f"{int(total_gb * 0.75)}GiB", "cpu": "24GiB"}
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        print(f"  [acegpt_irab] base ← {self.base_model_id} (4bit={self.load_in_4bit})", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_id, **kwargs)
        if is_lora:
            from peft import PeftModel
            print(f"  [acegpt_irab] LoRA adapter ← {self.model_path}", flush=True)
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
        if not self.load_in_4bit:
            self.model.to(self.device)
        self.model.eval()

    def _generate_one(self, word: str, sentence: str) -> str:
        import torch
        prompt = _format_prompt(word, sentence)
        enc = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=self.max_input_length, add_special_tokens=False)
        # When loaded in 4-bit, model is on cuda via device_map; otherwise we sent it.
        target_device = next(self.model.parameters()).device
        enc = {k: v.to(target_device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
        gen = out[0][prompt_len:]
        text = self.tok.decode(gen, skip_special_tokens=True).strip()
        if "\n" in text:
            text = text.split("\n", 1)[0].strip()
        return text

    def predict(self, sentence: str) -> List[WordIrab]:
        out: List[WordIrab] = []
        for w in _split_into_words(sentence):
            irab = self._generate_one(w, sentence)
            out.append(WordIrab(word=w, irab=irab))
        return out
