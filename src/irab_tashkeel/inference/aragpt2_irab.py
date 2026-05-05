"""AraGPT2-large + LoRA per-word i'rāb inference wrapper.

Same per-word call signature as `ArAT5IrabPredictor`, so it slots into
`run_baselines.py` and the structural-extraction metrics with no special
plumbing. Loads the base causal LM + a PEFT LoRA adapter from
`runs/irab_aragpt2_distill_v2_<jobid>/final/`.

Usage:
    from .aragpt2_irab import AraGPT2IrabPredictor
    pred = AraGPT2IrabPredictor("runs/irab_aragpt2_distill_v2_487443/final")
    items = pred.predict("ذهب الطالب إلى المدرسة")
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
class AraGPT2IrabPredictor:
    model_path: str
    base_model_id: Optional[str] = None
    max_input_length: int = 320
    max_new_tokens: int = 192
    device: Optional[str] = None

    def __post_init__(self):
        import torch
        # Stub `SequenceSummary` in transformers.modeling_utils before the
        # AraGPT2 dynamic module is imported. aubmindlab/aragpt2-large bundles
        # a `modeling_aragpt2.py` that does
        #     from transformers.modeling_utils import PreTrainedModel, SequenceSummary
        # and the symbol was removed in transformers ≥4.40. The class is only
        # used for the multiple-choice classification head we never invoke.
        import transformers.modeling_utils as _tmu
        if not hasattr(_tmu, "SequenceSummary"):
            class _SequenceSummaryStub:
                def __init__(self, *a, **kw): pass
            _tmu.SequenceSummary = _SequenceSummaryStub

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

        print(f"  [aragpt2_irab] tokenizer ← {self.model_path}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        print(f"  [aragpt2_irab] base ← {self.base_model_id} (dtype=bf16, device={self.device})", flush=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, torch_dtype=dtype, trust_remote_code=True,
        )
        if is_lora:
            from peft import PeftModel
            print(f"  [aragpt2_irab] LoRA adapter ← {self.model_path}", flush=True)
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def _generate_one(self, word: str, sentence: str) -> str:
        import torch
        prompt = _format_prompt(word, sentence)
        enc = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=self.max_input_length, add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=False,  # bundled modeling_aragpt2.py is incompatible with transformers ≥4.40 DynamicCache
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
        gen = out[0][prompt_len:]
        text = self.tok.decode(gen, skip_special_tokens=True).strip()
        # Trim at first newline (model can run on into a new prompt)
        if "\n" in text:
            text = text.split("\n", 1)[0].strip()
        return text

    def predict(self, sentence: str) -> List[WordIrab]:
        out: List[WordIrab] = []
        for w in _split_into_words(sentence):
            irab = self._generate_one(w, sentence)
            out.append(WordIrab(word=w, irab=irab))
        return out
