"""Convert MTLExample lists into instruction-tuning text pairs for LLM SFT.

Output format (model-agnostic; chat template applied per-model):

    SYSTEM:  أنت مدقق نحوي عربي خبير. أعرب الكلمات في الجملة التالية إعرابًا تامًا.
    USER:    <undiacritized sentence>
    ASSISTANT: word1: <full irab>
               word2: <full irab>
               ...

We only emit examples that have non-empty per-word i'rāb targets aligned to
the bare-text words. Examples without irab supervision are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from ...data.schema import MTLExample


SYSTEM_PROMPT = (
    "أنت مدقق نحوي عربي خبير. عند إعطائك جملة عربية، "
    "أعرب كل كلمة فيها إعرابًا تامًا على غرار الإعراب التقليدي. "
    "أخرج كل كلمة في سطر منفصل بصيغة: الكلمة: الإعراب الكامل."
)

USER_TEMPLATE = "أعرب الكلمات في الجملة التالية:\n{sentence}"


@dataclass
class SftPair:
    """One supervised fine-tuning instance."""
    system: str
    user: str
    assistant: str
    source: str
    n_words: int


def _format_assistant(words: List[str], irabs: List[str]) -> str:
    lines = []
    for w, ir in zip(words, irabs):
        if not ir or not w:
            continue
        lines.append(f"{w}: {ir}")
    return "\n".join(lines)


def example_to_pair(ex: MTLExample) -> Optional[SftPair]:
    """Convert one MTLExample to an SftPair, or None if it has no usable i'rāb."""
    if not ex.irab_targets or not any(ex.irab_targets):
        return None
    bare_words = ex.bare_text.split()
    if len(bare_words) != len(ex.irab_targets):
        return None
    irabs = list(ex.irab_targets)
    if not any(ir for ir in irabs):
        return None
    assistant = _format_assistant(bare_words, irabs)
    if not assistant:
        return None
    return SftPair(
        system=SYSTEM_PROMPT,
        user=USER_TEMPLATE.format(sentence=ex.bare_text),
        assistant=assistant,
        source=ex.source,
        n_words=len(bare_words),
    )


def examples_to_pairs(examples: Iterable[MTLExample]) -> List[SftPair]:
    """Bulk convert; drops examples with no usable irab."""
    out: List[SftPair] = []
    for ex in examples:
        p = example_to_pair(ex)
        if p is not None:
            out.append(p)
    return out


def pair_to_chat(pair: SftPair) -> List[dict]:
    """Convert an SftPair to a `messages` list compatible with HF chat templates."""
    return [
        {"role": "system", "content": pair.system},
        {"role": "user", "content": pair.user},
        {"role": "assistant", "content": pair.assistant},
    ]


def pair_to_chat_inference(sentence: str, system: str = SYSTEM_PROMPT) -> List[dict]:
    """Inference-time chat (no assistant turn)."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_TEMPLATE.format(sentence=sentence)},
    ]
