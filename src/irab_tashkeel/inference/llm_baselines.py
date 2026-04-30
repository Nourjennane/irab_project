"""LLM-based baselines: Claude zero-shot and Claude few-shot RAG.

These are immediately-shippable systems (no GPU training needed). Use them as:
  1. The bar that any fine-tuned model must clear, AND
  2. A working production fallback if fine-tuning fails or is overkill.

Reads ANTHROPIC_API_KEY from the environment. Never commit the key.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SYSTEM = """أنت مدقق نحوي عربي خبير. عند إعطائك جملة عربية فصيحة (MSA)، أعرب كل كلمة في الجملة إعرابًا تامًا على غرار الإعراب التقليدي.

قواعد الإخراج (يجب الالتزام بها بدقة):
- أخرج JSON فقط (مصفوفة من الكائنات).
- كل كائن يحوي: word (الكلمة بدون تشكيل)، irab (الإعراب الكامل)، pos، case، role، marker.
- "case" من المجموعة: rafʿ، naṣb، jarr، jazm، mabni.
- "marker" مثل: الضمة الظاهرة، الفتحة الظاهرة، الكسرة الظاهرة، السكون، الواو، الياء، تنوين الفتح.
- "role" من نحو: فاعل، مفعول به، مضاف إليه، اسم مجرور، حال، نعت، مبتدأ، خبر، اسم إن، خبر إن، مفعول مطلق، تمييز.
- لا تكتب نصًا خارج الـ JSON."""

USER_TEMPLATE = "الجملة:\n{sentence}\n\nاكتب الإعراب الكامل لكل كلمة كمصفوفة JSON."


@dataclass
class WordIrab:
    word: str
    irab: str
    pos: Optional[str] = None
    case: Optional[str] = None
    role: Optional[str] = None
    marker: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WordIrab":
        return cls(
            word=str(d.get("word", "")),
            irab=str(d.get("irab", "")),
            pos=d.get("pos"), case=d.get("case"),
            role=d.get("role"), marker=d.get("marker"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"word": self.word, "irab": self.irab, "pos": self.pos,
                "case": self.case, "role": self.role, "marker": self.marker}


# ---------------------------------------------------------------------------
# Robust JSON extraction (Claude often wraps in code fences or adds preamble)
# ---------------------------------------------------------------------------
def _parse_json_array(raw: str) -> Optional[List[Dict[str, Any]]]:
    if not raw:
        return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                obj = v
                break
        else:
            return None
    if not isinstance(obj, list):
        return None
    return [e for e in obj if isinstance(e, dict)]


# ---------------------------------------------------------------------------
# Retrieval (cheap word-jaccard) for the few-shot RAG baseline
# ---------------------------------------------------------------------------
def _normalize_for_match(s: str) -> List[str]:
    s = re.sub(r"[ً-ْٰ]", "", s or "")
    return re.findall(r"[ء-ي]+", s)


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class FewShotExample:
    sentence: str
    irab_lines: str   # multi-line "word: irab" formatted answer


def load_yarob_fewshots(repo_dir: Path | str = "data/yarob_src") -> List[FewShotExample]:
    """Load Yarob examples as (sentence, formatted irab block) pairs.

    Used as the retrieval pool for the few-shot RAG baseline.
    """
    from ..data.yarob import load_yarob_examples
    out: List[FewShotExample] = []
    for ex in load_yarob_examples(repo_dir, download_if_missing=False):
        words = ex.bare_text.split()
        irabs = ex.irab_targets or []
        if len(words) != len(irabs):
            continue
        block = "\n".join(f"{w}: {i}" for w, i in zip(words, irabs) if i)
        out.append(FewShotExample(sentence=ex.bare_text, irab_lines=block))
    return out


def load_distilled_fewshots(
    path: Path | str = "data/distilled_irab.jsonl",
) -> List[FewShotExample]:
    """Load Claude-distilled MSA pairs as RAG few-shot examples."""
    import json as _json
    path = Path(path)
    if not path.exists():
        return []
    out: List[FewShotExample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            sent = (row.get("sentence") or "").strip()
            items = row.get("items") or []
            if not sent or not items:
                continue
            block_lines: List[str] = []
            for it in items:
                w = (it.get("word") or "").strip()
                ir = (it.get("irab") or "").strip()
                if w and ir:
                    block_lines.append(f"{w}: {ir}")
            if not block_lines:
                continue
            out.append(FewShotExample(sentence=sent, irab_lines="\n".join(block_lines)))
    return out


def load_combined_fewshots(
    include_yarob: bool = True,
    include_distilled: bool = True,
    yarob_dir: Path | str = "data/yarob_src",
    distilled_path: Path | str = "data/distilled_irab.jsonl",
) -> List[FewShotExample]:
    """Combine Yarob (manual gold) + distilled (Claude-generated MSA) pools."""
    pool: List[FewShotExample] = []
    if include_yarob:
        pool.extend(load_yarob_fewshots(yarob_dir))
    if include_distilled:
        pool.extend(load_distilled_fewshots(distilled_path))
    return pool


def retrieve_fewshots(
    query: str, pool: Sequence[FewShotExample], k: int = 5,
) -> List[FewShotExample]:
    qtok = _normalize_for_match(query)
    scored = [(p, _jaccard(qtok, _normalize_for_match(p.sentence))) for p in pool]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [p for p, _ in scored[:k]]


# ---------------------------------------------------------------------------
# Claude callers
# ---------------------------------------------------------------------------
def _claude_call(messages: List[Dict[str, str]], model: str, max_tokens: int = 2048) -> Tuple[str, int, int]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    from anthropic import Anthropic
    client = Anthropic()
    r = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=messages[0]["content"] if messages and messages[0]["role"] == "system" else SYSTEM,
        messages=[m for m in messages if m["role"] != "system"],
        temperature=0.0,
    )
    text = "".join(block.text for block in r.content if block.type == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens


def claude_zero_shot(sentence: str, model: str = "claude-haiku-4-5") -> List[WordIrab]:
    """Vanilla zero-shot Claude i'rab."""
    raw, _, _ = _claude_call(
        [{"role": "user", "content": USER_TEMPLATE.format(sentence=sentence)}],
        model=model,
    )
    items = _parse_json_array(raw) or []
    return [WordIrab.from_dict(d) for d in items]


def claude_fewshot_rag(
    sentence: str,
    pool: Sequence[FewShotExample],
    k: int = 5,
    model: str = "claude-haiku-4-5",
) -> List[WordIrab]:
    """Few-shot RAG: retrieve k similar Yarob sentences, prepend as in-context examples."""
    examples = retrieve_fewshots(sentence, pool, k=k)
    fewshot_block = ""
    for ex in examples:
        fewshot_block += f"\nمثال:\nالجملة: {ex.sentence}\n{ex.irab_lines}\n"

    user_msg = (
        "إليك أمثلة على الإعراب التقليدي:\n"
        f"{fewshot_block}\n"
        "والآن أعرب الجملة التالية بنفس الأسلوب، وأخرج النتيجة كمصفوفة JSON كما حُدد في النظام:\n"
        f"الجملة: {sentence}"
    )
    raw, _, _ = _claude_call([{"role": "user", "content": user_msg}], model=model)
    items = _parse_json_array(raw) or []
    return [WordIrab.from_dict(d) for d in items]


# ---------------------------------------------------------------------------
# Convenience: run a baseline on a list of sentences and write a JSONL
# ---------------------------------------------------------------------------
def run_baseline(
    sentences: Sequence[str],
    out_path: Path | str,
    method: str = "zero_shot",
    model: str = "claude-haiku-4-5",
    fewshot_pool: Optional[Sequence[FewShotExample]] = None,
    k: int = 5,
) -> Path:
    """Run a baseline on every sentence and write JSONL: {sentence, items}."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fn = (lambda s: claude_zero_shot(s, model=model)) if method == "zero_shot" else \
         (lambda s: claude_fewshot_rag(s, fewshot_pool or [], k=k, model=model))
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sentences:
            try:
                items = fn(s)
            except Exception as e:
                print(f"  [{method}] error on '{s[:40]}…': {e}")
                continue
            f.write(json.dumps({
                "sentence": s,
                "items": [i.to_dict() for i in items],
                "method": method, "model": model,
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"  ✓ wrote {n} predictions to {out_path}")
    return out_path
