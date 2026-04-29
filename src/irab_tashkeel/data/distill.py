"""LLM distillation harness for synthetic MSA i'rāb supervision.

Generates ~10–20K MSA i'rāb training pairs using GPT-4o (OpenAI) or Claude
3.5 Sonnet (Anthropic) as the teacher. Output is JSONL of MTLExample-shaped
dicts that `build_dataset.py` can ingest as a new source `distilled`.

User-triggered (not auto-invoked):
    OPENAI_API_KEY=... python -m irab_tashkeel.data.distill \\
        --provider openai --n 1000 --budget_usd 5 --out data/distilled_irab.jsonl

The script prints an estimated cost from a 50-sample dry run before spending.
Default seed sentences come from a small hard-coded MSA news corpus baked into
the script; pass --seed_file your_sentences.txt to override.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# Approximate per-million-token cost (USD). Update if pricing changes.
COSTS = {
    "openai:gpt-4o":         {"in": 2.50, "out": 10.00},
    "openai:gpt-4o-mini":    {"in": 0.15, "out":  0.60},
    "anthropic:claude-3-5-sonnet-latest": {"in": 3.00, "out": 15.00},
    "anthropic:claude-3-5-haiku-latest":  {"in": 0.80, "out":  4.00},
    "anthropic:claude-haiku-4-5":         {"in": 1.00, "out":  5.00},
    "anthropic:claude-sonnet-4-5":        {"in": 3.00, "out": 15.00},
    "anthropic:claude-sonnet-4-6":        {"in": 3.00, "out": 15.00},
    "anthropic:claude-opus-4-7":          {"in": 15.0, "out": 75.00},
}


SYSTEM = """أنت مدقق نحوي عربي خبير. عند إعطائك جملة عربية فصيحة (MSA)، أعرب كل كلمة في الجملة إعرابًا تامًا على غرار الإعراب التقليدي.

قواعد الإخراج (يجب الالتزام بها بدقة):
- أخرج JSON فقط (مصفوفة من الكائنات).
- كل كائن يحوي: word (الكلمة بدون تشكيل)، irab (الإعراب الكامل)، pos، case، role، marker.
- "case" من المجموعة: rafʿ، naṣb، jarr، jazm، mabni.
- "marker" مثل: الضمة الظاهرة، الفتحة الظاهرة، الكسرة الظاهرة، السكون، الواو، الياء، تنوين الفتح، …
- "role" من نحو: فاعل، مفعول به، مضاف إليه، اسم مجرور، حال، نعت، مبتدأ، خبر، مفعول مطلق، تمييز، …
- لا تكتب نصًا خارج الـ JSON."""

USER_TEMPLATE = "الجملة:\n{sentence}\n\nاكتب الإعراب الكامل لكل كلمة كمصفوفة JSON."


# A small bootstrap of MSA news sentences. Replace via --seed_file in real runs.
DEFAULT_SEEDS = [
    "أعلنت الحكومة عن إطلاق برنامج جديد لدعم الشركات الناشئة في مجال التكنولوجيا.",
    "تشير الدراسات الحديثة إلى ارتفاع ملحوظ في معدلات التوظيف خلال الربع الأخير.",
    "افتتح وزير الثقافة معرضًا فنيًا يضم أعمالًا لفنانين شباب من مختلف المحافظات.",
    "تعمل المدارس على تطوير مناهجها لمواكبة التحولات الرقمية المتسارعة في العالم.",
    "زار وفد اقتصادي رفيع المستوى العاصمة لبحث فرص التعاون التجاري بين البلدين.",
    "حذر الأطباء من خطورة الإفراط في تناول السكريات على صحة القلب والشرايين.",
    "شاركت الجامعة في مؤتمر دولي لمناقشة آخر التطورات في مجال الذكاء الاصطناعي.",
    "أكد المدرب أن الفريق جاهز لمواجهة التحديات في البطولة المقبلة بكل ثقة.",
    "أصدرت دار النشر طبعة جديدة من رواية شهيرة بعد سنوات من نفاد النسخ.",
    "تواصل الفرق التطوعية جهودها في تنظيف الشواطئ بمشاركة واسعة من السكان.",
]


@dataclass
class DistillSample:
    sentence: str
    items: List[Dict]   # raw teacher JSON: [{word, irab, pos, case, role, marker}, ...]
    teacher: str
    in_tokens: int
    out_tokens: int


def estimate_cost(n_samples: int, avg_in: int, avg_out: int, key: str) -> float:
    c = COSTS[key]
    return n_samples * (avg_in * c["in"] + avg_out * c["out"]) / 1_000_000


def call_openai(model: str, sentence: str) -> tuple[str, int, int]:
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TEMPLATE.format(sentence=sentence)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens


def call_anthropic(model: str, sentence: str) -> tuple[str, int, int]:
    from anthropic import Anthropic
    client = Anthropic()
    r = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(sentence=sentence)}],
        temperature=0.0,
    )
    text = "".join(block.text for block in r.content if block.type == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens


def parse_teacher_json(raw: str) -> Optional[List[Dict]]:
    """Extract a JSON array from the teacher output. Tolerant to extra text."""
    if not raw:
        return None
    text = raw.strip()
    # Strip code fences.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    # Many teachers wrap the array under a top-level key when forced JSON object.
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
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
    if not isinstance(obj, list) or not obj:
        return None
    out = []
    for entry in obj:
        if not isinstance(entry, dict):
            continue
        if "word" in entry and "irab" in entry:
            out.append(entry)
    return out or None


def main():
    p = argparse.ArgumentParser(description="LLM-distill MSA i'rāb pairs.")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p.add_argument("--model", default=None,
                   help="defaults: openai→gpt-4o-mini, anthropic→claude-3-5-haiku-latest")
    p.add_argument("--n", type=int, required=True, help="how many samples to generate")
    p.add_argument("--budget_usd", type=float, required=True, help="hard cost cap")
    p.add_argument("--out", type=Path, default=Path("data/distilled_irab.jsonl"))
    p.add_argument("--seed_file", type=Path, default=None,
                   help="optional file with one MSA sentence per line; falls back to bundled seeds")
    p.add_argument("--source", choices=["bundled", "padt"], default="bundled",
                   help="bundled = baked-in 10 sentences (only useful for tiny smoke runs). "
                        "padt = pull MSA news sentences from data/ud_padt (preferred for real runs).")
    p.add_argument("--padt_dir", type=Path, default=Path("data/ud_padt"))
    p.add_argument("--dry_run_n", type=int, default=10,
                   help="number of warm-up samples used to estimate cost before continuing")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    model = args.model or {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-latest",
    }[args.provider]
    cost_key = f"{args.provider}:{model}"
    if cost_key not in COSTS:
        sys.exit(f"unknown cost key {cost_key} — add it to COSTS in distill.py")

    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")

    if args.seed_file and args.seed_file.exists():
        seeds = [line.strip() for line in args.seed_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"using {len(seeds)} sentences from {args.seed_file}")
    elif args.source == "padt":
        from .ud_arabic import load_padt_examples
        ex = load_padt_examples(args.padt_dir, download_if_missing=False)
        # Filter to length-balanced 5-25 word sentences (the plan's spec).
        seeds = [e.bare_text for e in ex if 5 <= len(e.word_offsets) <= 25]
        print(f"using {len(seeds)} sentences from PADT (length 5-25 words)")
    else:
        seeds = DEFAULT_SEEDS
        print(f"⚠ using {len(seeds)} bundled seed sentences (pass --source padt for the real corpus)")
    if not seeds:
        sys.exit("no source sentences available; aborting")
    rng.shuffle(seeds)
    if len(seeds) < args.n:
        # Cycle through seeds to reach N (the teacher will paraphrase variations).
        cycles = (args.n + len(seeds) - 1) // len(seeds)
        seeds = (seeds * cycles)[: args.n]
    else:
        seeds = seeds[: args.n]

    call = call_openai if args.provider == "openai" else call_anthropic

    # ---- Dry run for cost estimate ----
    print(f"\n=== Dry run ({args.dry_run_n} samples) — model={model} ===")
    dry_in, dry_out, dry_n = 0, 0, 0
    dry_seen: List[DistillSample] = []
    for s in seeds[: args.dry_run_n]:
        try:
            raw, in_tok, out_tok = call(model, s)
        except Exception as e:
            print(f"  call failed: {e}")
            continue
        parsed = parse_teacher_json(raw)
        if parsed is None:
            print(f"  unparseable for: {s[:50]}…")
            continue
        dry_seen.append(DistillSample(sentence=s, items=parsed, teacher=cost_key,
                                       in_tokens=in_tok, out_tokens=out_tok))
        dry_in += in_tok
        dry_out += out_tok
        dry_n += 1
    if dry_n == 0:
        sys.exit("dry run produced no parseable samples; aborting before spending more")
    avg_in = dry_in // dry_n
    avg_out = dry_out // dry_n
    est_cost = estimate_cost(args.n, avg_in, avg_out, cost_key)
    print(f"  dry samples ok: {dry_n}/{args.dry_run_n}")
    print(f"  avg tokens: in={avg_in} out={avg_out}")
    print(f"  estimated cost for full run of {args.n}: ${est_cost:.2f}")
    if est_cost > args.budget_usd:
        sys.exit(f"estimated cost ${est_cost:.2f} > budget ${args.budget_usd:.2f}; aborting")

    # ---- Full run ----
    print(f"\n=== Generating {args.n} samples → {args.out} ===")
    n_done = len(dry_seen)
    spent = sum(estimate_cost(1, s.in_tokens, s.out_tokens, cost_key) for s in dry_seen)
    with open(args.out, "w", encoding="utf-8") as fout:
        for s in dry_seen:
            fout.write(json.dumps({"sentence": s.sentence, "items": s.items, "teacher": s.teacher}, ensure_ascii=False) + "\n")
        for i, sentence in enumerate(seeds[args.dry_run_n:], start=args.dry_run_n):
            if spent >= args.budget_usd:
                print(f"  ✋ budget hit (${spent:.2f}); stopping at {n_done} samples")
                break
            try:
                raw, in_tok, out_tok = call(model, sentence)
            except Exception as e:
                print(f"  [{i+1}/{args.n}] call failed: {e}; sleeping 5s")
                time.sleep(5)
                continue
            parsed = parse_teacher_json(raw)
            if parsed is None:
                continue
            fout.write(json.dumps({"sentence": sentence, "items": parsed, "teacher": cost_key}, ensure_ascii=False) + "\n")
            n_done += 1
            spent += estimate_cost(1, in_tok, out_tok, cost_key)
            if n_done % 50 == 0:
                print(f"  [{i+1}/{args.n}]  saved={n_done}  spent=${spent:.2f}")
    print(f"\n✓ saved {n_done} distilled samples → {args.out}  (~${spent:.2f})")


if __name__ == "__main__":
    main()
