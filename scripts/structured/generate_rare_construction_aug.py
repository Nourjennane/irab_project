"""Generate targeted rare-construction augmentation via the Anthropic API.

For each of the 8 rare constructions (kana, inna, hal, tamyeez, badal,
exception, adjective_chain, idafa_chain), prompt Claude Haiku to produce
short MSA sentences with structured per-word labels (case + role + marker).
Output is appended to ``data/structured_v1/aug_rare_construction.jsonl`` in
the same SentenceIrab schema as the main training corpus.

Cost: ~640 sentences × ~250 tokens ≈ 160K tokens × Haiku 4.5 ≈ $0.20-1.

Usage:
    ANTHROPIC_API_KEY=sk-... python scripts/structured/generate_rare_construction_aug.py \\
        --n_per_construction 80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irab_tashkeel.structured.schema import (
    canonicalize_case, canonicalize_role, canonicalize_marker, derive_pos,
)
from irab_tashkeel.structured.word_irab import SentenceIrab, WordIrab


CONSTRUCTIONS = {
    "kana_sister": {
        "trigger_examples": "كان، ليس، أصبح، ظل، صار",
        "instruction": "Each sentence must begin with a kāna sister and have an explicit ism (raf') and khabar (nasb).",
    },
    "inna_sister": {
        "trigger_examples": "إن، أن، لكن، ليت، لعل",
        "instruction": "Each sentence must begin with an inna sister and have an explicit ism (nasb) and khabar (raf').",
    },
    "hal": {
        "trigger_examples": "ضاحكا، فرحا، مسرعا",
        "instruction": "Each sentence contains a hāl (state adverbial), case nasb.",
    },
    "tamyeez": {
        "trigger_examples": "علما، خلقا، شجاعة",
        "instruction": "Each sentence contains a tamyīz (specifier), case nasb, after a quantifier or comparative.",
    },
    "badal": {
        "trigger_examples": "محمد رسول الله، أبوك زيد",
        "instruction": "Each sentence contains a badal (apposition) that agrees in case with its head.",
    },
    "exception": {
        "trigger_examples": "إلا، سوى، عدا، خلا",
        "instruction": "Each sentence is an istithnāʾ construction with explicit exception particle.",
    },
    "adjective_chain": {
        "trigger_examples": "الطالب المجتهد النشيط",
        "instruction": "Each sentence contains 2+ adjectives modifying the same noun (chained naat).",
    },
    "idafa_chain": {
        "trigger_examples": "كتاب الطالب المجتهد",
        "instruction": "Each sentence contains an iḍāfa chain (noun + mudāf-ilayh, possibly multi-step).",
    },
}


PROMPT = """Generate {n} short MSA Arabic sentences targeting {tag} constructions.

{instruction}

For each sentence, output a single JSON line with this exact schema (no commentary, no markdown):
{{"sentence": "...", "items": [{{"word": "...", "irab": "<full Arabic i'rab prose>", "case": "<rafʿ|naṣb|jarr|jazm|mabni>", "role": "<canonical role term>", "marker": "<Arabic marker phrase>"}}, ...]}}

Constraints:
- Sentences must be 4-8 words.
- Use modern standard Arabic.
- Each item's "case" must be one of: rafʿ, naṣb, jarr, jazm, mabni.
- "role" should be the traditional iʿrāb role (مبتدأ، فاعل، خبر، مفعول به، اسم إن، خبر إن، اسم كان، خبر كان، نعت، حال، تمييز، بدل، مضاف إليه، اسم مجرور، حرف جر، حرف عطف، فعل، …).
- Diverse vocabulary across the {n} sentences.

Examples ({tag} triggers): {triggers}

Output {n} JSONL lines, one per sentence."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/structured_v1/aug_rare_construction.jsonl")
    ap.add_argument("--n_per_construction", type=int, default=80)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--dry_run", action="store_true",
                    help="Print prompts but don't call the API")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set. Use --dry_run or set the env var.",
              file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        client = None
    else:
        try:
            from anthropic import Anthropic
        except ImportError:
            print("ERROR: pip install anthropic", file=sys.stderr); sys.exit(2)
        client = Anthropic(api_key=api_key)

    written = 0
    with out_path.open("a") as fh:
        for tag, info in CONSTRUCTIONS.items():
            prompt = PROMPT.format(
                n=args.n_per_construction,
                tag=tag,
                instruction=info["instruction"],
                triggers=info["trigger_examples"],
            )
            if args.dry_run:
                print(f"--- {tag} ---\n{prompt[:300]}\n...")
                continue

            print(f"  generating {args.n_per_construction} sentences for {tag} ...", end=" ", flush=True)
            t0 = time.time()
            resp = client.messages.create(
                model=args.model,
                max_tokens=args.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            dt = time.time() - t0
            text = resp.content[0].text if resp.content else ""

            n_ok = 0
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                items = []
                for it in rec.get("items", []):
                    items.append(WordIrab(
                        word=it.get("word", ""),
                        case=canonicalize_case(it.get("case")),
                        role=canonicalize_role(it.get("role")),
                        marker=canonicalize_marker(it.get("marker")),
                        pos=derive_pos(it.get("role"), it.get("irab")),
                        irab_prose=it.get("irab"),
                    ))
                if not items:
                    continue
                sent = SentenceIrab(sentence=rec.get("sentence", ""), items=items)
                fh.write(sent.to_json_line() + "\n")
                n_ok += 1
                written += 1
            print(f"  {n_ok} kept ({dt:.1f}s)")

    print(f"\n  total written: {written}  ->  {out_path}")


if __name__ == "__main__":
    main()
