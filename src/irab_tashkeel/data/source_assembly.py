"""Source assembly for the v2 distillation campaign.

Pulls candidate Arabic sentences from multiple sources, normalises them, applies
length + language + register filters, deduplicates, and writes a single JSONL
with provenance metadata.

Sources:
  - PADT-UD (MSA news, local data/ud_padt/*.conllu)         ~7.7 K (~3.4 K post-filter)
  - Wikipedia AR first paragraphs (HF wikimedia/wikipedia)  configurable
  - Tashkeela classical (10 % register-diversity sample,
    local data/tashkeela/...)                               configurable

NyUAD-UD is intentionally NOT used: its CoNLL-U files have FORM=_ (fully
redacted for licensing); we cannot reconstruct surface text. Documented
limitation.

Length filter: 5–25 whitespace-separated tokens.

Output JSONL row:
    {"sentence": str, "source": str, "n_tokens": int, "len_chars": int,
     "id_in_source": str}

Usage:
    python -m irab_tashkeel.data.source_assembly --target 80000 --out data/distill_v2/sources.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

ARABIC_RE = re.compile(r"[؀-ۿ]")
NON_ARABIC_LETTERS_RE = re.compile(r"[A-Za-z一-鿿]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HASHTAG_AT_RE = re.compile(r"[@#]\S+")


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def normalise(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").strip()
    s = URL_RE.sub("", s)
    s = HASHTAG_AT_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_acceptable(s: str, min_tokens: int = 5, max_tokens: int = 25) -> bool:
    if not s:
        return False
    if not ARABIC_RE.search(s):
        return False
    # Reject mixed-language: any Latin letter -> reject
    if NON_ARABIC_LETTERS_RE.search(s):
        return False
    n = len(s.split())
    if n < min_tokens or n > max_tokens:
        return False
    # Reject mostly-numeric / mostly-punct
    arabic_chars = sum(1 for c in s if ARABIC_RE.match(c))
    if arabic_chars / max(1, len(s)) < 0.5:
        return False
    return True


# ---------------------------------------------------------------------------
# Loaders (one per source)
# ---------------------------------------------------------------------------
def iter_conllu_sentences(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (sentence_id, surface_text) from a CoNLL-U file."""
    sent_id = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# sent_id ="):
                sent_id = line.split("=", 1)[1].strip()
            elif line.startswith("# text ="):
                yield sent_id, line.split("=", 1)[1].strip()


def iter_padt(root: Path = Path("data/ud_padt")) -> Iterator[Tuple[str, str, str]]:
    for split in ("train", "dev", "test"):
        p = root / f"ar_padt-ud-{split}.conllu"
        if not p.exists():
            continue
        for sid, text in iter_conllu_sentences(p):
            yield (text, "padt", f"{sid}|{split}")


def iter_nyuad(root: Path = Path("data/ud_nyuad")) -> Iterator[Tuple[str, str, str]]:
    for split in ("train", "dev", "test"):
        p = root / f"ar_nyuad-ud-{split}.conllu"
        if not p.exists():
            continue
        for sid, text in iter_conllu_sentences(p):
            yield (text, "nyuad", f"{sid}|{split}")


def iter_wikipedia(target: int, seed: int = 42) -> Iterator[Tuple[str, str, str]]:
    """Pull Arabic Wikipedia first-paragraphs via HF datasets, sentence-split.

    Streams to avoid downloading the whole corpus.
    """
    from datasets import load_dataset
    print(f"  [wiki] streaming wikimedia/wikipedia (20231101.ar) ...", flush=True)
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.ar",
        split="train", streaming=True,
    )
    rng = random.Random(seed)
    splitter = re.compile(r"(?<=[.!؟])\s+")
    n_yielded = 0
    for ex in ds:
        if n_yielded >= target:
            break
        text = (ex.get("text") or "").strip()
        if not text:
            continue
        # Take first paragraph (split on double newline)
        first_para = text.split("\n\n", 1)[0]
        for sent in splitter.split(first_para):
            sent = sent.strip()
            if not sent:
                continue
            yield (sent, "wikipedia", str(ex.get("id", "")))
            n_yielded += 1
            if n_yielded >= target:
                break


def iter_tashkeela(target: int, root: Path = Path("data/tashkeela/Tashkeela-arabic-diacritized-text-utf8-0.3"),
                   seed: int = 42) -> Iterator[Tuple[str, str, str]]:
    """Sample classical Arabic snippets from Tashkeela for register diversity."""
    if not root.exists():
        print(f"  [tashkeela] {root} not found — skipping classical source")
        return
    files = list(root.rglob("*.txt"))
    rng = random.Random(seed)
    rng.shuffle(files)
    n = 0
    splitter = re.compile(r"(?<=[.!؟])\s+|\n+")
    for f in files:
        if n >= target:
            break
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for sent in splitter.split(text):
            sent = sent.strip()
            if not sent:
                continue
            # Strip diacritics for register-uniform downstream usage
            sent = re.sub(r"[ً-ْٰ]", "", sent)
            yield (sent, "tashkeela", f.name)
            n += 1
            if n >= target:
                break


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------
def assemble(target_total: int = 80_000,
             wiki_share: float = 0.5,
             tashkeela_share: float = 0.10,
             out_path: Path = Path("data/distill_v2/sources.jsonl"),
             seed: int = 42) -> Path:
    """Pull from each source, apply filters + dedup, write JSONL.

    Approximate composition: wiki_share + tashkeela_share + remainder from
    PADT+NyUAD (which together cap at ~27K usable rows).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_keys: set[str] = set()
    accepted: List[Tuple[str, str, str]] = []
    by_source: Counter = Counter()
    raw_by_source: Counter = Counter()

    sources_with_quota = [
        ("padt",       iter_padt,      None),                              # take all that pass
        ("tashkeela",  lambda: iter_tashkeela(int(target_total * tashkeela_share * 2.0)), None),
        ("wikipedia",  lambda: iter_wikipedia(int(target_total * wiki_share * 2.0)),     None),
    ]

    for src_name, src_iter, _ in sources_with_quota:
        if len(accepted) >= target_total:
            break
        print(f"\n[{src_name}] pulling ...", flush=True)
        try:
            for sent, source_label, sid in src_iter():
                raw_by_source[source_label] += 1
                norm = normalise(sent)
                if not is_acceptable(norm):
                    continue
                key = norm[:300]
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                accepted.append((norm, source_label, sid))
                by_source[source_label] += 1
                if len(accepted) >= target_total:
                    break
        except Exception as e:
            print(f"  [{src_name}] error: {e}")
            continue

    # Shuffle for random-order distillation
    rng = random.Random(seed)
    rng.shuffle(accepted)

    with open(out_path, "w", encoding="utf-8") as f:
        for sent, source, sid in accepted:
            row = {"sentence": sent, "source": source,
                   "n_tokens": len(sent.split()), "len_chars": len(sent),
                   "id_in_source": sid}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Stats
    print(f"\n=== Assembly summary ===")
    print(f"  total accepted: {len(accepted)}")
    print(f"  by source:")
    for src, n in by_source.most_common():
        raw = raw_by_source[src]
        kept_pct = (n / raw * 100) if raw else 0
        print(f"    {src:12} {n:>6}  (kept {kept_pct:.1f}% of {raw} raw)")
    print(f"  written → {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Source assembly for distill v2")
    p.add_argument("--target", type=int, default=80_000,
                   help="target accepted-sentence count")
    p.add_argument("--wiki_share", type=float, default=0.5,
                   help="approx fraction of total to pull from Wikipedia")
    p.add_argument("--tashkeela_share", type=float, default=0.10,
                   help="approx fraction of total to pull from Tashkeela classical")
    p.add_argument("--out", type=Path, default=Path("data/distill_v2/sources.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    assemble(target_total=args.target, wiki_share=args.wiki_share,
             tashkeela_share=args.tashkeela_share, out_path=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
