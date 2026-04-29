"""Combine all four data sources into a single training set.

Top-level entry: `build_combined_dataset(config)` returns a list[MTLExample].
"""

from __future__ import annotations

import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from .distilled_loader import load_distilled_examples
from .i3rab import load_i3rab_examples
from .qac import download_qac, load_qac_examples
from .schema import MTLExample
from .synthetic import generate_synthetic_examples
from .tashkeela import load_tashkeela_examples, load_tashkeela_sentences
from .ud_arabic import load_padt_examples
from .yarob import load_yarob_examples


def build_combined_dataset(
    tashkeela_n: int = 30000,
    qac_max_verses: Optional[int] = None,
    i3rab_path: Optional[Path] = None,
    synthetic_per_type: int = 2000,
    seed: int = 42,
    tashkeela_source: Optional[Path] = None,
    use_huggingface: bool = True,
    data_dir: Path = Path("data"),
    include_yarob: bool = True,
    include_padt: bool = True,
    include_distilled: bool = True,
) -> List[MTLExample]:
    """Load + combine all data sources.

    Returns a shuffled list of MTLExample ready for training.
    """
    rng = random.Random(seed)
    all_examples: List[MTLExample] = []

    # --- QAC ---
    qac_path = data_dir / "quran-morphology.txt"
    qac_examples = load_qac_examples(qac_path, max_verses=qac_max_verses)
    all_examples.extend(qac_examples)

    # --- Tashkeela ---
    try:
        tashkeela_examples = load_tashkeela_examples(
            source=tashkeela_source,
            max_sentences=tashkeela_n,
            use_huggingface=use_huggingface,
        )
        all_examples.extend(tashkeela_examples)
    except FileNotFoundError as e:
        print(f"⚠ Tashkeela not loaded: {e}")
        print("  Continuing without Tashkeela (QAC + I3rab + synthetic only).")
        tashkeela_examples = []

    # --- I3rab ---
    i3rab_examples = load_i3rab_examples(i3rab_path)
    all_examples.extend(i3rab_examples)

    # --- Yarob (manually curated per-word i'rab strings) ---
    if include_yarob:
        try:
            yarob_examples = load_yarob_examples(data_dir / "yarob_src")
            all_examples.extend(yarob_examples)
            if yarob_examples:
                print(f"  yarob: {len(yarob_examples)} examples")
        except Exception as e:
            print(f"⚠ Yarob not loaded: {e}")

    # --- UD_Arabic-PADT (free MSA news, ~6.7k sentences) ---
    if include_padt:
        try:
            padt_examples = load_padt_examples(data_dir / "ud_padt")
            all_examples.extend(padt_examples)
            if padt_examples:
                print(f"  ud_padt: {len(padt_examples)} examples")
        except Exception as e:
            print(f"⚠ UD-PADT not loaded: {e}")

    # --- LLM-distilled MSA pairs (only if you've run irab_tashkeel.data.distill) ---
    distilled_path = data_dir / "distilled_irab.jsonl"
    if include_distilled and distilled_path.exists():
        try:
            distilled = load_distilled_examples(distilled_path)
            all_examples.extend(distilled)
            if distilled:
                print(f"  distilled: {len(distilled)} examples")
        except Exception as e:
            print(f"⚠ distilled not loaded: {e}")

    # --- Synthetic errors ---
    # Collect gold diacritized sentences (QAC is our cleanest source)
    gold_pool = []
    for ex in qac_examples:
        # Reconstruct diacritized form
        from ..models.tokenizer import decode_diacritized
        if ex.mask_diac:
            diac = decode_diacritized(ex.bare_text, ex.diac_labels)
            if 30 < len(diac) < 400:
                gold_pool.append(diac)

    synthetic = generate_synthetic_examples(
        gold_sentences=gold_pool,
        per_type=synthetic_per_type,
        seed=seed,
    )
    all_examples.extend(synthetic)

    # Shuffle
    rng.shuffle(all_examples)

    return all_examples


def filter_by_word_count(
    examples: List[MTLExample], max_words: Optional[int]
) -> List[MTLExample]:
    """Length-based curriculum helper: keep only examples with ≤ max_words.

    Use to stage epochs (e.g., epoch 1 cap 8, epoch 2 cap 16, epoch 3 None).
    Pass `max_words=None` to disable filtering.
    """
    if max_words is None:
        return examples
    return [e for e in examples if len(e.word_offsets) <= max_words]


def report(examples: List[MTLExample]) -> Dict:
    """Print and return statistics about the dataset."""
    stats = {
        "total": len(examples),
        "by_source": dict(Counter(e.source for e in examples)),
        "mask_diac": sum(e.mask_diac for e in examples),
        "mask_irab": sum(e.mask_irab for e in examples),
        "mask_err": sum(e.mask_err for e in examples),
        "avg_length": sum(len(e.bare_text) for e in examples) / max(1, len(examples)),
        "avg_words": sum(len(e.word_offsets) for e in examples) / max(1, len(examples)),
    }
    print(f"Total examples: {stats['total']}")
    print(f"  by source:  {stats['by_source']}")
    print(f"  masks:      diac={stats['mask_diac']}, irab={stats['mask_irab']}, err={stats['mask_err']}")
    print(f"  avg chars:  {stats['avg_length']:.0f}")
    print(f"  avg words:  {stats['avg_words']:.1f}")
    return stats


def save_examples(examples: List[MTLExample], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(examples, f)


def load_examples(path: Path | str) -> List[MTLExample]:
    with open(path, "rb") as f:
        return pickle.load(f)
