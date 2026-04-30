"""Extract marker phrases from gold/silver i'rāb strings.

The marker is the surface morphological sign — الضمة الظاهرة, الفتحة المقدرة,
السكون, الواو لأنه جمع مذكر سالم, etc. It's the phrase that follows
"وعلامة رفعه" / "نصبه" / "جره" / "جزمه" in declinable words, or appears
after "مبني على" in indeclinable words.

For Mix A (per-word routing), AraT5v2 is fine-tuned to predict ONLY this
marker phrase, conditioned on (sentence, word, case, role) — Claude RAG
already nails case and role; AraT5v2 should specialize on marker style.

Outputs (per word):
    marker_phrase : the extracted marker (or None for "no-marker" class)
    has_marker    : bool
    marker_kind   : "declinable" | "mabni_surface" | "mahall" | "none"
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Regexes (work on normalized text — diacritics stripped, NFC-normalized)
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[ً-ْٰ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


# Declinable: "وعلامة رفعه/نصبه/جره/جزمه <MARKER> [<rest>]"
# Marker phrase ends at sentence-ending punctuation, "،", "وهو", "في محل",
# "لأنه", or end-of-string. Capture greedily up to ~8 words.
_DECLINABLE_RE = re.compile(
    r"وعلامة\s+(?:رفعه|نصبه|جره|جزمه|رفع[ه]?|نصب[ه]?|جر[ه]?|جزم[ه]?)"
    r"\s+(?P<marker>[^.،؛]{2,80}?)"
    r"(?=\s*(?:،|\.|؛|$|\bوهو\b|\bفي محل\b|\bلأن\b|\bلأنه\b|\bمتعلق\b|\bالواو\b))"
)

# Mabni with surface sign: "مبني على <MARKER>"
_MABNI_RE = re.compile(
    r"مبني[ةه]?\s+على\s+(?P<marker>[^.،؛]{2,40}?)"
    r"(?=\s*(?:،|\.|؛|$|\bفي محل\b|\bلا محل\b|\bمحل\b|\bوهو\b))"
)

# Pure positional ("في محل X") — no surface marker, classify as mahall
_MAHALL_RE = re.compile(r"في محل\s+(رفع|نصب|جر|جزم)")


@dataclass
class MarkerLabel:
    marker_phrase: Optional[str]
    has_marker: bool
    marker_kind: str   # "declinable" | "mabni_surface" | "mahall" | "none"


def extract_marker(irab_text: str) -> MarkerLabel:
    """Run the regex pipeline on a single i'rāb string.

    Order matters: declinable pattern first (it's the most specific and
    high-volume); mabni surface next; mahall as fallback flag.
    """
    text = _norm(irab_text)
    if not text:
        return MarkerLabel(None, False, "none")

    m = _DECLINABLE_RE.search(text)
    if m:
        marker = m.group("marker").strip(" ،.،؛:")
        # Drop trailing "في آخره" / "على آخره" — those are positional fluff
        marker = re.sub(r"\s+(?:على|في|عند)\s+آخر[ه]?\s*$", "", marker).strip()
        if marker:
            return MarkerLabel(marker, True, "declinable")

    m = _MABNI_RE.search(text)
    if m:
        marker = m.group("marker").strip(" ،.،؛:")
        if marker:
            return MarkerLabel(marker, True, "mabni_surface")

    if _MAHALL_RE.search(text):
        return MarkerLabel(None, False, "mahall")

    return MarkerLabel(None, False, "none")


# ---------------------------------------------------------------------------
# Build training data for the AraT5v2 marker fine-tune
# ---------------------------------------------------------------------------
@dataclass
class MarkerTrainingPair:
    sentence: str
    word: str
    case: Optional[str]      # rafʿ / naṣb / jarr / jazm / mabni / None
    role: Optional[str]      # textual role label or None
    marker_target: str       # canonical marker; "<NO_MARKER>" for mahall/none
    marker_kind: str
    source: str              # which dataset


def build_marker_pairs_from_distilled(
    path: Path | str,
) -> List[MarkerTrainingPair]:
    """Convert distilled JSONL to marker training pairs."""
    pairs: List[MarkerTrainingPair] = []
    path = Path(path)
    if not path.exists():
        return pairs
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sent = (row.get("sentence") or "").strip()
            items = row.get("items") or []
            if not sent or not items:
                continue
            for it in items:
                word = (it.get("word") or "").strip()
                irab = (it.get("irab") or "").strip()
                if not word or not irab:
                    continue
                lab = extract_marker(irab)
                pairs.append(MarkerTrainingPair(
                    sentence=sent,
                    word=word,
                    case=it.get("case"),
                    role=it.get("role"),
                    marker_target=lab.marker_phrase or "<NO_MARKER>",
                    marker_kind=lab.marker_kind,
                    source="distilled",
                ))
    return pairs


def build_marker_pairs_from_yarob(
    repo_dir: Path | str = "data/yarob_src",
) -> List[MarkerTrainingPair]:
    from ..data.yarob import load_yarob_examples
    pairs: List[MarkerTrainingPair] = []
    for ex in load_yarob_examples(repo_dir, download_if_missing=False):
        words = ex.bare_text.split()
        irabs = ex.irab_targets or []
        if len(words) != len(irabs):
            continue
        for w, ir in zip(words, irabs):
            if not ir:
                continue
            lab = extract_marker(ir)
            # Yarob doesn't carry case/role labels; leave them None — the
            # downstream encoder can run a coarse classifier or rely on
            # the sentence context to infer case.
            pairs.append(MarkerTrainingPair(
                sentence=ex.bare_text, word=w,
                case=None, role=None,
                marker_target=lab.marker_phrase or "<NO_MARKER>",
                marker_kind=lab.marker_kind,
                source="yarob",
            ))
    return pairs


def build_combined_marker_pairs(
    distilled_path: Path | str = "data/distilled_irab.jsonl",
    yarob_dir: Path | str = "data/yarob_src",
    out_path: Path | str = "data/marker_pairs.jsonl",
) -> Tuple[Path, Dict[str, int]]:
    """Build the unified marker training set; return (path, stats)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs: List[MarkerTrainingPair] = []
    pairs.extend(build_marker_pairs_from_distilled(distilled_path))
    pairs.extend(build_marker_pairs_from_yarob(yarob_dir))

    stats = {"total": len(pairs)}
    from collections import Counter
    by_kind = Counter(p.marker_kind for p in pairs)
    by_source = Counter(p.source for p in pairs)
    stats["by_kind"] = dict(by_kind)
    stats["by_source"] = dict(by_source)
    stats["with_marker"] = sum(1 for p in pairs if p.marker_target != "<NO_MARKER>")

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({
                "sentence": p.sentence,
                "word": p.word,
                "case": p.case,
                "role": p.role,
                "marker_target": p.marker_target,
                "marker_kind": p.marker_kind,
                "source": p.source,
            }, ensure_ascii=False) + "\n")

    return out_path, stats


def main():
    """CLI: build the marker training set."""
    import argparse
    p = argparse.ArgumentParser(description="Extract marker phrases for Mix A training")
    p.add_argument("--distilled", default="data/distilled_irab.jsonl")
    p.add_argument("--yarob_dir", default="data/yarob_src")
    p.add_argument("--out", default="data/marker_pairs.jsonl")
    args = p.parse_args()

    out, stats = build_combined_marker_pairs(args.distilled, args.yarob_dir, args.out)
    print(f"✓ wrote {stats['total']} marker pairs to {out}")
    print(f"  with_marker: {stats['with_marker']}  ({stats['with_marker']*100/max(1,stats['total']):.1f}%)")
    print(f"  by_kind:    {stats['by_kind']}")
    print(f"  by_source:  {stats['by_source']}")


if __name__ == "__main__":
    main()
