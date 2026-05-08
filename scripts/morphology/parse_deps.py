"""Phase 3 — Stanza UD parse pipeline.

Reads ``data/morph_v1/{train,val}.jsonl`` (the merged UD-PADT + distill_v2
corpus from Phase 1) and adds per-word dep features
(``deprel``, ``head_idx``, ``governor_upos``) to each item in records
where ``source == "distill_v2"`` (i.e. iʿrāb-supervised records that
need Stanza-parsed deps; UD-PADT records have ``has_morph=True`` and we
leave them dep-less for the first pass — they don't carry iʿrāb gradient
through the dep features anyway).

Output: ``data/morph_v1_dep/{train,val}.jsonl`` — same structure as input
plus per-item dep fields and a record-level ``has_dep`` flag.

Stanza pipeline cost: ~30 min for 7K sentences on a CPU node, ~10 min
with GPU. We run it offline once and cache the augmented corpus.

Usage:
    python scripts/morphology/parse_deps.py \\
        --in_train  data/morph_v1/train.jsonl \\
        --in_val    data/morph_v1/val.jsonl \\
        --out_dir   data/morph_v1_dep/

Alignment policy:
    Stanza tokenises differently from the whitespace tokenisation used
    by distill_v2. We map by surface match: for each whitespace-split
    word in the input sentence, find the Stanza-token whose ``text``
    matches (after light Arabic normalisation). When Stanza splits a
    distill_v2 word into multiple tokens (clitic chains), the LAST
    Stanza token's dep info is kept (matches the UD-PADT MWT collapse
    policy used in Phase 1).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_DIACRITICS = re.compile(r"[ً-ٰٟ]")
_NON_AR = re.compile(r"[^؀-ۿ]+")


def _norm_ar(s: str) -> str:
    """Strip diacritics + non-Arabic characters for surface matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = _DIACRITICS.sub("", s)
    s = _NON_AR.sub("", s)
    return s


def _align_stanza_to_words(
    words: List[str], stanza_tokens: List[Dict],
) -> List[Optional[Dict]]:
    """For each whitespace word, find the Stanza token (or last of a chain)
    that surface-matches it. Returns a per-word list of Stanza token dicts
    (or None if no match found).

    Stanza tokens are dicts with keys ``text``, ``deprel``, ``head_idx``,
    ``upos`` (and ``head_upos`` once we resolve it).
    """
    out: List[Optional[Dict]] = [None] * len(words)
    si = 0
    for wi, w in enumerate(words):
        target = _norm_ar(w)
        if not target:
            continue
        # Try to greedily match Stanza tokens whose concatenated normalised
        # text equals the target. The MWT-collapse policy keeps the last
        # token's dep info.
        acc = ""
        last = None
        scan = si
        while scan < len(stanza_tokens) and len(acc) < len(target):
            tok = stanza_tokens[scan]
            tok_norm = _norm_ar(tok["text"])
            if not tok_norm:
                scan += 1
                continue
            acc += tok_norm
            last = tok
            scan += 1
            if acc == target:
                break
        if last is not None and acc == target:
            out[wi] = last
            si = scan
        else:
            # No match found — leave as None (will become has_dep=False on
            # this record OR a per-word <unk> filler depending on policy).
            # Keep si advancing so we don't loop forever.
            si += 1
    return out


def _process_record(rec: Dict, nlp) -> Dict:
    """Run Stanza on a record's sentence, attach dep info to each item.

    Sets ``rec["has_dep"] = True`` if all (or "enough") items received
    dep features; otherwise leaves the original record unchanged.
    """
    sentence = rec.get("sentence", "")
    if not sentence:
        rec["has_dep"] = False
        return rec
    try:
        doc = nlp(sentence)
    except Exception:
        rec["has_dep"] = False
        return rec

    # Flatten Stanza output to a per-token list with linear indices.
    stanza_tokens: List[Dict] = []
    word_to_global: Dict[Tuple[int, int], int] = {}  # (sent_idx, word_idx) -> global idx
    for sent_idx, sent in enumerate(doc.sentences):
        for w in sent.words:
            global_idx = len(stanza_tokens)
            word_to_global[(sent_idx, w.id)] = global_idx
            stanza_tokens.append({
                "text": w.text or "",
                "deprel": w.deprel or "",
                "head_idx_local": int(w.head) if w.head is not None else 0,
                "upos": w.upos or "",
                "_sent_idx": sent_idx,
                "_word_id": w.id,
            })

    # Resolve governor UPOS: per word, look at its head index within the same sentence.
    for tok in stanza_tokens:
        si = tok["_sent_idx"]
        head_local = tok["head_idx_local"]
        if head_local == 0:
            tok["governor_upos"] = ""
        else:
            global_head = word_to_global.get((si, head_local))
            if global_head is not None and global_head < len(stanza_tokens):
                tok["governor_upos"] = stanza_tokens[global_head]["upos"]
            else:
                tok["governor_upos"] = ""

    # Align Stanza tokens to whitespace words.
    words = [it.get("word", "") for it in rec.get("items", [])]
    aligned = _align_stanza_to_words(words, stanza_tokens)

    n_matched = sum(1 for a in aligned if a is not None)
    if n_matched == 0:
        rec["has_dep"] = False
        return rec

    # Build per-word dep info, with global head index across the sentence
    for it, tok in zip(rec.get("items", []), aligned):
        if tok is None:
            it["deprel"] = "<unk>"
            it["head_idx"] = 0
            it["governor_upos"] = ""
            continue
        # Global 1-based head: align Stanza word.head (sentence-local) to
        # the global flat token index, then convert to 1-based UD convention.
        si = tok["_sent_idx"]
        head_local = tok["head_idx_local"]
        if head_local == 0:
            head_global_1b = 0
        else:
            ghead = word_to_global.get((si, head_local))
            head_global_1b = (ghead + 1) if ghead is not None else 0
        it["deprel"] = tok["deprel"] or "<unk>"
        it["head_idx"] = head_global_1b
        it["governor_upos"] = tok.get("governor_upos", "") or ""

    rec["has_dep"] = (n_matched / len(words)) >= 0.5  # at least half aligned
    rec["_dep_match_rate"] = n_matched / len(words) if words else 0.0
    return rec


def _process_file(in_path: Path, out_path: Path, nlp, *, source_filter: str = "distill_v2") -> Dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_dep = 0
    n_skipped_source = 0
    n_low_match = 0
    with in_path.open() as fh, out_path.open("w") as out_fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_total += 1
            if source_filter and rec.get("source") != source_filter:
                # Pass through UD-PADT records unchanged
                rec.setdefault("has_dep", False)
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_skipped_source += 1
                continue
            rec = _process_record(rec, nlp)
            if rec.get("has_dep"):
                n_dep += 1
            else:
                n_low_match += 1
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_total % 200 == 0:
                print(f"  ... processed {n_total} records ({n_dep} dep-enriched)")
    return {
        "in": str(in_path), "out": str(out_path),
        "n_total": n_total, "n_dep_enriched": n_dep,
        "n_passthrough": n_skipped_source, "n_low_match": n_low_match,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_train", required=True)
    ap.add_argument("--in_val", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lang", default="ar")
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    print(f"[parse_deps] loading Stanza pipeline (lang={args.lang}, gpu={args.use_gpu})...")
    import stanza
    # download_method=None skips the network check for resources.json updates
    # (compute node has no internet; we pre-downloaded models on login node).
    nlp = stanza.Pipeline(
        lang=args.lang,
        processors="tokenize,mwt,pos,lemma,depparse",
        verbose=False,
        use_gpu=args.use_gpu,
        download_method=None,
    )

    out_dir = Path(args.out_dir)
    summaries = []
    for split, in_path in [("train", args.in_train), ("val", args.in_val)]:
        print(f"[parse_deps] processing {split} from {in_path}...")
        out_path = out_dir / f"{split}.jsonl"
        s = _process_file(Path(in_path), out_path, nlp)
        summaries.append({**s, "split": split})
        print(f"  done: {s}")

    summary_path = out_dir / "parse_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"[parse_deps] summary written to {summary_path}")


if __name__ == "__main__":
    main()
