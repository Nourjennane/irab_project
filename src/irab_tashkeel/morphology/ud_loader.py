"""UD Arabic-PADT CoNLL-U loader with multi-word-token collapsing.

The CoNLL-U format used by UD-PADT looks like:

    # sent_id = afp.20000715.0075:p1u1
    # text    = برلين ترفض حصول ...
    1   برلين   بَرلِين   X    X---------    Foreign=Yes      ...
    2   ترفض    رَفَض    VERB  VIIA-3FS--    Aspect=Imp|Gender=Fem|...
    3-4 وفي     _        _    _              _                ...
    3   و       وَ       CCONJ C---------    _                ...
    4   في      فِي      ADP   P---------    AdpType=Prep     ...
    ...

------------------------------------------------------------------
Multi-word-token (MWT) collapsing policy (Phase 1, frozen):

UD encodes Arabic clitics like فـ + الـ + كتاب as a multi-word token
``3-5 فالكتاب`` followed by three segment lines ``3 ف``, ``4 ال``, ``5 كتاب``.
distill_v2 keeps the surface word as one token. To stay tokenization-
compatible with rev 2's existing pipeline, we ALWAYS collapse MWT back to
their surface form.

Concretely: when we see a line ``a-b SURFACE _ _ _ _ ...``, we emit ONE
WordMorph for SURFACE and absorb the morphology of all segment lines
``a..b`` (a..b INCLUSIVE) into the surface word. The absorption rule for
each feature:

- POS:       UPOS of the head segment (the *last* numbered segment, which
             in UD-PADT is the content word, e.g. ``5 كتاب`` for ``3-5 فالكتاب``).
- Gender:    head segment's value, fall back to first non-und along the segments.
- Number:    head segment's value.
- Definite:  head segment's value.
- Person/Aspect/Mood/Voice: head segment's value (verb features only on verbs).

The choice of "head = last segment" is based on Arabic grammar: clitics
attach to the left of a content word, so the rightmost segment is the
content word and carries the syntactic features.

This collapsing is logged in ``WordMorph.source_id_range`` so the smoke
test can verify alignment row-for-row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .schema import (
    UPOS_TO_CANONICAL_POS,
    canonicalize_morph_feature,
    parse_feats,
)
from .word_morph import SentenceMorph, WordMorph


# Range like "3-5"
_MWT_RANGE = re.compile(r"^(\d+)-(\d+)$")


@dataclass
class _RawTokenLine:
    id_str: str
    form: str
    lemma: str
    upos: str
    xpos: str
    feats_raw: str
    head: str
    deprel: str
    deps: str
    misc: str


def _parse_token_line(line: str) -> _RawTokenLine:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 10:
        raise ValueError(f"Bad CoNLL-U line ({len(parts)} cols): {line!r}")
    return _RawTokenLine(*parts)


def _segment_to_word_morph(t: _RawTokenLine, source_id_range: str) -> WordMorph:
    feats = parse_feats(t.feats_raw)
    return WordMorph(
        word=t.form,
        upos=t.upos,
        pos=UPOS_TO_CANONICAL_POS.get(t.upos, "noun"),
        gender=canonicalize_morph_feature("gender", feats),
        number=canonicalize_morph_feature("number", feats),
        definite=canonicalize_morph_feature("definite", feats),
        person=canonicalize_morph_feature("person", feats),
        aspect=canonicalize_morph_feature("aspect", feats),
        mood=canonicalize_morph_feature("mood", feats),
        voice=canonicalize_morph_feature("voice", feats),
        source_id_range=source_id_range,
    )


def _collapse_mwt_block(
    surface_token: _RawTokenLine,
    segments: List[_RawTokenLine],
) -> WordMorph:
    """Collapse a UD multi-word-token block to a single WordMorph.

    Args:
        surface_token: the ``a-b`` line carrying the surface form.
        segments: the ``a..b`` numbered lines giving per-segment FEATS.

    The output WordMorph's surface form is ``surface_token.form``; its
    morphology is taken from the *last* segment (the content word for
    Arabic clitic chains), with non-und values from earlier segments used
    as fallbacks.
    """
    a, b = _MWT_RANGE.match(surface_token.id_str).groups()  # type: ignore[arg-type]
    if not segments:
        # Defensive: shouldn't happen on well-formed UD-PADT
        return WordMorph(word=surface_token.form, source_id_range=f"{a}-{b}")

    # Head = last segment.
    head = segments[-1]
    head_morph = _segment_to_word_morph(head, source_id_range=f"{a}-{b}")
    # Override the surface form to match the MWT line.
    head_morph.word = surface_token.form
    head_morph.source_id_range = f"{a}-{b}"

    # Fall back to earlier segments for any "und" values.  This is
    # conservative (rare in PADT) but covers e.g. a clitic that carries
    # Person but the head doesn't.
    for seg in segments[:-1]:
        seg_morph = _segment_to_word_morph(seg, source_id_range=str(seg.id_str))
        for attr in ("gender", "number", "definite", "person", "aspect",
                     "mood", "voice"):
            if getattr(head_morph, attr) == "und":
                v = getattr(seg_morph, attr)
                if v != "und":
                    setattr(head_morph, attr, v)
    return head_morph


def parse_conllu(path: str | Path) -> Iterator[SentenceMorph]:
    """Yield ``SentenceMorph`` records for each sentence in a CoNLL-U file.

    Multi-word-tokens are collapsed to their surface form per the policy
    documented in this module's docstring.
    """
    path = Path(path)
    sent_id: Optional[str] = None
    text: Optional[str] = None
    pending_words: List[WordMorph] = []
    pending_mwt: Optional[Tuple[_RawTokenLine, List[_RawTokenLine], int]] = None
    # ^ (mwt_line, segs_so_far, end_id)

    def _flush_sentence():
        nonlocal sent_id, text, pending_words, pending_mwt
        if pending_mwt is not None:
            # Defensive: incomplete MWT at end of sentence; emit head only.
            mwt_line, segs, _end = pending_mwt
            pending_words.append(_collapse_mwt_block(mwt_line, segs))
            pending_mwt = None
        if pending_words:
            yield_sent = SentenceMorph(
                sentence=text or " ".join(w.word for w in pending_words),
                items=pending_words,
                sent_id=sent_id,
            )
            pending_words = []
            sent_id = None
            text = None
            return yield_sent
        return None

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line_strip = line.rstrip("\n")
            if not line_strip:
                # End of sentence.
                out = _flush_sentence()
                if out is not None:
                    yield out
                continue
            if line_strip.startswith("#"):
                if line_strip.startswith("# sent_id"):
                    sent_id = line_strip.split("=", 1)[1].strip()
                elif line_strip.startswith("# text"):
                    text = line_strip.split("=", 1)[1].strip()
                continue
            # Token line.
            t = _parse_token_line(line_strip)
            m = _MWT_RANGE.match(t.id_str)
            if m:
                # Start of an MWT block. Open a buffer.
                end_id = int(m.group(2))
                pending_mwt = (t, [], end_id)
                continue
            if "." in t.id_str:
                # Empty/elliptical node (e.g. "5.1") — skip; rare in PADT.
                continue
            # Numbered segment.
            tok_id = int(t.id_str)
            if pending_mwt is not None:
                mwt_line, segs, end_id = pending_mwt
                segs.append(t)
                if tok_id == end_id:
                    pending_words.append(_collapse_mwt_block(mwt_line, segs))
                    pending_mwt = None
                continue
            # Plain token (non-MWT).
            pending_words.append(_segment_to_word_morph(t, source_id_range=str(tok_id)))

    # Flush the last sentence if no trailing blank line.
    out = _flush_sentence()
    if out is not None:
        yield out
