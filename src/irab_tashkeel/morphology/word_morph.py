"""Per-word morphology dataclass + sentence container.

These are torch-free (so data tooling and tests can import them on a CPU-
only machine without dragging in transformers/torch). They parallel
:class:`irab_tashkeel.structured.word_irab.WordIrab` exactly so the merged
dataset can carry both label sets per word.

Each :class:`WordMorph` corresponds to ONE surface word — multi-word-tokens
in UD-PADT have already been collapsed back to their surface form by the
loader before a WordMorph is constructed (see ``ud_loader.py``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WordMorph:
    """One word's morphology features + UPOS + canonical 6-class POS."""

    word: str
    upos: Optional[str] = None       # raw UD UPOS (NOUN/VERB/...)
    pos: Optional[str] = None        # canonical 6-class (matches rev 2 POS_LABELS)
    gender: Optional[str] = None     # canonical (m/f/und)
    number: Optional[str] = None     # canonical (sg/dual/pl/und)
    definite: Optional[str] = None   # canonical (def/indef/cons/und)
    person: Optional[str] = None     # canonical (1/2/3/und)
    aspect: Optional[str] = None     # canonical (imp/perf/und)
    mood: Optional[str] = None       # canonical (ind/imp_mood/sub/jus/und)
    voice: Optional[str] = None      # canonical (act/pass/und)

    # Phase 1 rev: not used for training, but useful for sanity-checking
    # alignment (we save the original CoNLL-U id range that produced this
    # surface word — covers MWT collapsing).
    source_id_range: Optional[str] = None    # e.g. "3" or "1-2" or "5-7"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WordMorph":
        return cls(
            word=d["word"], upos=d.get("upos"), pos=d.get("pos"),
            gender=d.get("gender"), number=d.get("number"),
            definite=d.get("definite"), person=d.get("person"),
            aspect=d.get("aspect"), mood=d.get("mood"), voice=d.get("voice"),
            source_id_range=d.get("source_id_range"),
        )


@dataclass
class SentenceMorph:
    """A sentence + its per-word morphology."""

    sentence: str
    items: List[WordMorph] = field(default_factory=list)
    sent_id: Optional[str] = None
    source: str = "UD-PADT"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"sentence": self.sentence,
                             "items": [w.to_dict() for w in self.items]}
        if self.sent_id is not None:
            d["sent_id"] = self.sent_id
        if self.source != "UD-PADT":
            d["source"] = self.source
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SentenceMorph":
        return cls(
            sentence=d["sentence"],
            items=[WordMorph.from_dict(it) for it in d.get("items", [])],
            sent_id=d.get("sent_id"),
            source=d.get("source", "UD-PADT"),
        )

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json_line(cls, line: str) -> "SentenceMorph":
        return cls.from_dict(json.loads(line))
