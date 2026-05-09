"""UD Arabic-PADT → schema_v2 loader (gold treebank).

Ingests the Universal Dependencies Arabic-PADT corpus into
schema_v2. Annotation quality is GOLD_TREEBANK; provenance is
``parser_origin="ud_padt_gold"``.

UD-PADT carries:

- gold dependency heads + DEPREL
- gold morphology (Gender, Number, Person, Definite, Mood, Voice,
  Aspect, Case, VerbForm)
- gold UPOS

UD-PADT does NOT carry traditional Arabic iʿrāb (case / role /
marker in our taxonomy). Layer C fields stay :class:`LabelTag`
unset (``value=None``) for UD-PADT records — every consumer must
handle this explicitly via the per-token ``LabelTag.is_present``
check or via the sentence-level
``AnnotationCompleteness.has_role`` flag.

The CoNLL-U format we read:

  ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC

Files: ``data/ud_padt/ar_padt-ud-{train,dev,test}.conllu``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ..normalization import arabic_normalize, normalize_text
from ..schema_v2 import (
    AnnotationCompleteness, AnnotationQuality, Domain, LabelTag,
    Morphology, Sentence, Token,
)
from .base import BaseLoader, register_loader


# UD UPOS → our canonical POS values
_UPOS_TO_CANONICAL = {
    "NOUN": "noun",
    "VERB": "verb",
    "ADJ":  "adjective",
    "ADV":  "adverb",
    "PRON": "pronoun",
    "PROPN": "noun",
    "AUX":  "verb",
    "DET":  "particle",
    "ADP":  "particle",
    "CCONJ": "particle",
    "SCONJ": "particle",
    "PART": "particle",
    "INTJ": "particle",
    "NUM":  "noun",
    "SYM":  "punctuation",
    "PUNCT": "punctuation",
    "X":    "noun",
}


# UD FEATS → our morphology axes
_GENDER_MAP = {"Masc": "masc", "Fem": "fem", "Common": "common"}
_NUMBER_MAP = {"Sing": "sg", "Dual": "dual", "Plur": "plural", "Plural": "plural"}
_PERSON_MAP = {"1": "1", "2": "2", "3": "3"}
_DEFINITE_MAP = {"Def": "definite", "Ind": "indefinite", "Cons": "construct"}
_VOICE_MAP = {"Act": "active", "Pass": "passive"}
_MOOD_MAP = {"Ind": "indicative", "Sub": "subjunctive", "Jus": "jussive",
             "Imp": "imperative"}
_ASPECT_MAP = {"Imp": "imperfective", "Perf": "perfective"}
_CASE_MAP = {"Nom": "raf", "Acc": "nasb", "Gen": "jarr"}


def _parse_feats(feats: str) -> Dict[str, str]:
    if not feats or feats == "_":
        return {}
    out: Dict[str, str] = {}
    for kv in feats.split("|"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def _make_label(value: Optional[str]) -> LabelTag:
    if value:
        return LabelTag(value=value, source="gold_treebank", confidence=1.0)
    return LabelTag()


@register_loader
class UdPadtLoader(BaseLoader):
    """Universal Dependencies Arabic-PADT gold treebank.

    Pass ``split`` ("train" / "dev" / "test") to choose the file.
    Default is ``"train"``.
    """

    source_id          = "ud_padt"
    domain             = Domain.MSA_NEWS.value
    annotation_quality = AnnotationQuality.GOLD_TREEBANK.value
    parser_origin      = "ud_padt_gold"
    license            = "CC BY-NC-SA 3.0"

    def __init__(self, root: str | Path, split: str = "train"):
        super().__init__(root)
        if split not in ("train", "dev", "test"):
            raise ValueError(f"split must be train/dev/test, got {split!r}")
        self.split = split

    def _path(self) -> Path:
        return self.root / "data" / "ud_padt" / f"ar_padt-ud-{self.split}.conllu"

    # -- raw ------------------------------------------------------------------

    def iter_raw(self) -> Iterator[Dict[str, Any]]:
        path = self._path()
        if not path.exists():
            return
        with path.open() as fh:
            current_meta: Dict[str, str] = {}
            current_tokens: List[Dict[str, str]] = []
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("#"):
                    if "=" in line:
                        k, v = line[1:].split("=", 1)
                        current_meta[k.strip()] = v.strip()
                    continue
                if not line:
                    if current_tokens:
                        yield {"meta": dict(current_meta), "tokens": current_tokens}
                        current_meta = {}
                        current_tokens = []
                    continue
                fields = line.split("\t")
                if len(fields) < 10:
                    continue
                token_id = fields[0]
                # Skip multi-word ranges (e.g. 1-2) and empty nodes (e.g. 1.1)
                if "-" in token_id or "." in token_id:
                    continue
                current_tokens.append({
                    "id": token_id, "form": fields[1], "lemma": fields[2],
                    "upos": fields[3], "xpos": fields[4], "feats": fields[5],
                    "head": fields[6], "deprel": fields[7],
                    "deps": fields[8], "misc": fields[9],
                })
            if current_tokens:
                yield {"meta": dict(current_meta), "tokens": current_tokens}

    # -- normalize ------------------------------------------------------------

    def normalize_row(self, raw: Dict[str, Any], idx: int) -> Optional[Sentence]:
        meta = raw["meta"]
        raw_tokens = raw["tokens"]
        if not raw_tokens:
            return None

        sentence_text = meta.get("text", " ".join(t["form"] for t in raw_tokens))
        sent_id = meta.get("sent_id", str(idx))

        tokens: List[Token] = []
        any_morph = False
        any_dep = False
        for i, rt in enumerate(raw_tokens):
            feats = _parse_feats(rt["feats"])
            morph = Morphology(
                gender=_make_label(_GENDER_MAP.get(feats.get("Gender", ""))),
                number=_make_label(_NUMBER_MAP.get(feats.get("Number", ""))),
                person=_make_label(_PERSON_MAP.get(feats.get("Person", ""))),
                definite=_make_label(_DEFINITE_MAP.get(feats.get("Definite", ""))),
                pos=_make_label(_UPOS_TO_CANONICAL.get(rt["upos"])),
                voice=_make_label(_VOICE_MAP.get(feats.get("Voice", ""))),
                mood=_make_label(_MOOD_MAP.get(feats.get("Mood", ""))),
                aspect=_make_label(_ASPECT_MAP.get(feats.get("Aspect", ""))),
            )
            if any(getattr(morph, k).is_present
                   for k in ("gender", "number", "definite", "pos")):
                any_morph = True

            head_str = rt["head"]
            head_idx_0based = -1
            if head_str.isdigit():
                h = int(head_str)
                if h == 0:
                    head_idx_0based = -2     # root marker
                elif h > 0:
                    head_idx_0based = h - 1
                    any_dep = True

            tok = Token(
                index=i,
                surface=rt["form"],
                normalized=arabic_normalize(rt["form"]),
                morph=morph,
                pos=_make_label(_UPOS_TO_CANONICAL.get(rt["upos"])),
                dep_head_idx=head_idx_0based,
                dep_label=_make_label(rt["deprel"] if rt["deprel"] != "_" else None),
                # UD case feature → our case label space (only Nom/Acc/Gen → raf/nasb/jarr)
                case=_make_label(_CASE_MAP.get(feats.get("Case", ""))),
                # UD-PADT does not give iʿrāb role / marker — left unset
            )
            tokens.append(tok)

        comp = AnnotationCompleteness(
            has_morph=any_morph, has_dep=any_dep,
            has_role=False,                 # UD-PADT has no iʿrāb role
            has_marker=False,
            has_constructions=False, has_clauses=False,
            has_reasoning=False, has_graph=False, has_discourse=False,
        )
        n_present = sum(1 for t in tokens for f in (t.case, t.role, t.marker)
                        if f.is_present)
        comp.fields_complete_pct = n_present / max(3 * len(tokens), 1)

        meta_obj = self._make_metadata(source_id_within=sent_id)
        # Override per-layer origin since UD-PADT has gold dep + gold morph
        meta_obj.morph_origin = "ud_padt_gold"
        meta_obj.dep_origin = "ud_padt_gold"
        meta_obj.role_origin = ""
        meta_obj.marker_origin = ""

        return Sentence(
            raw_text=sentence_text,
            normalized_text=normalize_text(sentence_text),
            tokens=tokens,
            metadata=meta_obj,
            completeness=comp,
        )
