"""Stanza Arabic UD pipeline → templated traditional i'rāb prose.

Published-prior-work baseline. Uses Stanza's Arabic Universal Dependencies
parser (Qaf model trained on PADT-UD) and maps each predicted (UPOS, Case,
deprel) tuple to a single short Arabic i'rāb string.

The mapping is intentionally minimal (~25 rules + a generic fallback). The
goal is NOT to beat Sonnet RAG; it is to give the paper an external, fully-
deterministic, no-LLM baseline whose strengths and limits are obvious from
the code.

Usage:
    from .stanza_baseline import StanzaBaseline
    sb = StanzaBaseline()
    items = sb.predict("ذهب الولد إلى المدرسة")
    # → [{"word": "ذهب", "irab": "فعل ماض مبني على الفتح"}, ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


NO_PARSE = "<NO_PARSE>"


@dataclass
class StanzaWordIrab:
    word: str
    irab: str
    pos: Optional[str] = None
    case: Optional[str] = None
    role: Optional[str] = None
    marker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"word": self.word, "irab": self.irab,
                "pos": self.pos, "case": self.case,
                "role": self.role, "marker": self.marker}


# ---------------------------------------------------------------------------
# UD → traditional Arabic i'rāb mapping
# ---------------------------------------------------------------------------
def _verb_irab(feats: Dict[str, str]) -> str:
    aspect = feats.get("Aspect", "")
    mood = feats.get("Mood", "")
    if aspect == "Imp" or mood == "Imp":
        return "فعل أمر مبني على السكون"
    if aspect == "Perf":
        return "فعل ماض مبني على الفتح"
    if aspect == "Imp" == False or "Imp" in feats.get("Tense", ""):
        pass
    if mood == "Sub":
        return "فعل مضارع منصوب وعلامة نصبه الفتحة الظاهرة"
    if mood == "Jus":
        return "فعل مضارع مجزوم وعلامة جزمه السكون"
    if mood in {"Ind", ""} and aspect not in {"Perf"}:
        return "فعل مضارع مرفوع وعلامة رفعه الضمة الظاهرة"
    return "فعل مضارع مرفوع وعلامة رفعه الضمة الظاهرة"


def _noun_irab(feats: Dict[str, str], deprel: str, prev_upos: str) -> str:
    case = feats.get("Case", "")
    if case == "Nom":
        if deprel in {"nsubj", "nsubj:pass"}:
            return "فاعل مرفوع وعلامة رفعه الضمة الظاهرة"
        if deprel == "root":
            return "مبتدأ مرفوع وعلامة رفعه الضمة الظاهرة"
        return "اسم مرفوع وعلامة رفعه الضمة الظاهرة"
    if case == "Acc":
        if deprel in {"obj", "iobj"}:
            return "مفعول به منصوب وعلامة نصبه الفتحة الظاهرة"
        if deprel == "obl":
            return "ظرف منصوب وعلامة نصبه الفتحة الظاهرة"
        return "اسم منصوب وعلامة نصبه الفتحة الظاهرة"
    if case == "Gen":
        if prev_upos == "ADP":
            return "اسم مجرور وعلامة جره الكسرة الظاهرة"
        if deprel in {"nmod", "nmod:poss"}:
            return "مضاف إليه مجرور وعلامة جره الكسرة الظاهرة"
        return "اسم مجرور وعلامة جره الكسرة الظاهرة"
    return "اسم"


def _pron_irab(feats: Dict[str, str], deprel: str) -> str:
    case = feats.get("Case", "")
    if case == "Nom" or deprel in {"nsubj", "nsubj:pass"}:
        return "ضمير مبني في محل رفع فاعل"
    if case == "Acc" or deprel in {"obj", "iobj"}:
        return "ضمير مبني في محل نصب مفعول به"
    if case == "Gen":
        return "ضمير مبني في محل جر"
    return "ضمير مبني"


def ud_to_irab(upos: str, feats: Dict[str, str], deprel: str, prev_upos: str = "") -> str:
    if upos == "VERB" or upos == "AUX":
        return _verb_irab(feats)
    if upos == "NOUN" or upos == "PROPN":
        return _noun_irab(feats, deprel, prev_upos)
    if upos == "PRON":
        return _pron_irab(feats, deprel)
    if upos == "ADJ":
        case = feats.get("Case", "")
        if case == "Nom":
            return "نعت مرفوع وعلامة رفعه الضمة الظاهرة"
        if case == "Acc":
            return "نعت منصوب وعلامة نصبه الفتحة الظاهرة"
        if case == "Gen":
            return "نعت مجرور وعلامة جره الكسرة الظاهرة"
        return "صفة"
    if upos == "ADP":
        return "حرف جر مبني على الكسر"
    if upos == "DET":
        return "أداة تعريف"
    if upos == "CCONJ":
        return "حرف عطف مبني"
    if upos == "SCONJ":
        return "حرف مصدري ونصب مبني"
    if upos == "PART":
        return "حرف مبني"
    if upos == "NUM":
        case = feats.get("Case", "")
        if case == "Nom":
            return "اسم مرفوع وعلامة رفعه الضمة الظاهرة"
        if case == "Acc":
            return "اسم منصوب وعلامة نصبه الفتحة الظاهرة"
        if case == "Gen":
            return "اسم مجرور وعلامة جره الكسرة الظاهرة"
        return "اسم"
    if upos == "ADV":
        return "ظرف منصوب وعلامة نصبه الفتحة الظاهرة"
    return NO_PARSE


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------
class StanzaBaseline:
    def __init__(self, lang: str = "ar"):
        import stanza
        self.nlp = stanza.Pipeline(
            lang=lang,
            processors="tokenize,mwt,pos,lemma,depparse",
            verbose=False,
            use_gpu=False,
        )

    def predict(self, sentence: str) -> List[StanzaWordIrab]:
        if not sentence or not sentence.strip():
            return []
        try:
            doc = self.nlp(sentence)
        except Exception as e:
            return [StanzaWordIrab(word=sentence, irab=f"{NO_PARSE} (pipeline error: {e})")]
        out: List[StanzaWordIrab] = []
        for sent in doc.sentences:
            prev_upos = ""
            for w in sent.words:
                feats = _parse_feats(w.feats or "")
                irab = ud_to_irab(w.upos or "", feats, w.deprel or "", prev_upos)
                out.append(StanzaWordIrab(
                    word=w.text or "",
                    irab=irab,
                    pos=w.upos,
                    case=feats.get("Case"),
                    role=w.deprel,
                    marker=None,
                ))
                prev_upos = w.upos or ""
        return out


def _parse_feats(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for kv in s.split("|"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out
