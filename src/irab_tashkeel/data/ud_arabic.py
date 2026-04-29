"""Universal Dependencies Arabic treebanks (PADT, NYUAD).

Both treebanks are CoNLL-U; we template UD `deprel` + morphology features
into traditional Arabic i'rab strings via the same mapping family used in
qac.py. Output: per-sentence MTLExample with detailed `irab_targets`.

Data sources (free, GitHub):
  - UD_Arabic-PADT  ~7.6k MSA news sentences,  CC-BY-NC-SA-3.0
  - UD_Arabic-NYUAD ~19.7k MSA news sentences, CC-BY-SA-4.0 (annotation only,
    surface forms are in Vform feature so we can use it standalone)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models.labels import IRAB_TO_ID
from ..models.tokenizer import (
    compute_word_offsets, strip_diacritics, text_to_diac_labels,
)
from .schema import MTLExample


PADT_REPO = "https://github.com/UniversalDependencies/UD_Arabic-PADT.git"
NYUAD_REPO = "https://github.com/UniversalDependencies/UD_Arabic-NYUAD.git"


# ---------------------------------------------------------------------------
# UD relation + morphology → Arabic i'rab string
# ---------------------------------------------------------------------------
_CASE_MARFU = "مرفوع وعلامة رفعه الضمة الظاهرة"
_CASE_MANSUB = "منصوب وعلامة نصبه الفتحة الظاهرة"
_CASE_MAJRUR = "مجرور وعلامة جره الكسرة الظاهرة"
_CASE_MAJZUM = "مجزوم وعلامة جزمه السكون"

_CASE_TO_AR = {"Nom": "رفع", "Acc": "نصب", "Gen": "جر"}


def _parse_features(feat_str: str) -> Dict[str, str]:
    if feat_str in ("_", ""):
        return {}
    return dict(kv.split("=", 1) for kv in feat_str.split("|") if "=" in kv)


def _verb_irab(feats: Dict[str, str]) -> str:
    aspect = feats.get("Aspect", "")
    mood = feats.get("Mood", "")
    voice = feats.get("Voice", "Act")
    if aspect == "Perf":
        return "فعل ماضٍ مبني على الفتح" + (" للمجهول" if voice == "Pass" else "")
    if aspect == "Imp":
        if mood == "Ind":
            return f"فعل مضارع {_CASE_MARFU}"
        if mood == "Sub":
            return f"فعل مضارع {_CASE_MANSUB}"
        if mood == "Jus":
            return f"فعل مضارع {_CASE_MAJZUM}"
        return f"فعل مضارع {_CASE_MARFU}"
    if aspect == "Imp" or feats.get("VerbForm") == "Imp":
        return "فعل أمر مبني على السكون"
    return "فعل"


def _noun_irab(deprel: str, case: str, head_pos: Optional[str], head_has_prep: bool) -> str:
    """Pick a role label based on UD deprel, then attach the case template.

    Role labels are pure (no case word); the case template appends the
    rafʿ/naṣb/jarr clause with its sign. A `qualifier` adds optional
    "بحرف الجر" / "(مضاف إليه)" annotations when relevant.
    """
    base_deprel = deprel.split(":")[0]

    role_label = "اسم"
    qualifier = ""
    if base_deprel == "nsubj":
        role_label = "فاعل"
    elif base_deprel == "nsubj:pass":
        role_label = "نائب فاعل"
    elif base_deprel in {"obj", "iobj"}:
        role_label = "مفعول به"
    elif base_deprel == "obl":
        if case == "Acc":
            role_label = "مفعول فيه ظرف"
        elif case == "Gen" and head_has_prep:
            role_label = "اسم"
            qualifier = "بحرف الجر"
    elif base_deprel == "nmod":
        if case == "Gen":
            if head_has_prep:
                role_label = "اسم"
                qualifier = "بحرف الجر"
            else:
                role_label = "مضاف إليه"
        elif case == "Nom":
            role_label = "خبر"
        elif case == "Acc":
            role_label = "حال"
    elif base_deprel == "appos":
        role_label = "بدل"
    elif base_deprel == "amod":
        role_label = "نعت"
    elif base_deprel == "xcomp":
        role_label = "حال" if case == "Acc" else "خبر"
    elif base_deprel == "ccomp":
        role_label = "خبر جملة"
    elif base_deprel == "advmod":
        role_label = "ظرف"
    elif base_deprel == "vocative":
        role_label = "منادى"
    elif base_deprel == "discourse":
        role_label = "حرف"
    elif base_deprel == "root":
        role_label = "مبتدأ" if case == "Nom" else "اسم"

    case_template = {
        "Nom": _CASE_MARFU,
        "Acc": _CASE_MANSUB,
        "Gen": _CASE_MAJRUR,
    }.get(case, "")

    if case_template and qualifier:
        return f"{role_label} {case_template} ({qualifier})"
    if case_template:
        return f"{role_label} {case_template}"
    return role_label


def _ud_token_to_irab(
    upos: str, deprel: str, feats: Dict[str, str],
    head_pos: Optional[str], head_has_prep: bool,
) -> str:
    base_deprel = deprel.split(":")[0]

    if upos == "VERB" or upos == "AUX":
        return _verb_irab(feats)
    if upos == "ADP":
        return "حرف جر مبني لا محل له من الإعراب"
    if upos == "CCONJ":
        return "حرف عطف مبني لا محل له من الإعراب"
    if upos == "SCONJ":
        return "حرف مصدري مبني لا محل له من الإعراب"
    if upos == "PART":
        return "حرف مبني لا محل له من الإعراب"
    if upos == "DET":
        return "أداة تعريف مبنية على السكون لا محل لها من الإعراب"
    if upos == "PRON":
        case = feats.get("Case", "")
        mahall = _CASE_TO_AR.get(case, "رفع")
        return f"ضمير مبني في محل {mahall}"
    if upos == "PROPN":
        case = feats.get("Case", "")
        ir = _noun_irab(base_deprel, case, head_pos, head_has_prep)
        return ir.replace("اسم", "اسم علم", 1) if ir.startswith("اسم") else ir
    if upos in {"NOUN", "ADJ", "NUM"}:
        case = feats.get("Case", "")
        ir = _noun_irab(base_deprel, case, head_pos, head_has_prep)
        if upos == "ADJ" and ir.startswith("اسم"):
            ir = ir.replace("اسم", "صفة", 1)
        return ir
    if upos == "INTJ":
        return "اسم فعل أمر مبني"
    return "كلمة"  # fallback (PUNCT, X, SYM are filtered out before)


# ---------------------------------------------------------------------------
# CoNLL-U parser
# ---------------------------------------------------------------------------
def _parse_conllu_sentence(lines: List[str]) -> Optional[Tuple[str, List[Dict]]]:
    """Parse one CoNLL-U sentence block.

    Returns (sentence_text, tokens) or None if the block has no tokens.
    Skips MWT range lines (id with a hyphen) — we use the atomic components.
    Skips PUNCT and tokens whose form is missing.
    """
    text = ""
    tokens: List[Dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("# text ="):
            text = line.split("=", 1)[1].strip()
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 8:
            continue
        tid, form, lemma, upos, xpos, feat_str, head, deprel = cols[:8]
        if "-" in tid or "." in tid:
            continue  # MWT range or empty node
        if upos == "PUNCT":
            continue
        feats = _parse_features(feat_str)
        misc_str = cols[9] if len(cols) > 9 else "_"
        misc = _parse_features(misc_str.replace("|", "|"))
        diacritized = misc.get("Vform", form)
        try:
            head_id = int(head)
        except ValueError:
            head_id = 0
        tokens.append({
            "id": int(tid),
            "form": form,
            "diacritized": diacritized,
            "upos": upos,
            "xpos": xpos,
            "feats": feats,
            "head": head_id,
            "deprel": deprel,
        })
    if not tokens:
        return None
    return text, tokens


def _parse_conllu_file(path: Path) -> List[Tuple[str, List[Dict]]]:
    """Yield (text, tokens) for every sentence in a CoNLL-U file."""
    out: List[Tuple[str, List[Dict]]] = []
    buf: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if buf:
                    parsed = _parse_conllu_sentence(buf)
                    if parsed is not None:
                        out.append(parsed)
                    buf = []
            else:
                buf.append(line)
    if buf:
        parsed = _parse_conllu_sentence(buf)
        if parsed is not None:
            out.append(parsed)
    return out


# ---------------------------------------------------------------------------
# Conversion to MTLExample
# ---------------------------------------------------------------------------
def _tokens_to_example(text: str, tokens: List[Dict], source: str, sent_id: str) -> Optional[MTLExample]:
    if len(tokens) < 2 or len(tokens) > 60:
        return None

    # Build {id: token} for head lookup.
    by_id = {t["id"]: t for t in tokens}

    # Pre-compute, for each token, whether its head has a `case` (preposition)
    # child distinct from itself — i.e., whether this token sits behind a prep.
    head_has_prep: Dict[int, bool] = {tid: False for tid in by_id}
    for t in tokens:
        if t["deprel"].split(":")[0] == "case":
            parent = t["head"]
            if parent in head_has_prep:
                head_has_prep[parent] = True

    # Reconstruct diacritized + bare text from token diacritized forms.
    diac_text = " ".join(t["diacritized"] for t in tokens)
    try:
        bare_text, diac_ids = text_to_diac_labels(diac_text)
    except Exception:
        return None
    bare_text = bare_text.strip()
    if not bare_text:
        return None
    bare_words = bare_text.split()
    if len(bare_words) != len(tokens):
        # Shouldn't happen for atomic tokens but guard anyway.
        return None

    irab_targets: List[str] = []
    irab_ids: List[int] = []
    for t in tokens:
        head_token = by_id.get(t["head"])
        head_pos = head_token["upos"] if head_token else None
        ir = _ud_token_to_irab(
            t["upos"], t["deprel"], t["feats"],
            head_pos, head_has_prep.get(t["id"], False),
        )
        irab_targets.append(ir)
        # Coarse class — pick a sensible bucket.
        if t["upos"] in {"VERB", "AUX"}:
            coarse = "fiil"
        elif t["upos"] == "ADP":
            coarse = "harf_jarr"
        elif t["upos"] in {"CCONJ", "SCONJ"}:
            coarse = "harf_atf"
        elif t["upos"] == "PART":
            coarse = "harf_nafy"
        elif t["upos"] == "PRON":
            coarse = "mabni_noun"
        else:
            case = t["feats"].get("Case", "")
            if case == "Nom":
                coarse = "N_marfu"
            elif case == "Acc":
                coarse = "N_mansub"
            elif case == "Gen":
                coarse = "ism_majrur" if head_has_prep.get(t["id"], False) else "mudaf_ilayh"
            else:
                coarse = "other"
        irab_ids.append(IRAB_TO_ID.get(coarse, IRAB_TO_ID["other"]))

    word_offsets = compute_word_offsets(bare_text)

    return MTLExample(
        bare_text=bare_text,
        diac_labels=diac_ids,
        mask_diac=True,
        word_offsets=word_offsets,
        irab_labels=irab_ids,
        mask_irab=True,
        err_labels=[0] * len(bare_text),
        mask_err=False,
        source=source,
        sent_id=sent_id,
        irab_targets=irab_targets,
    )


def _git_clone(repo_url: str, target_dir: Path) -> bool:
    target_dir = Path(target_dir)
    if target_dir.exists() and any(target_dir.glob("*.conllu")):
        return True
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(target_dir)],
            check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠ clone failed for {repo_url}: {e.stderr.decode().strip() if e.stderr else e}")
        return False


def _load_treebank(repo_dir: Path, source_tag: str, max_sentences: Optional[int]) -> List[MTLExample]:
    examples: List[MTLExample] = []
    for path in sorted(repo_dir.glob("*.conllu")):
        for text, tokens in _parse_conllu_file(path):
            ex = _tokens_to_example(text, tokens, source_tag, sent_id=path.stem)
            if ex is None:
                continue
            examples.append(ex)
            if max_sentences and len(examples) >= max_sentences:
                return examples
    return examples


def load_padt_examples(
    repo_dir: Path | str = "data/ud_padt",
    download_if_missing: bool = True,
    max_sentences: Optional[int] = None,
) -> List[MTLExample]:
    repo_dir = Path(repo_dir)
    if download_if_missing and not _git_clone(PADT_REPO, repo_dir):
        return []
    return _load_treebank(repo_dir, source_tag="ud_padt", max_sentences=max_sentences)


def load_nyuad_examples(
    repo_dir: Path | str = "data/ud_nyuad",
    download_if_missing: bool = True,
    max_sentences: Optional[int] = None,
) -> List[MTLExample]:
    """NYUAD ships annotation only — surface forms come from PATB (LDC-licensed).

    Without LDC access the form column is all underscores and unusable for
    training. To use NYUAD: place the PATB-merged conllu files in repo_dir
    (with real `form` and `Vform=` filled in), then call this loader.
    """
    repo_dir = Path(repo_dir)
    if download_if_missing and not _git_clone(NYUAD_REPO, repo_dir):
        return []
    examples = _load_treebank(repo_dir, source_tag="ud_nyuad", max_sentences=max_sentences)
    if examples:
        return examples
    print("⚠ NYUAD: no usable sentences — surface forms appear to be masked. "
          "Merge with PATB text to use this treebank. Skipping.")
    return []
