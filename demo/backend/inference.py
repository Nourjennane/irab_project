"""Model loading + sentence analysis for the demo backend.

Returns a polished, structured response: per-token predictions with
top-k alternatives + confidence band + warnings; sentence-level
construction detection; a lightweight graph spec (nodes + edges); a
reasoning trace; and a comparison helper. The frontend consumes this
without having to do heuristic post-processing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]


CHECKPOINT_PATHS = {
    "recovery":  ROOT / "runs" / "validated_nextgen_recovery",
    "phase3a":   ROOT / "runs" / "phase3a_491240" / "final",
    "stage7":    ROOT / "runs" / "nextgen" / "stage_7" / "final",
}


SAMPLE_SENTENCES: List[Dict[str, str]] = [
    {"id": "msa1",     "tag": "MSA",        "text": "ذهبَ الطالبُ إلى المدرسةِ"},
    {"id": "msa2",     "tag": "MSA",        "text": "إنَّ الطلابَ مجتهدونَ"},
    {"id": "msa3",     "tag": "MSA",        "text": "كان الجوُّ جميلاً"},
    {"id": "idafa",    "tag": "Idafa",      "text": "كتابُ الطالبِ على الطاولةِ"},
    {"id": "quran1",   "tag": "Quranic",    "text": "بسم الله الرحمن الرحيم"},
    {"id": "quran2",   "tag": "Quranic",    "text": "الحمد لله رب العالمين"},
    {"id": "nest1",    "tag": "Nested",     "text": "إن كان الجوُّ جميلاً ذهبتُ"},
    {"id": "ambig1",   "tag": "Ambiguous",  "text": "في البيتِ رجلٌ"},
]


# Curated hard-case gallery (for the "Hard Cases" tab)
HARD_CASES: List[Dict[str, str]] = [
    {
        "id": "hard_idafa_ambig",
        "tag": "Idafa attachment",
        "text": "كتابُ المعلمِ الجديدُ",
        "why": "Ambiguous attachment: 'الجديد' (new) can describe كتاب (the book) or المعلم (the teacher). Both readings are grammatically valid."
    },
    {
        "id": "hard_nested_idafa",
        "tag": "Nested idafa",
        "text": "بابُ بيتِ الجارِ",
        "why": "Three-level idafa chain. The model collapses to fully = 0.18 on this construction family — see docs/failure_analysis/."
    },
    {
        "id": "hard_prep_vs_idafa",
        "tag": "Preposition vs idafa",
        "text": "أمامَ المسجدِ سيارةٌ",
        "why": "أمامَ can be read as a preposition (mafoul fih) or as the head of an idafa. The downstream noun's case is the same either way."
    },
    {
        "id": "hard_coord_ambig",
        "tag": "Coordination scope",
        "text": "زارَ الطالبَ والمعلمَ في الصفِّ",
        "why": "Did 'في الصف' apply to both 'الطالب' and 'المعلم' or only to 'المعلم'? The coordination scope is unresolved."
    },
    {
        "id": "hard_quranic",
        "tag": "Quranic",
        "text": "وَإِذْ قَالَ رَبُّكَ لِلْمَلَائِكَةِ",
        "why": "Classical Arabic with implicit subjects and discourse-driven structure. Outside MSA distribution."
    },
    {
        "id": "hard_omitted",
        "tag": "Omitted governor",
        "text": "في البيتِ",
        "why": "Headless prepositional phrase — the governing verb is omitted but assumed by context. Hard to resolve from one-sentence input."
    },
]


# Roles in the surface-ambiguous family — when a model prediction belongs
# here, the analysis layer surfaces alternative readings to the user.
SURFACE_AMBIGUOUS_ROLES = {
    "mudaaf_ilayh", "mafoul_bih", "mubtada", "fail",
    "ism_majrur", "naat", "matuf", "badal",
    "ism_inna", "khabar_inna", "khabar_kana", "khabar",
}


def _confidence_band(conf: Optional[float]) -> str:
    if conf is None:
        return "unknown"
    if conf >= 0.90:
        return "high"
    if conf >= 0.70:
        return "medium"
    if conf >= 0.50:
        return "low"
    return "very_low"


def _calibration_warning(role_conf: Optional[float]) -> Optional[str]:
    """Return a calibration warning string if the model is in a bin
    where it is historically overconfident."""
    if role_conf is None:
        return None
    if role_conf >= 0.95:
        return ("High confidence on a hard role. The validated model "
                "is correct only ~37% of the time at conf ≥ 0.95 on "
                "the held-out role axis.")
    return None


def _detect_constructions_from_roles(tokens: List[Dict[str, Any]]
                                       ) -> List[Dict[str, Any]]:
    """Detect construction spans from predicted roles. Heuristic but
    aligned with the project's canonical construction families."""
    detected: List[Dict[str, Any]] = []
    n = len(tokens)

    # kana sisters: ism_kana / khabar_kana
    has_kana = any(t["role"] in ("ism_kana", "khabar_kana") for t in tokens)
    if has_kana:
        members = [t["index"] for t in tokens
                    if t["role"] in ("ism_kana", "khabar_kana")]
        # try to locate kana token (usually preceding the ism)
        gov = None
        if members:
            gov = max(0, members[0] - 1)
        detected.append({
            "id": "c_kana", "family": "kana_sisters",
            "members": members, "governor": gov,
            "explanation": (
                "كان (kana) and its sisters take the subject as raf "
                "(ism_kana) and the predicate as nasb (khabar_kana)."
            ),
        })

    # inna sisters
    has_inna = any(t["role"] in ("ism_inna", "khabar_inna") for t in tokens)
    if has_inna:
        members = [t["index"] for t in tokens
                    if t["role"] in ("ism_inna", "khabar_inna")]
        gov = None
        if members:
            gov = max(0, members[0] - 1)
        detected.append({
            "id": "c_inna", "family": "inna_sisters",
            "members": members, "governor": gov,
            "explanation": (
                "إنّ (inna) and its sisters take the noun as nasb "
                "(ism_inna) and the predicate as raf (khabar_inna)."
            ),
        })

    # idafa: any mudaaf_ilayh + the preceding token
    for t in tokens:
        if t["role"] == "mudaaf_ilayh" and t["index"] > 0:
            members = [t["index"] - 1, t["index"]]
            detected.append({
                "id": f"c_idafa_{t['index']}", "family": "idafa",
                "members": members, "governor": members[0],
                "explanation": (
                    f"Iḍāfa: {tokens[members[0]]['surface']} → "
                    f"{tokens[members[1]]['surface']} (the second noun "
                    "completes the first; second is مضاف إليه in jarr)."
                ),
            })

    # ism_majrur (object of preposition)
    for t in tokens:
        if t["role"] == "ism_majrur" and t["index"] > 0:
            detected.append({
                "id": f"c_jar_{t['index']}", "family": "harf_jarr_phrase",
                "members": [t["index"] - 1, t["index"]],
                "governor": t["index"] - 1,
                "explanation": (
                    f"Prepositional phrase: "
                    f"{tokens[t['index']-1]['surface']} "
                    f"{tokens[t['index']]['surface']} (the noun is in "
                    "jarr, governed by the preceding particle)."
                ),
            })

    return detected


def _build_graph(tokens: List[Dict[str, Any]],
                  constructions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce a tiny graph spec (nodes + edges) for SVG rendering."""
    nodes = [{
        "id": f"t{t['index']}", "type": "token",
        "label": t["surface"], "role": t["role"],
        "case": t["case"], "marker": t["marker"],
        "index": t["index"],
    } for t in tokens]

    edges: List[Dict[str, Any]] = []
    # Sequential edges (visual scaffolding)
    for i in range(len(tokens) - 1):
        edges.append({
            "id": f"seq_{i}", "src": f"t{i}", "dst": f"t{i+1}",
            "type": "seq", "label": "",
        })
    # Construction edges (visual highlight)
    for c in constructions:
        members = c.get("members", [])
        gov = c.get("governor")
        if gov is None:
            continue
        for m in members:
            if m == gov or not (0 <= m < len(tokens)):
                continue
            edges.append({
                "id": f"{c['id']}_{m}", "src": f"t{gov}", "dst": f"t{m}",
                "type": "construction", "label": c["family"],
                "construction_id": c["id"],
            })

    return {"nodes": nodes, "edges": edges}


def _build_reasoning(tokens: List[Dict[str, Any]],
                      constructions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Per-important-token natural-language reasoning."""
    out: List[Dict[str, str]] = []
    for t in tokens:
        role = t["role"]
        case = t["case"]
        marker = t["marker"]
        why_role = ""
        why_case = ""
        why_marker = ""

        if role == "fail":
            why_role = "Predicted as فاعل (subject) — agent of an active verb."
            why_case = "Subjects take raf (nominative)."
        elif role == "mafoul_bih":
            why_role = "Predicted as مفعول به (direct object)."
            why_case = "Direct objects take nasb (accusative)."
        elif role == "mudaaf_ilayh":
            why_role = ("Predicted as مضاف إليه — second member of an iḍāfa, "
                        "completes the head noun.")
            why_case = "Mudaaf ilayh always takes jarr (genitive)."
        elif role == "ism_majrur":
            why_role = "Predicted as اسم مجرور — object of a preceding preposition."
            why_case = "Ism majrur takes jarr (genitive)."
        elif role == "ism_kana":
            why_role = "Subject of كان or its sisters."
            why_case = "Ism kana takes raf (nominative)."
        elif role == "khabar_kana":
            why_role = "Predicate of كان or its sisters."
            why_case = "Khabar kana takes nasb (accusative)."
        elif role == "ism_inna":
            why_role = "Subject of إنّ or its sisters."
            why_case = "Ism inna takes nasb (accusative)."
        elif role == "khabar_inna":
            why_role = "Predicate of إنّ or its sisters."
            why_case = "Khabar inna takes raf (nominative)."
        elif role == "mubtada":
            why_role = "Topic of a nominal sentence."
            why_case = "Mubtada takes raf (nominative)."
        elif role == "khabar":
            why_role = "Predicate of a nominal sentence."
            why_case = "Khabar takes raf (nominative)."
        else:
            why_role = f"Predicted role: {role}."
            why_case = f"Case: {case}."

        if marker:
            why_marker = (
                f"Marker: {marker} — the visible diacritic that signals "
                f"the {case} case."
            )

        out.append({
            "index": t["index"],
            "surface": t["surface"],
            "role": role, "case": case, "marker": marker,
            "why_role":   why_role,
            "why_case":   why_case,
            "why_marker": why_marker,
        })
    return out


class ModelHolder:
    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}

    def available_checkpoints(self) -> List[str]:
        return [n for n, p in CHECKPOINT_PATHS.items()
                if (p / "pytorch_model.bin").exists()]

    def get(self, name: str):
        if name in self._models:
            return self._models[name], self._tokenizers[name]
        path = CHECKPOINT_PATHS.get(name)
        if path is None or not (path / "pytorch_model.bin").exists():
            raise FileNotFoundError(f"checkpoint missing: {name} → {path}")
        model, tok = _load_checkpoint(path)
        self._models[name] = model
        self._tokenizers[name] = tok
        return model, tok


def _load_checkpoint(path: Path):
    import os
    os.environ.setdefault("USE_TF", "NO")
    import torch
    from transformers import AutoTokenizer
    from irab_tashkeel.morphology.dep_aware_model import DepAwareStructuredModel

    tok = AutoTokenizer.from_pretrained(
        str(path) if (path / "tokenizer.json").exists()
        else "UBC-NLP/AraT5v2-base-1024",
    )
    model = DepAwareStructuredModel(
        encoder_name="UBC-NLP/AraT5v2-base-1024",
        enable_morph_heads=True, morph_heads_enabled=None,
        enable_dep_features=True,
    )
    sd = torch.load(path / "pytorch_model.bin", map_location="cpu",
                    weights_only=True)
    model.load_state_dict(sd, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    return model, tok


def analyze_sentence(holder: ModelHolder, text: str,
                     checkpoint: str = "recovery") -> Dict[str, Any]:
    """Run one sentence through the model and return a polished
    structured response."""
    import torch
    import torch.nn.functional as F
    from irab_tashkeel.structured.schema import (
        ID_TO_CASE, ID_TO_MARKER, ID_TO_POS, ID_TO_ROLE,
        POS_LABELS, ROLE_LABELS,
    )

    model, tokenizer = holder.get(checkpoint)
    device = next(model.parameters()).device

    words = text.strip().split()
    if not words:
        return {"text": text, "tokens": [], "constructions": [],
                "graph": {"nodes": [], "edges": []}, "reasoning": [],
                "warnings": []}

    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                    truncation=True, max_length=512)
    word_ids = enc.word_ids(batch_index=0)
    starts: List[int] = []
    ends:   List[int] = []
    for wi in range(len(words)):
        idxs = [i for i, w in enumerate(word_ids) if w == wi]
        if not idxs:
            continue
        starts.append(idxs[0])
        ends.append(idxs[-1] + 1)
    word_starts = torch.tensor([starts], device=device)
    word_ends   = torch.tensor([ends], device=device)
    word_mask   = torch.ones((1, len(starts)), dtype=torch.long, device=device)

    enc = {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}

    with torch.no_grad():
        out = model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            word_starts=word_starts, word_ends=word_ends, word_mask=word_mask,
            return_dict=True,
        )

    case_p   = F.softmax(out["case_logits"],   dim=-1)[0]
    role_p   = F.softmax(out["role_logits"],   dim=-1)[0]
    marker_p = F.softmax(out["marker_logits"], dim=-1)[0]
    pos_p    = F.softmax(out["pos_logits"],    dim=-1)[0]

    tokens_out: List[Dict[str, Any]] = []
    n_low_conf = 0
    n_calib_warn = 0
    for w in range(len(words)):
        # Top-1 + top-3 alternatives for role
        case_top = int(case_p[w].argmax().item())
        case_conf = float(case_p[w, case_top].item())
        role_topk = torch.topk(role_p[w], k=min(3, role_p.shape[-1]))
        role_top = int(role_topk.indices[0].item())
        role_conf = float(role_topk.values[0].item())
        role_alts = [
            {"role":  ID_TO_ROLE.get(int(role_topk.indices[k].item())),
             "conf":  round(float(role_topk.values[k].item()), 4)}
            for k in range(min(3, role_topk.indices.shape[-1]))
        ]
        marker_top = int(marker_p[w].argmax().item())
        marker_conf = float(marker_p[w, marker_top].item())
        pos_top = int(pos_p[w].argmax().item())
        pos_conf = float(pos_p[w, pos_top].item())

        band = _confidence_band(role_conf)
        warn = _calibration_warning(role_conf)
        if band in ("low", "very_low"):
            n_low_conf += 1
        if warn:
            n_calib_warn += 1

        tokens_out.append({
            "index": w, "surface": words[w],
            "case":   ID_TO_CASE.get(case_top),
            "role":   ID_TO_ROLE.get(role_top),
            "marker": ID_TO_MARKER.get(marker_top),
            "pos":    ID_TO_POS.get(pos_top, POS_LABELS[pos_top] if pos_top < len(POS_LABELS) else None),
            "case_conf":   round(case_conf,   4),
            "role_conf":   round(role_conf,   4),
            "marker_conf": round(marker_conf, 4),
            "pos_conf":    round(pos_conf,    4),
            "role_alternatives": role_alts,
            "confidence_band":   band,
            "calibration_warning": warn,
            "is_surface_ambiguous": ID_TO_ROLE.get(role_top) in SURFACE_AMBIGUOUS_ROLES,
        })

    constructions = _detect_constructions_from_roles(tokens_out)
    graph = _build_graph(tokens_out, constructions)
    reasoning = _build_reasoning(tokens_out, constructions)

    warnings: List[str] = []
    if n_low_conf > 0:
        warnings.append(f"{n_low_conf} token(s) have low role confidence (< 0.7).")
    if n_calib_warn > 0:
        warnings.append(
            f"{n_calib_warn} token(s) at conf ≥ 0.95 — the model is "
            f"~37% accurate at this band on the held-out role axis. "
            f"Treat with caution."
        )

    return {
        "text": text, "checkpoint": checkpoint,
        "n_tokens": len(words), "tokens": tokens_out,
        "constructions": constructions,
        "graph": graph,
        "reasoning": reasoning,
        "warnings": warnings,
        "calibration_note": (
            "Validated recovery: ECE on role ≈ 0.49 on failures. "
            "Use the high-confidence-wrong warning when shown."
            if checkpoint == "recovery"
            else f"Checkpoint: {checkpoint}"
        ),
    }


def compare_sentence(holder: ModelHolder, text: str) -> Dict[str, Any]:
    """Run the same sentence through every available checkpoint."""
    out: Dict[str, Any] = {"text": text,
                            "available": holder.available_checkpoints()}
    for name in ("recovery", "phase3a", "stage7"):
        try:
            out[name] = analyze_sentence(holder, text, checkpoint=name)
        except FileNotFoundError as e:
            out[name] = {"error": str(e)}

    # Token-level diff between recovery and phase3a
    if (isinstance(out.get("recovery"), dict)
            and isinstance(out.get("phase3a"), dict)
            and "tokens" in out["recovery"] and "tokens" in out["phase3a"]):
        diff = []
        for r, p in zip(out["recovery"]["tokens"], out["phase3a"]["tokens"]):
            changes = []
            if r["role"] != p["role"]:
                changes.append({"axis": "role", "phase3a": p["role"],
                                "recovery": r["role"]})
            if r["case"] != p["case"]:
                changes.append({"axis": "case", "phase3a": p["case"],
                                "recovery": r["case"]})
            if r["marker"] != p["marker"]:
                changes.append({"axis": "marker", "phase3a": p["marker"],
                                "recovery": r["marker"]})
            diff.append({
                "index": r["index"], "surface": r["surface"],
                "changes": changes, "n_changes": len(changes),
            })
        out["diff"] = diff

    return out
