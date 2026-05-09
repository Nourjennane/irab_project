"""Model loading + sentence analysis for the demo backend.

Both validated stage_7 and Phase 3-A baseline can be served side by
side (different paths). Loaded lazily on first request and cached;
backend stays cold-startable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]


CHECKPOINT_PATHS = {
    "stage7":  ROOT / "runs" / "validated_nextgen_stage7",
    "phase3a": ROOT / "runs" / "phase3a_491240" / "final",
}


SAMPLE_SENTENCES: List[Dict[str, str]] = [
    {"id": "msa1",     "tag": "MSA",       "text": "ذهب الطالبُ إلى المدرسةِ."},
    {"id": "msa2",     "tag": "MSA",       "text": "إنَّ الطلابَ مجتهدونَ."},
    {"id": "msa3",     "tag": "MSA",       "text": "كان الجوُّ جميلاً."},
    {"id": "quran1",   "tag": "Quranic",   "text": "بسم الله الرحمن الرحيم"},
    {"id": "quran2",   "tag": "Quranic",   "text": "الحمد لله رب العالمين"},
    {"id": "nest1",    "tag": "Nested",    "text": "إن كان الجوُّ جميلاً ذهبتُ."},
    {"id": "nest2",    "tag": "Nested",    "text": "ظنَّ المعلمُ أنَّ الطالبَ مجتهدٌ."},
    {"id": "ambig1",   "tag": "Ambiguous", "text": "في البيت رجلٌ."},
    {"id": "ambig2",   "tag": "Ambiguous", "text": "خرج الطلابُ من المدرسة."},
]


class ModelHolder:
    """Lazy singleton-ish loader for one or both checkpoints.

    Loading the model is expensive (~30s + 2GB GPU); we load on the
    first /api/analyze call for that checkpoint and cache.
    """

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
    model.to(device)
    model.eval()
    return model, tok


def analyze_sentence(holder: ModelHolder, text: str,
                     checkpoint: str = "stage7") -> Dict[str, Any]:
    """Run one sentence through the model and return per-token analysis."""
    import torch
    import torch.nn.functional as F
    from irab_tashkeel.structured.schema import (
        ID_TO_CASE, ID_TO_MARKER, ID_TO_POS, ID_TO_ROLE,
        POS_LABELS,
    )

    model, tokenizer = holder.get(checkpoint)
    device = next(model.parameters()).device

    words = text.strip().split()
    if not words:
        return {"text": text, "tokens": []}

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

    tokens_out = []
    for w in range(len(words)):
        ci = int(case_p[w].argmax().item())
        ri = int(role_p[w].argmax().item())
        mi = int(marker_p[w].argmax().item())
        pi = int(pos_p[w].argmax().item())
        tokens_out.append({
            "index": w,
            "surface": words[w],
            "case":   ID_TO_CASE.get(ci),
            "role":   ID_TO_ROLE.get(ri),
            "marker": ID_TO_MARKER.get(mi),
            "pos":    ID_TO_POS.get(pi, POS_LABELS[pi] if pi < len(POS_LABELS) else None),
            "case_conf":   round(float(case_p[w, ci].item()), 4),
            "role_conf":   round(float(role_p[w, ri].item()), 4),
            "marker_conf": round(float(marker_p[w, mi].item()), 4),
            "pos_conf":    round(float(pos_p[w, pi].item()), 4),
        })

    return {
        "text": text, "checkpoint": checkpoint,
        "n_tokens": len(words), "tokens": tokens_out,
        "calibration_note": (
            "stage_7 calib_gap on role ≈ 0.20 — confidences are uncalibrated "
            "and tend to overstate certainty"
            if checkpoint == "stage7"
            else "phase3a calib_gap on role ≈ 0.025"
        ),
    }


def compare_sentence(holder: ModelHolder, text: str) -> Dict[str, Any]:
    """Run the same sentence through both checkpoints and zip results."""
    out: Dict[str, Any] = {"text": text, "available": holder.available_checkpoints()}
    for name in ("phase3a", "stage7"):
        try:
            out[name] = analyze_sentence(holder, text, checkpoint=name)
        except FileNotFoundError as e:
            out[name] = {"error": str(e)}
    return out
