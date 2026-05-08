"""End-to-end structured i'rāb predictor (Phase 3 / v1 rebuild).

Loads a trained :class:`StructuredIrabModel`, runs forward inference, optionally
applies symbolic constraint reranking + Jaccard retrieval qualitative lookup,
and renders Arabic prose via the deterministic template renderer.

Two entry points:

* :meth:`StructuredPredictor.predict_sentence` returns a rich
  :class:`SentenceIrab` carrying confidence + constraints-fired + retrieved
  examples. Used by the qualitative trace + paper figures.
* :meth:`StructuredPredictor.predict_for_baseline` returns ``[{word, irab}, ...]``
  in the format expected by :mod:`irab_tashkeel.evaluation.run_baselines`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from ..retrieval import JaccardRetriever, RetrievedExample
from ..structured.dataset import IGNORE
from ..structured.model import StructuredIrabModel
from ..structured.schema import (
    CASE_LABELS, ROLE_LABELS, MARKER_LABELS, POS_LABELS,
    ID_TO_CASE, ID_TO_ROLE, ID_TO_MARKER, ID_TO_POS,
)
from ..structured.word_irab import SentenceIrab, WordIrab
from .symbolic_constraints import apply_constraints, ConstraintTrace
from .template_renderer import render_word


@dataclass
class StructuredPredictorConfig:
    apply_constraints: bool = True
    constraint_lambda_case: float = 1.5
    constraint_lambda_role: float = 0.8
    enabled_constraints: Optional[set] = None       # None -> all four
    render_prose: bool = True
    retriever_k: int = 0                             # 0 disables retrieval
    device: str = "auto"


class StructuredPredictor:
    """Load + run a trained :class:`StructuredIrabModel`."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        cfg: Optional[StructuredPredictorConfig] = None,
        retriever: Optional[JaccardRetriever] = None,
    ):
        self.cfg = cfg or StructuredPredictorConfig()
        self.retriever = retriever
        self.model_dir = Path(model_dir)

        from transformers import AutoTokenizer

        # Load training-time config (for encoder name) if present
        tcfg_path = self.model_dir / "structured_config.json"
        encoder_name = "UBC-NLP/AraT5v2-base-1024"
        if tcfg_path.exists():
            tcfg = json.loads(tcfg_path.read_text())
            encoder_name = tcfg.get("encoder_name", encoder_name)

        # Tokenizer is saved alongside the model dir
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)

        # Build model and load state-dict
        self.model = StructuredIrabModel(encoder_name=encoder_name)
        sd_path = self.model_dir / "pytorch_model.bin"
        sd = torch.load(sd_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[StructuredPredictor] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        self.model.eval()

        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)
        self.model.to(self.device)

    # -------- core inference --------
    def _encode_sentence(self, sentence: str):
        """Whitespace-split + per-word SentencePiece tokenize + record spans."""
        words = sentence.strip().split()
        if not words:
            return None

        ids: List[int] = []
        word_starts: List[int] = []
        word_ends: List[int] = []
        kept_words: List[str] = []
        for w in words:
            sub = self.tokenizer.encode(w, add_special_tokens=False)
            if not sub:
                continue
            start = len(ids)
            if start + len(sub) >= 320 - 1:
                break
            ids.extend(sub)
            word_starts.append(start)
            word_ends.append(len(ids))
            kept_words.append(w)
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id
        if eos_id is not None:
            ids.append(int(eos_id))

        return {
            "input_ids": torch.tensor([ids], dtype=torch.long, device=self.device),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=self.device),
            "word_starts": torch.tensor([word_starts], dtype=torch.long, device=self.device),
            "word_ends": torch.tensor([word_ends], dtype=torch.long, device=self.device),
            "word_mask": torch.ones((1, len(kept_words)), dtype=torch.long, device=self.device),
            "words": kept_words,
        }

    @torch.no_grad()
    def predict_sentence(self, sentence: str) -> SentenceIrab:
        enc = self._encode_sentence(sentence)
        if enc is None:
            return SentenceIrab(sentence=sentence, items=[])

        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            word_starts=enc["word_starts"],
            word_ends=enc["word_ends"],
            word_mask=enc["word_mask"],
            return_dict=True,
        )
        case_logits = out["case_logits"]
        role_logits = out["role_logits"]
        marker_logits = out["marker_logits"]
        pos_logits = out["pos_logits"]

        trace: Optional[ConstraintTrace] = None
        if self.cfg.apply_constraints:
            case_logits, role_logits, trace = apply_constraints(
                case_logits, role_logits, pos_logits,
                words=[enc["words"]],
                word_mask=enc["word_mask"],
                lambda_case=self.cfg.constraint_lambda_case,
                lambda_role=self.cfg.constraint_lambda_role,
                enabled=self.cfg.enabled_constraints,
            )

        case_probs = torch.softmax(case_logits, dim=-1)
        role_probs = torch.softmax(role_logits, dim=-1)
        marker_probs = torch.softmax(marker_logits, dim=-1)
        pos_probs = torch.softmax(pos_logits, dim=-1)
        case_conf, case_pred = case_probs[0].max(dim=-1)
        role_conf, role_pred = role_probs[0].max(dim=-1)
        marker_conf, marker_pred = marker_probs[0].max(dim=-1)
        pos_conf, pos_pred = pos_probs[0].max(dim=-1)

        items: List[WordIrab] = []
        for i, w in enumerate(enc["words"]):
            ci = int(case_pred[i].item())
            ri = int(role_pred[i].item())
            mi = int(marker_pred[i].item())
            pi = int(pos_pred[i].item())
            fired = trace.fired[0][i] if trace is not None else []
            wi = WordIrab(
                word=w,
                case=ID_TO_CASE.get(ci),
                role=ID_TO_ROLE.get(ri),
                marker=ID_TO_MARKER.get(mi),
                pos=ID_TO_POS.get(pi),
                case_conf=float(case_conf[i].item()),
                role_conf=float(role_conf[i].item()),
                marker_conf=float(marker_conf[i].item()),
                pos_conf=float(pos_conf[i].item()),
                constraints_fired=list(fired),
            )
            if self.cfg.render_prose:
                wi.irab_prose = render_word(wi)
            items.append(wi)

        return SentenceIrab(sentence=sentence, items=items)

    # -------- run_baselines.py adapter --------
    def predict_for_baseline(self, sentence: str) -> List[dict]:
        """Return ``[{word, irab}]`` exactly as ``run_baselines.evaluate_baseline`` expects.

        ``irab`` is the rendered Arabic prose; this lets the existing
        structural-extraction harness (``evaluation/structural.py``) score the
        rebuild on the same surface as every other system in the paper.
        """
        sent = self.predict_sentence(sentence)
        out: List[dict] = []
        for w in sent.items:
            d = {
                "word": w.word,
                "irab": w.irab_prose or render_word(w),
                "case": w.case,
                "role": w.role,
                "marker": w.marker,
                "pos": w.pos,
                "constraints_fired": list(w.constraints_fired),
            }
            if w.case_conf is not None:
                d["min_conf"] = w.min_confidence()
            out.append(d)
        return out

    # The run_baselines harness calls .predict_fn(sentence) and expects the
    # returned items to expose .to_dict(). Wrap our list-of-dicts to that shape.
    def predict(self, sentence: str):
        class _Item(dict):
            def to_dict(self):
                return dict(self)
        return [_Item(d) for d in self.predict_for_baseline(sentence)]

    # -------- retrieval (qualitative) --------
    def retrieve_similar(self, sentence: str, k: int = 5) -> List[RetrievedExample]:
        if self.retriever is None:
            return []
        return self.retriever.get_top_k(sentence, k=k)
