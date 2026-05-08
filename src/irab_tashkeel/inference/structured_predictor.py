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
from ..structured.taxonomy_v4 import (
    ID_TO_ROLE_V4, N_ROLE_V4,
)
from ..structured.word_irab import SentenceIrab, WordIrab
from .symbolic_constraints import apply_constraints, apply_hierarchical, ConstraintTrace
from .template_renderer import render_word


@dataclass
class StructuredPredictorConfig:
    apply_constraints: bool = True
    constraint_lambda_case: float = 1.5
    constraint_lambda_role: float = 0.8
    enabled_constraints: Optional[set] = None       # None -> all 9
    apply_hierarchical: bool = True                  # role -> case biasing after argmax
    hierarchical_lambda: float = 1.0
    return_attention: bool = False                   # surface last-layer attention weights
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

        # Build model and load state-dict.  Reconstruct CRF flag + taxonomy
        # + Phase 1 morph + Phase 2 conditioning from structured_config.json.
        use_crf = False
        taxonomy = "v3"
        enable_morph_heads = False
        morph_heads_enabled_cfg: Optional[List[str]] = None
        conditioning_mechanism: Optional[str] = None
        conditioning_detached = False
        enable_dep_features = False
        if tcfg_path.exists():
            tcfg = json.loads(tcfg_path.read_text())
            use_crf = bool(tcfg.get("use_crf_role", False))
            taxonomy = tcfg.get("taxonomy", "v3")
            enable_morph_heads = bool(tcfg.get("enable_morph_heads", False))
            morph_heads_enabled_cfg = tcfg.get("morph_heads_enabled")
            conditioning_mechanism = tcfg.get("conditioning_mechanism")
            conditioning_detached = bool(tcfg.get("conditioning_detached", False))
            enable_dep_features = bool(tcfg.get("enable_dep_features", False))

        # Phase 4a: taxonomy switch picks the right ID_TO_ROLE map at predict-time.
        if taxonomy == "v4":
            self.id_to_role = ID_TO_ROLE_V4
            n_role_kw = {"n_role": N_ROLE_V4}
        else:
            self.id_to_role = ID_TO_ROLE
            n_role_kw = {}
        self.taxonomy = taxonomy

        # If conditioning OR dep features were used at training time we MUST
        # instantiate the matching subclass — the iʿrāb heads were trained
        # to consume the corresponding modulated/augmented representation,
        # not raw pooled features.
        needs_morph_class = (
            enable_morph_heads
            or (conditioning_mechanism and conditioning_mechanism != "none")
            or enable_dep_features
        )
        morph_set = set(morph_heads_enabled_cfg) if morph_heads_enabled_cfg else None
        if needs_morph_class and enable_dep_features:
            # Phase 3 path: dep-aware model. At inference time we pass
            # has_dep=False per word (no Stanza at predict time in this
            # iteration), so the model falls through to pooled_irab=pooled.
            # The iʿrāb heads were trained on a mix of has_dep=True
            # (distill_v2) and has_dep=False (UD-PADT, masked) examples,
            # so the no-dep inference path is in-distribution.
            from ..morphology.dep_aware_model import DepAwareStructuredModel
            self.model = DepAwareStructuredModel(
                encoder_name=encoder_name,
                use_crf_role=use_crf,
                output_attentions=self.cfg.return_attention,
                enable_morph_heads=True,
                morph_heads_enabled=morph_set,
                enable_dep_features=True,
                **n_role_kw,
            )
            self.has_morph_path = True
            self.has_dep_path = True
        elif needs_morph_class:
            from ..morphology.morph_model import MorphAugmentedStructuredModel
            self.model = MorphAugmentedStructuredModel(
                encoder_name=encoder_name,
                use_crf_role=use_crf,
                output_attentions=self.cfg.return_attention,
                enable_morph_heads=True,
                morph_heads_enabled=morph_set,
                conditioning_mechanism=conditioning_mechanism,
                conditioning_detached=conditioning_detached,
                **n_role_kw,
            )
            self.has_morph_path = True
            self.has_dep_path = False
        else:
            self.model = StructuredIrabModel(
                encoder_name=encoder_name,
                use_crf_role=use_crf,
                output_attentions=self.cfg.return_attention,
                **n_role_kw,
            )
            self.has_morph_path = False
            self.has_dep_path = False
        sd_path = self.model_dir / "pytorch_model.bin"
        sd = torch.load(sd_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            # Filter: missing role_crf.* keys are expected when CRF wasn't trained
            non_crf_missing = [k for k in missing if not k.startswith("role_crf.")]
            non_crf_unexpected = [k for k in unexpected if not k.startswith("role_crf.")]
            if non_crf_missing or non_crf_unexpected:
                print(f"[StructuredPredictor] load_state_dict non-CRF missing={non_crf_missing} unexpected={non_crf_unexpected}")
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
        attentions = out.get("attentions")

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

        # Decode role first (Viterbi if CRF, argmax otherwise) so the
        # hierarchical step can use the structured role prediction.
        if self.model.use_crf_role and self.model.role_crf is not None:
            paths = self.model.role_crf.decode(role_logits, enc["word_mask"])
            role_pred = torch.zeros_like(case_logits[..., 0], dtype=torch.long)
            for b, p in enumerate(paths):
                for j, t in enumerate(p):
                    role_pred[b, j] = t
            role_probs = torch.softmax(role_logits, dim=-1)
            role_conf = role_probs[0].gather(-1, role_pred[0].unsqueeze(-1)).squeeze(-1)
            role_pred = role_pred[0]
        else:
            role_probs = torch.softmax(role_logits, dim=-1)
            role_conf, role_pred = role_probs[0].max(dim=-1)

        # Hierarchical role -> case biasing
        if self.cfg.apply_hierarchical:
            case_logits, hier_trace = apply_hierarchical(
                case_logits,
                role_pred=role_pred.unsqueeze(0) if role_pred.dim() == 1 else role_pred,
                word_mask=enc["word_mask"],
                lambda_hier=self.cfg.hierarchical_lambda,
                trace=trace,
            )
            trace = hier_trace

        case_probs = torch.softmax(case_logits, dim=-1)
        marker_probs = torch.softmax(marker_logits, dim=-1)
        pos_probs = torch.softmax(pos_logits, dim=-1)
        case_conf, case_pred = case_probs[0].max(dim=-1)
        marker_conf, marker_pred = marker_probs[0].max(dim=-1)
        pos_conf, pos_pred = pos_probs[0].max(dim=-1)

        # Per-word influence: pool attention by averaging over the subwords
        # of each input word, then row-wise normalise so each row sums to 1.
        # This becomes the demo's attention-heatmap feed.
        per_word_influence: Optional[torch.Tensor] = None
        if attentions is not None:
            attn = attentions[0]                 # (T, T) — averaged over heads in model.forward
            T = attn.shape[0]
            n_words = len(enc["words"])
            ws = enc["word_starts"][0]
            we = enc["word_ends"][0]
            # influence[i, j] = mean over t in word i, mean over t' in word j
            inf = torch.zeros((n_words, n_words), dtype=attn.dtype)
            for i in range(n_words):
                rs, re = int(ws[i].item()), int(we[i].item())
                if re <= rs: continue
                row = attn[rs:re].mean(dim=0)  # (T,)
                for j in range(n_words):
                    cs, ce = int(ws[j].item()), int(we[j].item())
                    if ce <= cs: continue
                    inf[i, j] = row[cs:ce].mean()
            # row-normalise so each word's influence row sums to 1
            inf = inf / inf.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            per_word_influence = inf

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
                role=self.id_to_role.get(ri),
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

        sent = SentenceIrab(sentence=sentence, items=items)
        # Stash the attention matrix on the SentenceIrab so the demo / qual
        # trace can read it back. We don't add it to WordIrab to keep the
        # dataclass clean for serialisation.
        if per_word_influence is not None:
            sent.influence = per_word_influence.tolist()  # type: ignore[attr-defined]
        return sent

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
