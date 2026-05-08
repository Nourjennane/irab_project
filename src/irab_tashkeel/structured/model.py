"""Multi-head structured i'rāb classifier on top of an Arabic encoder.

The encoder is the encoder half of an AraT5v2 (or any compatible) seq2seq
checkpoint, loaded via ``transformers.AutoModel.from_pretrained(...).get_encoder()``.
On top of pooled per-word hidden states we run four independent linear heads
(case / role / marker / POS), each with its own cross-entropy loss.

Word pooling: mean of the hidden states over each word's subword span.

Output during inference includes per-head softmax confidence + entropy so
downstream code can flag low-confidence predictions and do ablation logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .crf import LinearChainCRF
from .schema import N_CASE, N_ROLE, N_MARKER, N_POS

IGNORE = -100


def _word_mean_pool(hidden: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor,
                    word_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool encoder hidden states over each word's subword span.

    Args:
        hidden:    (B, T, H)
        starts:    (B, W)  inclusive start of each word
        ends:      (B, W)  exclusive end
        word_mask: (B, W)  1 where word is real, 0 where padded
    Returns:
        (B, W, H) per-word pooled vectors (zeros where padded).
    """
    bsz, T, H = hidden.shape
    W = starts.size(1)
    device = hidden.device

    # Build a (B, W, T) gather mask: position t belongs to word w iff starts[b,w] <= t < ends[b,w].
    t_idx = torch.arange(T, device=device).view(1, 1, T)            # (1, 1, T)
    s = starts.unsqueeze(-1)                                         # (B, W, 1)
    e = ends.unsqueeze(-1)                                           # (B, W, 1)
    span_mask = (t_idx >= s) & (t_idx < e)                           # (B, W, T) bool
    span_mask = span_mask & word_mask.bool().unsqueeze(-1)           # zero out padded words

    span_lens = span_mask.sum(dim=-1).clamp(min=1).unsqueeze(-1)     # (B, W, 1)
    # (B, W, T) @ (B, T, H) -> (B, W, H)
    pooled = torch.einsum("bwt,bth->bwh", span_mask.to(hidden.dtype), hidden)
    pooled = pooled / span_lens.to(hidden.dtype)
    pooled = pooled * word_mask.unsqueeze(-1).to(hidden.dtype)
    return pooled


def _word_first_pool(hidden: torch.Tensor, starts: torch.Tensor,
                     word_mask: torch.Tensor) -> torch.Tensor:
    """Pick the first-subword hidden state for each word (BERT-style).

    Token classification literature consistently finds first-subword pooling
    works at least as well as mean over Latin-script tokenizers; for Arabic
    SentencePiece (where the first subword usually corresponds to the stem),
    the first-subword vector preserves a cleaner morphological signal.
    """
    bsz, T, H = hidden.shape
    W = starts.size(1)
    # Gather hidden[batch, starts[batch, word]] for each (batch, word).
    # starts is (B, W); we need indices clamped to T-1 to avoid OOB on padded slots.
    safe_starts = starts.clamp(min=0, max=T - 1)
    pooled = torch.gather(hidden, dim=1,
                          index=safe_starts.unsqueeze(-1).expand(bsz, W, H))
    pooled = pooled * word_mask.unsqueeze(-1).to(hidden.dtype)
    return pooled


@dataclass
class StructuredOutput:
    case_logits: torch.Tensor          # (B, W, N_CASE)
    role_logits: torch.Tensor          # (B, W, N_ROLE)
    marker_logits: torch.Tensor        # (B, W, N_MARKER)
    pos_logits: torch.Tensor           # (B, W, N_POS)
    word_mask: torch.Tensor            # (B, W)
    loss: Optional[torch.Tensor] = None
    case_loss: Optional[torch.Tensor] = None
    role_loss: Optional[torch.Tensor] = None
    marker_loss: Optional[torch.Tensor] = None
    pos_loss: Optional[torch.Tensor] = None


class StructuredIrabModel(nn.Module):
    """AraT5v2 encoder + four classification heads.

    Args:
        encoder_name: HuggingFace repo id of the seq2seq checkpoint to load
            the encoder from (e.g. ``UBC-NLP/AraT5v2-base-1024``).  We pull
            ``model.get_encoder()`` so only the encoder weights are loaded
            into memory.
        head_dropout: dropout applied on the pooled per-word vector before each head.
        loss_weights: (case, role, marker, pos) — multiplied with each head's CE.
    """

    def __init__(
        self,
        encoder_name: str = "UBC-NLP/AraT5v2-base-1024",
        head_dropout: float = 0.1,
        loss_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.5),
        label_smoothing: float = 0.0,
        role_class_weights: Optional[torch.Tensor] = None,
        pooling_strategy: str = "mean",
        use_crf_role: bool = False,
        output_attentions: bool = False,
        n_role: Optional[int] = None,
    ):
        super().__init__()
        # Prefer T5EncoderModel for T5-family checkpoints: it loads ONLY the
        # encoder, avoiding the cyclic encoder<->decoder<->shared-embedding
        # references that segfault during CUDA destructor teardown on
        # transformers 4.46 + torch 2.5.  For non-T5 checkpoints (AraBERT etc)
        # we fall back to AutoModel which is already encoder-only.
        cfg = AutoConfig.from_pretrained(encoder_name)
        is_t5 = cfg.model_type == "t5" or cfg.model_type == "mt5"
        if is_t5:
            from transformers import T5EncoderModel
            self.encoder = T5EncoderModel.from_pretrained(encoder_name)
            self.hidden_size = self.encoder.config.d_model
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)
            self.hidden_size = getattr(self.encoder.config, "d_model",
                                       getattr(self.encoder.config, "hidden_size"))

        # Phase 4a — role head dim is parametrised (default = N_ROLE = 25 for
        # the v3 / rev 2 / Phase 1 path). When n_role is provided (e.g. 34 for
        # taxonomy_v4), the role head and the role class-weights buffer scale
        # accordingly. All other heads are unchanged.
        self._n_role = int(n_role) if n_role is not None else N_ROLE

        self.dropout = nn.Dropout(head_dropout)
        self.case_head = nn.Linear(self.hidden_size, N_CASE)
        self.role_head = nn.Linear(self.hidden_size, self._n_role)
        self.marker_head = nn.Linear(self.hidden_size, N_MARKER)
        self.pos_head = nn.Linear(self.hidden_size, N_POS)
        self.loss_weights = loss_weights
        self.encoder_name = encoder_name
        self.label_smoothing = float(label_smoothing)
        if pooling_strategy not in ("mean", "first"):
            raise ValueError(f"pooling_strategy must be 'mean' or 'first', got {pooling_strategy}")
        self.pooling_strategy = pooling_strategy

        # Role class weights are saved as a non-trainable buffer so they
        # travel with the model checkpoint and the device move. Phase 4a:
        # buffer dim follows self._n_role (so v3 → 25, v4 → 34).
        if role_class_weights is None:
            self.register_buffer("role_class_weights", torch.ones(self._n_role))
            self._has_role_weights = False
        else:
            w = role_class_weights.detach().to(torch.float32).clone()
            assert w.shape == (self._n_role,), \
                f"role_class_weights must be shape ({self._n_role},), got {tuple(w.shape)}"
            self.register_buffer("role_class_weights", w)
            self._has_role_weights = True

        self.use_crf_role = use_crf_role
        if use_crf_role:
            # CRF taxonomy size follows self._n_role
            self.role_crf = LinearChainCRF(self._n_role)
        else:
            self.role_crf = None
        self.output_attentions = output_attentions

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(**kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        word_starts: torch.LongTensor,
        word_ends: torch.LongTensor,
        word_mask: torch.LongTensor,
        case_labels: Optional[torch.LongTensor] = None,
        role_labels: Optional[torch.LongTensor] = None,
        marker_labels: Optional[torch.LongTensor] = None,
        pos_labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        # The HF Trainer often passes a `labels` kwarg or `num_items_in_batch`;
        # ignore harmlessly.
        enc_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.output_attentions:
            enc_kwargs["output_attentions"] = True
        enc_out = self.encoder(**enc_kwargs)
        hidden = enc_out.last_hidden_state                                   # (B, T, H)
        if self.pooling_strategy == "first":
            pooled = _word_first_pool(hidden, word_starts, word_mask)
        else:
            pooled = _word_mean_pool(hidden, word_starts, word_ends, word_mask)
        pooled = self.dropout(pooled)

        case_logits = self.case_head(pooled)
        role_logits = self.role_head(pooled)
        marker_logits = self.marker_head(pooled)
        pos_logits = self.pos_head(pooled)

        out = StructuredOutput(
            case_logits=case_logits,
            role_logits=role_logits,
            marker_logits=marker_logits,
            pos_logits=pos_logits,
            word_mask=word_mask,
        )

        if case_labels is not None:
            wc, wr, wm, wp = self.loss_weights
            ls = self.label_smoothing
            loss_case = F.cross_entropy(
                case_logits.reshape(-1, N_CASE), case_labels.reshape(-1),
                ignore_index=IGNORE, label_smoothing=ls,
            )
            role_weight = self.role_class_weights if self._has_role_weights else None
            if self.use_crf_role and self.role_crf is not None:
                # CRF: emissions = role_logits, tags = role_labels (clamped),
                # mask = word_mask. The CRF NLL is per-sequence; we average it
                # and DO NOT apply label smoothing (CRFs don't combine cleanly
                # with label smoothing; the structural transitions already act
                # as regularisation).
                loss_role = self.role_crf(role_logits, role_labels, word_mask)
            else:
                loss_role = F.cross_entropy(
                    role_logits.reshape(-1, self._n_role), role_labels.reshape(-1),
                    ignore_index=IGNORE, label_smoothing=ls, weight=role_weight,
                )
            loss_marker = F.cross_entropy(
                marker_logits.reshape(-1, N_MARKER), marker_labels.reshape(-1),
                ignore_index=IGNORE, label_smoothing=ls,
            )
            loss_pos = F.cross_entropy(
                pos_logits.reshape(-1, N_POS), pos_labels.reshape(-1),
                ignore_index=IGNORE, label_smoothing=ls,
            )
            out.case_loss = loss_case
            out.role_loss = loss_role
            out.marker_loss = loss_marker
            out.pos_loss = loss_pos
            out.loss = wc * loss_case + wr * loss_role + wm * loss_marker + wp * loss_pos

        if return_dict:
            # Trainer.compute_loss expects a dict-like with .loss
            d = {
                "loss": out.loss,
                "case_logits": out.case_logits,
                "role_logits": out.role_logits,
                "marker_logits": out.marker_logits,
                "pos_logits": out.pos_logits,
                "word_mask": out.word_mask,
                "case_loss": out.case_loss,
                "role_loss": out.role_loss,
                "marker_loss": out.marker_loss,
                "pos_loss": out.pos_loss,
            }
            if self.output_attentions and hasattr(enc_out, "attentions") and enc_out.attentions:
                # last-layer attention, mean over heads: (B, T, T)
                last = enc_out.attentions[-1]
                d["attentions"] = last.mean(dim=1)
            return d
        return out

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        word_starts: torch.LongTensor,
        word_ends: torch.LongTensor,
        word_mask: torch.LongTensor,
    ):
        """Return (preds_dict, confidence_dict, full_output_dict) per head.

        preds_dict[head] : (B, W) long
        confidence_dict[head] : (B, W) float — max softmax prob
        For role, when CRF is enabled, ``preds["role"]`` holds the Viterbi-
        decoded path and ``confs["role"]`` is the per-word marginal probability
        (approximated as the softmax-prob of the Viterbi tag at each position).
        """
        out = self.forward(input_ids, attention_mask, word_starts, word_ends, word_mask, return_dict=True)
        preds = {}
        confs = {}
        for name in ("case", "marker", "pos"):
            logits = out[f"{name}_logits"]
            probs = torch.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            preds[name] = idx
            confs[name] = conf
        # Role: Viterbi if CRF, argmax otherwise.
        role_logits = out["role_logits"]
        if self.use_crf_role and self.role_crf is not None:
            paths = self.role_crf.decode(role_logits, word_mask)
            B, W, _ = role_logits.shape
            role_idx = torch.zeros((B, W), dtype=torch.long, device=role_logits.device)
            for b, p in enumerate(paths):
                for j, t in enumerate(p):
                    role_idx[b, j] = t
            role_probs = torch.softmax(role_logits, dim=-1)
            role_conf = role_probs.gather(-1, role_idx.unsqueeze(-1)).squeeze(-1)
            preds["role"] = role_idx
            confs["role"] = role_conf
        else:
            role_probs = torch.softmax(role_logits, dim=-1)
            role_conf, role_idx = role_probs.max(dim=-1)
            preds["role"] = role_idx
            confs["role"] = role_conf
        return preds, confs, out
