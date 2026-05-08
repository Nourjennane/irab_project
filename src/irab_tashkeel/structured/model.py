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

        self.dropout = nn.Dropout(head_dropout)
        self.case_head = nn.Linear(self.hidden_size, N_CASE)
        self.role_head = nn.Linear(self.hidden_size, N_ROLE)
        self.marker_head = nn.Linear(self.hidden_size, N_MARKER)
        self.pos_head = nn.Linear(self.hidden_size, N_POS)
        self.loss_weights = loss_weights
        self.encoder_name = encoder_name

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
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = enc_out.last_hidden_state                                   # (B, T, H)
        pooled = _word_mean_pool(hidden, word_starts, word_ends, word_mask)  # (B, W, H)
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
            loss_case = F.cross_entropy(
                case_logits.reshape(-1, N_CASE), case_labels.reshape(-1), ignore_index=IGNORE
            )
            loss_role = F.cross_entropy(
                role_logits.reshape(-1, N_ROLE), role_labels.reshape(-1), ignore_index=IGNORE
            )
            loss_marker = F.cross_entropy(
                marker_logits.reshape(-1, N_MARKER), marker_labels.reshape(-1), ignore_index=IGNORE
            )
            loss_pos = F.cross_entropy(
                pos_logits.reshape(-1, N_POS), pos_labels.reshape(-1), ignore_index=IGNORE
            )
            out.case_loss = loss_case
            out.role_loss = loss_role
            out.marker_loss = loss_marker
            out.pos_loss = loss_pos
            out.loss = wc * loss_case + wr * loss_role + wm * loss_marker + wp * loss_pos

        if return_dict:
            # Trainer.compute_loss expects a dict-like with .loss
            return {
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
        """Return (preds_dict, confidence_dict) per head.

        preds_dict[head] : (B, W) long
        confidence_dict[head] : (B, W) float — max softmax prob
        """
        out = self.forward(input_ids, attention_mask, word_starts, word_ends, word_mask, return_dict=True)
        preds = {}
        confs = {}
        for name in ("case", "role", "marker", "pos"):
            logits = out[f"{name}_logits"]
            probs = torch.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            preds[name] = idx
            confs[name] = conf
        return preds, confs, out
