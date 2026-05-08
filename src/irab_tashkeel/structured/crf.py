"""Linear-chain CRF for the role head.

Pure-pytorch implementation, no external CRF library. Used to replace the
role head's independent cross-entropy loss with a sequence-aware NLL that
captures syntactic transitions (e.g. *harf_jarr* -> *ism_majrur*,
*kāna* -> *ism_kana* -> *khabar_kana*).

The CRF is masked: word-level pad slots (``word_mask=0``) are skipped in both
the partition and the gold-path score, so the CRF sees a variable-length
sequence per batch element.

Stability: the transition matrix is initialized from the empirical role
bigram log-probabilities of the training corpus (computed by
:func:`compute_role_bigrams`). This avoids the cold-start instability that
random-init CRFs sometimes show on the first few hundred steps.

Interfaces:

* :meth:`LinearChainCRF.forward(emissions, tags, mask) -> nll` — average NLL.
* :meth:`LinearChainCRF.decode(emissions, mask) -> List[List[int]]` — Viterbi
  best paths per batch element.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """Linear-chain CRF with start, end, and transition parameters."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        # Init at zero so the CRF starts close to the unstructured emission-only
        # baseline; bigram init (init_from_bigrams) overrides this if available.
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))

    @torch.no_grad()
    def init_from_bigrams(
        self,
        transitions_log: torch.Tensor,
        start_log: torch.Tensor,
        end_log: torch.Tensor,
    ) -> None:
        """Set start/end/transitions from empirical log-bigram statistics."""
        assert transitions_log.shape == (self.num_tags, self.num_tags)
        assert start_log.shape == (self.num_tags,)
        assert end_log.shape == (self.num_tags,)
        self.transitions.copy_(transitions_log)
        self.start_transitions.copy_(start_log)
        self.end_transitions.copy_(end_log)

    # ------------------------------------------------------------------
    def forward(
        self,
        emissions: torch.Tensor,        # (B, T, K)
        tags: torch.Tensor,             # (B, T)
        mask: torch.Tensor,             # (B, T)  uint8/bool/long
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Negative log-likelihood of the gold tag sequence."""
        if mask.dtype != torch.bool:
            mask = mask.bool()
        # Replace ignored (=-100 or out-of-range) tags with 0 to avoid OOB
        # gathers; the mask zero will zero out their contribution anyway.
        safe_tags = tags.clone()
        safe_tags[~mask] = 0
        safe_tags.clamp_(min=0, max=self.num_tags - 1)
        log_num = self._gold_score(emissions, safe_tags, mask)
        log_den = self._partition(emissions, mask)
        nll = log_den - log_num
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        return nll

    # ------------------------------------------------------------------
    def _gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of emission + transition scores along the gold path."""
        B, T, K = emissions.shape
        device = emissions.device
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(
            1, tags[:, :1]
        ).squeeze(1)
        for t in range(1, T):
            mask_t = mask[:, t].to(emissions.dtype)
            emit_t = emissions[:, t].gather(1, tags[:, t : t + 1]).squeeze(1)
            trans_t = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (emit_t + trans_t) * mask_t
        # Add end-transition for the last *real* position in each row.
        last_idx = (mask.long().sum(dim=1) - 1).clamp(min=0)
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    # ------------------------------------------------------------------
    def _partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """log-sum-exp over all valid tag sequences (forward algorithm)."""
        B, T, K = emissions.shape
        # alpha[b, k] = log Z over paths ending in tag k at current step
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            mask_t = mask[:, t].to(emissions.dtype).unsqueeze(-1)  # (B, 1)
            # broadcast: (B, K, K) where score[b, prev_k, next_k]
            broadcast = (
                alpha.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + emissions[:, t].unsqueeze(1)
            )
            new_alpha = torch.logsumexp(broadcast, dim=1)
            alpha = mask_t * new_alpha + (1.0 - mask_t) * alpha
        return torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Viterbi decode. Returns per-batch list of best tag indices."""
        if mask.dtype != torch.bool:
            mask = mask.bool()
        B, T, K = emissions.shape
        device = emissions.device

        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, K)
        backpointers = torch.zeros((B, T, K), dtype=torch.long, device=device)
        for t in range(1, T):
            # broadcast: (B, K, K) score[b, prev, next] = score_prev + trans
            broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_idx = broadcast.max(dim=1)
            backpointers[:, t] = best_idx
            new_score = best_score + emissions[:, t]
            mask_t = mask[:, t].to(emissions.dtype).unsqueeze(-1)
            score = mask_t * new_score + (1.0 - mask_t) * score
        score = score + self.end_transitions.unsqueeze(0)
        best_last = score.argmax(dim=-1)

        out: List[List[int]] = []
        for b in range(B):
            length = int(mask[b].sum().item())
            if length == 0:
                out.append([])
                continue
            tag = int(best_last[b].item())
            path = [tag]
            for t in range(length - 1, 0, -1):
                tag = int(backpointers[b, t, tag].item())
                path.append(tag)
            path.reverse()
            out.append(path)
        return out


# ---------------------------------------------------------------------------
# Empirical bigram init
# ---------------------------------------------------------------------------
def compute_role_bigrams(
    train_jsonl: str | Path,
    role_to_id: dict,
    *,
    smoothing: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Empirical role-bigram log-probabilities for CRF init.

    Returns (transitions_log [K,K], start_log [K], end_log [K]).
    Add-`smoothing` prior on every cell to avoid -inf entries on unseen pairs.
    """
    K = len(role_to_id)
    counts = torch.full((K, K), smoothing, dtype=torch.float32)
    start_counts = torch.full((K,), smoothing, dtype=torch.float32)
    end_counts = torch.full((K,), smoothing, dtype=torch.float32)

    path = Path(train_jsonl)
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids = []
            for it in rec.get("items", []):
                r = it.get("role")
                if r in role_to_id:
                    ids.append(role_to_id[r])
            if not ids:
                continue
            start_counts[ids[0]] += 1.0
            end_counts[ids[-1]] += 1.0
            for i in range(len(ids) - 1):
                counts[ids[i], ids[i + 1]] += 1.0

    transitions_log = torch.log(counts / counts.sum(dim=1, keepdim=True))
    start_log = torch.log(start_counts / start_counts.sum())
    end_log = torch.log(end_counts / end_counts.sum())
    return transitions_log, start_log, end_log
