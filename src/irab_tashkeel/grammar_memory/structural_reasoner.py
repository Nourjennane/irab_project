"""Phase R2 — Retrieval-guided structural reasoners.

Implements per-construction reasoners that take a detected construction
span + top-k retrieved analogues from :class:`GrammarMemory` and emit:

1. A consensus-voted prediction for each word position in the span
   (case + role + marker, all canonical English labels)
2. A confidence score derived from per-position consensus rates and
   retrieval cosine scores
3. A human-readable rule string (e.g. "kana_completion: ism→raf,
   khabar→nasb")
4. A reasoning trace for explanation

The reasoner is family-specific because each construction family has
different syntactic templates and aligns differently. Surface-position
alignment is the first-pass strategy (position 0 = particle, 1 = ism,
2 = khabar for kana/inna; etc.).

A separate :class:`StructuralReasoningPredictor` (in
``structural_predictor.py``) consumes :class:`ReasoningOutput` and
applies it via three-tier confidence gating (override / strong-bias /
fallback) on top of Phase 3-A logits.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .memory import RetrievalHit
from ..structured.schema import (
    CASE_LABELS, ROLE_LABELS, MARKER_LABELS,
    canonicalize_case, canonicalize_role, canonicalize_marker,
)


# ---------------------------------------------------------------------------
# Reasoning output
# ---------------------------------------------------------------------------

@dataclass
class ReasoningOutput:
    """Structural reasoning output for one construction span."""
    family: str
    span: Tuple[int, int]
    span_len: int
    predicted: List[Dict]                 # one dict per position: {case, role, marker}
    consensus_per_pos: List[Dict]         # one dict per position: {case_rate, role_rate, marker_rate}
    confidence: float                     # 0..1 — overall confidence
    consensus_rate: float                 # mean across position * field
    n_hits: int                           # how many retrievals contributed
    mean_cosine: float
    rule: str                             # human-readable rule string
    reasoning_trace: str                  # multi-line explanation
    valid: bool = True                    # False when insufficient data
    note: str = ""                        # explanation when not valid
    # R2-v2: canonical grammar rule for case, applied with maximum confidence.
    # Per-position canonical case label (or None to fall through to consensus).
    canonical_case: List[Optional[str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers — canonicalization wrappers (handle both already-canonical and prose)
# ---------------------------------------------------------------------------

_CASE_SET = set(CASE_LABELS)
_ROLE_SET = set(ROLE_LABELS)
_MARKER_SET = set(MARKER_LABELS)


def _read_case(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    if s in _CASE_SET:
        return s
    c = canonicalize_case(s)
    return c if c in _CASE_SET else None


def _read_role(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    if s in _ROLE_SET:
        return s
    c = canonicalize_role(s)
    return c if c in _ROLE_SET else None


def _read_marker(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    if s in _MARKER_SET:
        return s
    c = canonicalize_marker(s)
    return c if c in _MARKER_SET else None


# ---------------------------------------------------------------------------
# Common consensus-voting machinery
# ---------------------------------------------------------------------------

def _vote_on_position(
    pos_idx: int,
    retrieved: List[RetrievalHit],
) -> Tuple[Counter, Counter, Counter, int]:
    """Aggregate per-position votes (case, role, marker) across retrievals.

    Each retrieval contributes one vote per (case, role, marker) at the
    aligned word position. Returns (case_votes, role_votes, marker_votes,
    n_voters) where n_voters is the number of retrievals that had a word
    at this position with at least one valid label.
    """
    case_votes: Counter = Counter()
    role_votes: Counter = Counter()
    marker_votes: Counter = Counter()
    n_voters = 0
    for h in retrieved:
        items = h.instance.items
        if pos_idx >= len(items):
            continue
        item = items[pos_idx]
        c = _read_case(item.get("case"))
        r = _read_role(item.get("role"))
        m = _read_marker(item.get("marker"))
        if not (c or r or m):
            continue
        n_voters += 1
        if c:
            case_votes[c] += 1
        if r:
            role_votes[r] += 1
        if m:
            marker_votes[m] += 1
    return case_votes, role_votes, marker_votes, n_voters


def _consensus_predict(
    span_len: int,
    retrieved: List[RetrievalHit],
) -> Tuple[List[Dict], List[Dict], float]:
    """Per-position consensus prediction + per-position consensus rates.

    Returns (predicted, consensus_per_pos, mean_consensus_rate).
    Each predicted dict has keys (case, role, marker) — None if no votes.
    Each consensus dict has keys (case_rate, role_rate, marker_rate).
    """
    predicted: List[Dict] = []
    consensus_per_pos: List[Dict] = []
    rates: List[float] = []
    for i in range(span_len):
        case_v, role_v, marker_v, n = _vote_on_position(i, retrieved)
        if n == 0:
            predicted.append({"case": None, "role": None, "marker": None})
            consensus_per_pos.append({"case_rate": 0.0, "role_rate": 0.0, "marker_rate": 0.0})
            continue
        # argmax with ties broken alphabetically (stable)
        def _top(c: Counter):
            if not c:
                return (None, 0.0)
            top, count = c.most_common(1)[0]
            return (top, count / n)
        c_top, c_rate = _top(case_v)
        r_top, r_rate = _top(role_v)
        m_top, m_rate = _top(marker_v)
        predicted.append({"case": c_top, "role": r_top, "marker": m_top})
        consensus_per_pos.append({
            "case_rate": c_rate, "role_rate": r_rate, "marker_rate": m_rate,
        })
        # mean of three field consensus rates for this position
        rates.append((c_rate + r_rate + m_rate) / 3.0)
    mean_rate = sum(rates) / max(len(rates), 1)
    return predicted, consensus_per_pos, mean_rate


def _format_trace(
    family: str,
    span: Tuple[int, int],
    query_words: List[str],
    retrieved: List[RetrievalHit],
    predicted: List[Dict],
    consensus_per_pos: List[Dict],
    confidence: float,
    rule: str,
    tier: str,
) -> str:
    """Pretty-print a multi-line reasoning trace for explanation."""
    lines: List[str] = []
    lines.append(f"Span [{span[0]}, {span[1]}] = {' '.join(query_words[span[0]:span[1]])}")
    lines.append(f"  Family: {family}")
    lines.append(f"  Retrieved {len(retrieved)} analogues:")
    for j, h in enumerate(retrieved[:3]):
        lines.append(
            f"    {j+1}. \"{h.instance.sentence[:60]}\" "
            f"(cosine {h.cosine:.3f}, sym {h.sym_overlap:.3f})"
        )
    lines.append("  Consensus:")
    for i, (pred, rate) in enumerate(zip(predicted, consensus_per_pos)):
        global_idx = span[0] + i
        word = query_words[global_idx] if global_idx < len(query_words) else "?"
        lines.append(
            f"    pos {i} (\"{word}\"): "
            f"case={pred.get('case')} ({rate.get('case_rate', 0):.2f}), "
            f"role={pred.get('role')} ({rate.get('role_rate', 0):.2f}), "
            f"marker={pred.get('marker')} ({rate.get('marker_rate', 0):.2f})"
        )
    lines.append(f"  Confidence: {confidence:.3f} → {tier}")
    lines.append(f"  Rule: {rule}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base reasoner
# ---------------------------------------------------------------------------

class ConstructionReasoner:
    """Base class. Subclasses set ``family`` and override ``rule_string``."""
    family: str = ""
    expected_span_len: int = 3
    min_retrievals: int = 3
    cosine_weight: float = 0.2     # confidence = 0.8 * consensus + 0.2 * mean_cosine

    def rule_string(self, particle_group: str, particle_surface: str) -> str:
        return f"{self.family}/{particle_group}: consensus from {self.min_retrievals}+ analogues"

    def canonical_case_rule(self, particle_group: str, particle_surface: str) -> List[Optional[str]]:
        """Canonical case labels per position, or None to fall through to consensus.

        R2-v2 fix #4: encode known grammar rules deterministically instead of
        voting on case from a noisy retrieval pool. Subclasses override.
        """
        return [None] * self.expected_span_len

    def reason(
        self,
        query_span: List[Dict],
        retrieved: List[RetrievalHit],
        query_words: List[str],
        span: Tuple[int, int],
        particle_group: str,
        particle_surface: str,
    ) -> ReasoningOutput:
        span_len = span[1] - span[0]
        if len(retrieved) < self.min_retrievals:
            return ReasoningOutput(
                family=self.family,
                span=span,
                span_len=span_len,
                predicted=[],
                consensus_per_pos=[],
                confidence=0.0,
                consensus_rate=0.0,
                n_hits=len(retrieved),
                mean_cosine=0.0,
                rule="",
                reasoning_trace="",
                valid=False,
                note=f"insufficient retrievals ({len(retrieved)} < {self.min_retrievals})",
            )

        predicted, consensus, consensus_rate = _consensus_predict(span_len, retrieved)
        mean_cosine = sum(h.cosine for h in retrieved) / len(retrieved)
        confidence = (1.0 - self.cosine_weight) * consensus_rate + self.cosine_weight * max(mean_cosine, 0.0)
        rule = self.rule_string(particle_group, particle_surface)
        canonical = self.canonical_case_rule(particle_group, particle_surface)
        # Pad/trim canonical to span_len
        if len(canonical) < span_len:
            canonical = list(canonical) + [None] * (span_len - len(canonical))
        canonical = list(canonical[:span_len])
        trace = _format_trace(
            self.family, span, query_words, retrieved,
            predicted, consensus, confidence, rule, tier="(set by predictor)",
        )
        return ReasoningOutput(
            family=self.family,
            span=span,
            span_len=span_len,
            predicted=predicted,
            consensus_per_pos=consensus,
            confidence=float(confidence),
            consensus_rate=float(consensus_rate),
            n_hits=len(retrieved),
            mean_cosine=float(mean_cosine),
            rule=rule,
            reasoning_trace=trace,
            valid=True,
            canonical_case=canonical,
        )


# ---------------------------------------------------------------------------
# Family-specific reasoners
# ---------------------------------------------------------------------------

class KanaReasoner(ConstructionReasoner):
    family = "kana_sisters"
    expected_span_len = 3

    def rule_string(self, particle_group: str, particle_surface: str) -> str:
        if particle_group == "kana_negation":
            return (
                f"kana_negation ({particle_surface}): particle is mabni, "
                "ism is raf, khabar is nasb (negated kana family)"
            )
        return (
            f"kana_completion ({particle_surface}): particle is mabni, "
            "ism is raf, khabar is nasb"
        )

    def canonical_case_rule(self, particle_group: str, particle_surface: str):
        # R2-v2 fix #4: hard-code the canonical kāna case rule.
        # Both kana_completion and kana_negation share the same case template:
        # particle=mabni, ism=raf, khabar=nasb.
        # Overriding case from this rule is more reliable than voting on case
        # from a retrieval pool that's polluted with prepositional-phrase
        # complements at pos 1/2 (case=jarr/raf instead of canonical raf/nasb).
        return ["mabni", "raf", "nasb"]


class IstithnaReasoner(ConstructionReasoner):
    family = "istithna"
    expected_span_len = 3
    # Istithna often only has 2 meaningful positions (particle + mustathna)
    min_retrievals: int = 3

    def rule_string(self, particle_group: str, particle_surface: str) -> str:
        if particle_group == "illa":
            return (
                f"istithna/illa ({particle_surface}): إلا is harf, "
                "mustathna is nasb in positive context"
            )
        if particle_group == "istithna_noun":
            return (
                f"istithna/noun ({particle_surface}): "
                f"{particle_surface} is nasb (mustathna), following is jarr (mudaaf_ilayh)"
            )
        if particle_group == "ma_3ada_phrase":
            return (
                f"istithna/ma_3ada ({particle_surface}): phrase governs nasb on the mustathna"
            )
        return f"istithna ({particle_surface}): consensus rule"


class MawsoolReasoner(ConstructionReasoner):
    family = "mawsool"
    expected_span_len = 3

    def rule_string(self, particle_group: str, particle_surface: str) -> str:
        if particle_group == "definite_relative":
            return (
                f"mawsool/definite ({particle_surface}): relative pronoun is mabni, "
                "role determined by what it modifies"
            )
        if particle_group == "indefinite_relative":
            return (
                f"mawsool/indefinite ({particle_surface}): "
                "indefinite relative is mabni, role context-dependent"
            )
        return f"mawsool ({particle_surface}): consensus rule"


class InnaReasoner(ConstructionReasoner):
    family = "inna_sisters"
    expected_span_len = 3

    def rule_string(self, particle_group: str, particle_surface: str) -> str:
        if particle_group == "inna_modal":
            return (
                f"inna_modal ({particle_surface}): particle is mabni, "
                "ism is nasb, khabar is raf (modal sister)"
            )
        return (
            f"inna_assertion ({particle_surface}): particle is mabni, "
            "ism is nasb, khabar is raf"
        )

    def canonical_case_rule(self, particle_group: str, particle_surface: str):
        # Inna mirrors kana with reversed cases: ism=nasb, khabar=raf.
        return ["mabni", "nasb", "raf"]


class QuranicProxyReasoner(ConstructionReasoner):
    family = "quranic_proxy"
    expected_span_len = 3

    def rule_string(self, particle_group: str, particle_surface: str) -> str:
        return (
            f"quranic_proxy/{particle_group} ({particle_surface}): "
            "particle is mabni, governing rule depends on subgroup"
        )


# ---------------------------------------------------------------------------
# Reasoner registry
# ---------------------------------------------------------------------------

REASONER_REGISTRY: Dict[str, ConstructionReasoner] = {
    "kana_sisters":  KanaReasoner(),
    "istithna":      IstithnaReasoner(),
    "mawsool":       MawsoolReasoner(),
    "inna_sisters":  InnaReasoner(),
    "quranic_proxy": QuranicProxyReasoner(),
    # iḍāfa explicitly not included — already well-handled by Phase 3-A
}


def get_reasoner(family: str) -> Optional[ConstructionReasoner]:
    """Return the reasoner for a construction family, or None if not built."""
    return REASONER_REGISTRY.get(family)


def supported_families() -> List[str]:
    return list(REASONER_REGISTRY.keys())
