"""schema_v2 → torch Dataset adapter.

Wraps a list of :class:`schema_v2.Sentence` objects so a torch
:class:`DataLoader` can iterate them. Tokenisation happens
lazily per item (not at construction time), so the dataset stays
memory-efficient for large corpora.

Returned items are flat dicts with raw text + per-token labels;
the :class:`Collator` is responsible for padding + tensorisation.

This design keeps the dataset agnostic to the encoder; the same
dataset works with AraT5v2 / AraBART / CAMeLBERT etc., differing
only in tokeniser at the collator stage.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..data_v2.schema_v2 import Sentence


# Optional torch import — keep heavy dependency lazy
def _require_torch():
    try:
        import torch
        from torch.utils.data import Dataset as _TorchDataset
        return torch, _TorchDataset
    except ImportError as e:
        raise ImportError(
            "training_v2.dataset requires torch; install with `pip install torch`"
        ) from e


class SchemaV2Dataset:
    """Light-weight Dataset wrapper around a list of schema_v2 sentences.

    The dataset itself doesn't depend on torch; the :class:`Collator`
    converts items to tensors at batch time. Use this directly as a
    torch.utils.data.Dataset since it implements ``__len__`` +
    ``__getitem__``.

    Returned items
    --------------

    .. code-block:: python

        {
          "sentence_id": str,
          "raw_text":    str,
          "words":       List[str],          # surface tokens
          "case":        List[str | None],
          "role":        List[str | None],
          "marker":      List[str | None],
          "pos":         List[str | None],
          "morph": {
              "gender":   List[str | None],
              "number":   List[str | None],
              "definite": List[str | None],
              "person":   List[str | None],
              "aspect":   List[str | None],
              "mood":     List[str | None],
              "voice":    List[str | None],
          },
          "dep_heads":   List[int],           # 0-based; -1 = unset
          "dep_labels":  List[str | None],
          "constructions": [{family, span, head_idx}],
          "metadata":    {domain, source, source_id, annotation_quality},
          "completeness": dict,
          "curriculum":   {difficulty_level, semantic_pressure_score, ...},
        }
    """

    def __init__(self, sentences: List[Sentence]):
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.sentences[idx]

        words = [t.surface for t in s.tokens]
        case  = [t.case.value for t in s.tokens]
        role  = [t.role.value for t in s.tokens]
        marker = [t.marker.value for t in s.tokens]
        pos   = [t.pos.value for t in s.tokens]

        morph = {
            "gender":   [t.morph.gender.value   for t in s.tokens],
            "number":   [t.morph.number.value   for t in s.tokens],
            "definite": [t.morph.definite.value for t in s.tokens],
            "person":   [t.morph.person.value   for t in s.tokens],
            "aspect":   [t.morph.aspect.value   for t in s.tokens],
            "mood":     [t.morph.mood.value     for t in s.tokens],
            "voice":    [t.morph.voice.value    for t in s.tokens],
        }

        dep_heads = [t.dep_head_idx if t.dep_head_idx is not None else -1
                     for t in s.tokens]
        dep_labels = [t.dep_label.value for t in s.tokens]

        constructions = [{
            "family": c.family, "subgroup": c.subgroup,
            "token_indices": list(c.token_indices),
            "head_idx": c.head_idx,
            "ambiguity_score": c.ambiguity_score,
        } for c in s.constructions]

        return {
            "sentence_id": s.sentence_id,
            "raw_text": s.raw_text,
            "words": words,
            "case": case, "role": role, "marker": marker, "pos": pos,
            "morph": morph,
            "dep_heads": dep_heads, "dep_labels": dep_labels,
            "constructions": constructions,
            "metadata": {
                "domain": s.metadata.domain,
                "source": s.metadata.source,
                "source_id": s.metadata.source_id,
                "annotation_quality": s.metadata.annotation_quality,
            },
            "completeness": {
                "has_morph": s.completeness.has_morph,
                "has_dep": s.completeness.has_dep,
                "has_role": s.completeness.has_role,
                "has_marker": s.completeness.has_marker,
                "fields_complete_pct": s.completeness.fields_complete_pct,
            },
            "curriculum": {
                "difficulty_level": s.curriculum.difficulty_level,
                "semantic_pressure_score": s.curriculum.semantic_pressure_score,
                "dependency_depth": s.curriculum.dependency_depth,
                "ambiguity_score": s.curriculum.ambiguity_score,
            },
        }
