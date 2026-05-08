"""Retrieval interface for syntax-aware augmentation.

Phase A ships only a Jaccard-based reference implementation
(:class:`JaccardRetriever`) — at inference time it surfaces the K nearest
training-corpus sentences for the qualitative trace, but does not yet feed
them back into the encoder.

The interface is FAISS-compatible (same ``get_top_k(query, k)`` signature) so
the journal version can swap in a dense retriever without touching the
predictor.
"""

from .jaccard_retriever import JaccardRetriever, RetrievedExample
from .grammar_memory import (
    GrammarMemory, GrammarExample,
    detect_constructions, CONSTRUCTION_TAGS,
)

__all__ = [
    "JaccardRetriever", "RetrievedExample",
    "GrammarMemory", "GrammarExample",
    "detect_constructions", "CONSTRUCTION_TAGS",
]
