"""active_learning — candidate mining for the annotation queue."""
from .uncertainty_sampling import (
    rank_by_uncertainty, score_max_entropy, score_min_top1,
)
from .disagreement_sampling import (
    rank_by_disagreement, score_disagreement,
)
from .diversity_sampling import diversity_rank, signature
from .hard_case_mining import (
    DEFAULT_WEIGHTS, composite_score, rank_candidates,
    structural_difficulty,
)

__all__ = [
    "rank_by_uncertainty", "score_max_entropy", "score_min_top1",
    "rank_by_disagreement", "score_disagreement",
    "diversity_rank", "signature",
    "DEFAULT_WEIGHTS", "composite_score", "rank_candidates",
    "structural_difficulty",
]
