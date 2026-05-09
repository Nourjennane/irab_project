"""reasoning — Step 9 reasoning supervision schema + populators."""
from .ingestor import BaseReasoningIngestor, JsonlReasoningIngestor
from .populator import (
    populate_reasoning_for_sentence, populate_reasoning_pass,
)
from .templates import (
    ALL_TEMPLATES, ReasoningTemplate, get_template, supported_families,
)

__all__ = [
    "ALL_TEMPLATES", "ReasoningTemplate", "get_template", "supported_families",
    "populate_reasoning_for_sentence", "populate_reasoning_pass",
    "BaseReasoningIngestor", "JsonlReasoningIngestor",
]
