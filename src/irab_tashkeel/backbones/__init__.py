"""backbones — Step 6 backbone registry + benchmark utilities.

Public API:

    registry.BackboneSpec
    registry.get_backbone(backbone_id) → BackboneSpec
    registry.all_backbones() → List[BackboneSpec]
    registry.REGISTRY                       — the full dict
    registry.by_arch_family(arch)
    registry.by_pretraining(pretraining)
"""
from .registry import (
    BackboneSpec, REGISTRY,
    all_backbones, by_arch_family, by_pretraining, get_backbone,
)

__all__ = [
    "BackboneSpec", "REGISTRY",
    "all_backbones", "by_arch_family", "by_pretraining", "get_backbone",
]
