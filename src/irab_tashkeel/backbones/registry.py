"""Backbone registry — Step 6 candidate models.

The frozen baseline established (§5.2(a)) that scaling within
the same Arabic-pretrained family on the same Haiku-distilled
corpus produces no measurable Gazelle improvement (every pairwise
McNemar p=1.000 across 296M / 580M / 792M / 13B). The Step 6
question is: does *substituting* the backbone (different
pretraining mix, different family) lift performance?

Per ``docs/roadmap/backbone_upgrade.md`` §benchmarks-to-run, the
benchmark suite covers ten backbones spanning:

  - encoder-decoder MSA (frozen-baseline reference)
  - encoder-decoder multi-dialect
  - encoder-only MSA
  - encoder-only classical-Arabic specialisation
  - multilingual T5
  - long-context encoder-only
  - decoder-only Arabic
  - instruction-tuned

This module provides a uniform handle for each: a ``BackboneSpec``
with HuggingFace model id, tokeniser id (when distinct), parameter
count estimate, default ``encoder_name`` used by training_v2,
licence, and notes on expected behaviour.

Loading actual weights happens in the trainer via
``transformers.AutoModel`` / ``AutoTokenizer``; this registry is
metadata-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class BackboneSpec:
    """Static description of one candidate backbone."""

    backbone_id:       str          # short kebab-case identifier
    hf_model_id:       str          # HuggingFace model hub id
    hf_tokenizer_id:   Optional[str] = None   # None → use hf_model_id
    arch_family:       str = ""     # "t5_enc_dec" / "bart_enc_dec" / "bert_enc" / ...
    arabic_pretraining: str = ""    # "msa" / "classical" / "multi_dialect" / "multilingual" / ...
    n_params_est:      str = ""     # human-readable "296M" / "1.2B"
    license:           str = ""
    notes:             str = ""


# ===========================================================================
# Backbone registry
# ===========================================================================

REGISTRY: Dict[str, BackboneSpec] = {

    # ---- Frozen-baseline reference ----------------------------------------
    "arat5v2-base": BackboneSpec(
        backbone_id="arat5v2-base",
        hf_model_id="UBC-NLP/AraT5v2-base-1024",
        arch_family="t5_enc_dec",
        arabic_pretraining="msa",
        n_params_est="296M",
        license="cc-by-nc-sa-4.0",
        notes="Frozen-baseline reference. Phase 3-A's encoder.",
    ),

    # ---- AraT5 family scale-up --------------------------------------------
    "arat5v2-large": BackboneSpec(
        backbone_id="arat5v2-large",
        hf_model_id="UBC-NLP/AraT5v2-large-1024",
        arch_family="t5_enc_dec",
        arabic_pretraining="msa",
        n_params_est="770M",
        license="cc-by-nc-sa-4.0",
        notes="Scale-up within same family. Tests within-family capacity.",
    ),

    # ---- AraBART (multi-dialect encoder-decoder) --------------------------
    "arabart-base": BackboneSpec(
        backbone_id="arabart-base",
        hf_model_id="moussaKam/AraBART",
        arch_family="bart_enc_dec",
        arabic_pretraining="multi_dialect",
        n_params_est="139M",
        license="apache-2.0",
        notes="Encoder-decoder, multi-dialect Arabic pretraining. "
              "Different architecture family from AraT5.",
    ),

    # ---- CAMeL-BERT family ------------------------------------------------
    "camelbert-msa": BackboneSpec(
        backbone_id="camelbert-msa",
        hf_model_id="CAMeL-Lab/bert-base-arabic-camelbert-msa",
        arch_family="bert_enc",
        arabic_pretraining="msa",
        n_params_est="135M",
        license="apache-2.0",
        notes="Encoder-only MSA. Head-to-head against AraT5v2 on same "
              "register.",
    ),
    "camelbert-ca": BackboneSpec(
        backbone_id="camelbert-ca",
        hf_model_id="CAMeL-Lab/bert-base-arabic-camelbert-ca",
        arch_family="bert_enc",
        arabic_pretraining="classical",
        n_params_est="135M",
        license="apache-2.0",
        notes="Classical-Arabic-specialised encoder. Expected to lift MASAQ "
              "Quranic role-F1 most.",
    ),
    "camelbert-mix": BackboneSpec(
        backbone_id="camelbert-mix",
        hf_model_id="CAMeL-Lab/bert-base-arabic-camelbert-mix",
        arch_family="bert_enc",
        arabic_pretraining="msa+classical+da",
        n_params_est="135M",
        license="apache-2.0",
        notes="Mixed-register pretraining. Tests whether a single "
              "multi-register model out-performs register-specialised ones.",
    ),

    # ---- AraBERT --------------------------------------------------------
    "arabert-large-v02": BackboneSpec(
        backbone_id="arabert-large-v02",
        hf_model_id="aubmindlab/bert-large-arabertv02",
        arch_family="bert_enc",
        arabic_pretraining="msa",
        n_params_est="370M",
        license="custom (aubmindlab)",
        notes="Larger encoder-only MSA. Tests encoder scaling within "
              "the BERT-style family.",
    ),

    # ---- Multilingual T5 -------------------------------------------------
    "mt5-base": BackboneSpec(
        backbone_id="mt5-base",
        hf_model_id="google/mt5-base",
        arch_family="t5_enc_dec",
        arabic_pretraining="multilingual",
        n_params_est="580M",
        license="apache-2.0",
        notes="Reference for multilingual vs Arabic-specific pretraining.",
    ),
    "mt5-large": BackboneSpec(
        backbone_id="mt5-large",
        hf_model_id="google/mt5-large",
        arch_family="t5_enc_dec",
        arabic_pretraining="multilingual",
        n_params_est="1.2B",
        license="apache-2.0",
        notes="Multilingual scale-up; pairs with mt5-base for capacity test.",
    ),

    # ---- Long-context encoder-only ---------------------------------------
    "xlm-r-base": BackboneSpec(
        backbone_id="xlm-r-base",
        hf_model_id="xlm-roberta-base",
        arch_family="xlm_roberta_enc",
        arabic_pretraining="multilingual",
        n_params_est="270M",
        license="mit",
        notes="Multilingual encoder reference. Long-range attention is "
              "limited to 512; not a true long-context candidate but "
              "useful as multilingual encoder baseline.",
    ),
}


# ===========================================================================
# Public API
# ===========================================================================

def get_backbone(backbone_id: str) -> BackboneSpec:
    if backbone_id not in REGISTRY:
        raise KeyError(
            f"unknown backbone {backbone_id!r}; "
            f"registered: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[backbone_id]


def all_backbones() -> List[BackboneSpec]:
    return list(REGISTRY.values())


def by_arch_family(arch_family: str) -> List[BackboneSpec]:
    return [b for b in REGISTRY.values() if b.arch_family == arch_family]


def by_pretraining(pretraining: str) -> List[BackboneSpec]:
    return [b for b in REGISTRY.values() if b.arabic_pretraining == pretraining]
