"""Canonical reasoning templates per construction family.

Each :class:`ReasoningTemplate` carries the canonical Arabic
grammatical justification for a construction, the derivation
chain (which named rules fire), the transformation logic
(case/role pattern), and pointers to common semantic + discourse
disambiguation notes.

Placeholders use ``{name}`` syntax and are substituted by
:mod:`reasoning.populator` at sentence-fill time.

Available placeholders
----------------------

- ``{particle_surface}``  — surface form of the construction's particle
- ``{ism_word}``          — surface of the ism (token at span pos 1)
- ``{khabar_word}``       — surface of the khabar (token at span pos 2)
- ``{head_noun}``         — for iḍāfa: head (muḍāf) surface
- ``{dependent_noun}``    — for iḍāfa: muḍāf-ilayh surface
- ``{target_word}``       — for istithnāʾ / mawṣūl: the word the
                             construction targets

Templates are deliberately bilingual (Arabic prose for the
justification + English-keyed transformation_logic) so the model
can be trained against either form.

These templates are *bronze_template* quality — high-confidence
rule statements without sentence-specific reasoning. The
ingestor (Step 9.2) will overlay textbook-grade traces from
external corpora when available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ReasoningTemplate:
    """A canonical reasoning template for one construction family."""
    family:                   str
    subgroup:                 str
    name:                     str             # short identifier
    transformation_logic:     str             # canonical English-keyed rule
    justification:            str             # Arabic prose with placeholders
    derivation_chain:         List[str] = field(default_factory=list)
    semantic_disambiguation:  str = ""
    discourse_notes:          str = ""
    confidence:               float = 0.95


# ===========================================================================
# Templates
# ===========================================================================

# Order matters only for documentation; lookup is by (family, subgroup) tuple.

ALL_TEMPLATES: List[ReasoningTemplate] = [
    # --- kāna sisters --------------------------------------------------------
    ReasoningTemplate(
        family="kana_sisters", subgroup="kana_completion",
        name="kana_completion_canonical",
        transformation_logic="kana_completion: particle→fil(mabni); ism→raf; khabar→nasb",
        justification=(
            "{particle_surface}: فعل ماضٍ ناقص من أخوات كان، مبني على الفتح. "
            "{ism_word}: اسم {particle_surface} مرفوع. "
            "{khabar_word}: خبر {particle_surface} منصوب."
        ),
        derivation_chain=[
            "rule:kana_family_particle",
            "rule:ism_kana_takes_raf",
            "rule:khabar_kana_takes_nasb",
        ],
        semantic_disambiguation=(
            "إذا كانت الجملة الاسمية بعد {particle_surface} مكونة من مبتدأ "
            "وخبر، فهي في محل نصب خبر {particle_surface}."
        ),
    ),
    ReasoningTemplate(
        family="kana_sisters", subgroup="kana_negation",
        name="kana_negation_canonical",
        transformation_logic="kana_negation: particle→fil(mabni); ism→raf; khabar→nasb",
        justification=(
            "{particle_surface}: فعل ماضٍ ناقص جامد من أخوات كان (ناف للجنس)، "
            "مبني على الفتح. {ism_word}: اسم {particle_surface} مرفوع. "
            "{khabar_word}: خبر {particle_surface} منصوب."
        ),
        derivation_chain=[
            "rule:kana_negation_particle",
            "rule:ism_kana_takes_raf",
            "rule:khabar_kana_takes_nasb",
        ],
    ),

    # --- inna sisters --------------------------------------------------------
    ReasoningTemplate(
        family="inna_sisters", subgroup="inna_assertion",
        name="inna_assertion_canonical",
        transformation_logic="inna_assertion: particle→harf_nasb(mabni); ism→nasb; khabar→raf",
        justification=(
            "{particle_surface}: حرف توكيد ونصب من أخوات إن، مبني على الفتح. "
            "{ism_word}: اسم {particle_surface} منصوب. "
            "{khabar_word}: خبر {particle_surface} مرفوع."
        ),
        derivation_chain=[
            "rule:inna_family_particle",
            "rule:ism_inna_takes_nasb",
            "rule:khabar_inna_takes_raf",
        ],
    ),
    ReasoningTemplate(
        family="inna_sisters", subgroup="inna_modal",
        name="inna_modal_canonical",
        transformation_logic="inna_modal: particle→harf_other(mabni); ism→nasb; khabar→raf",
        justification=(
            "{particle_surface}: حرف من أخوات إن (مشبه بالفعل)، مبني. "
            "{ism_word}: اسمها منصوب. {khabar_word}: خبرها مرفوع."
        ),
        derivation_chain=[
            "rule:inna_modal_particle",
            "rule:ism_inna_takes_nasb",
            "rule:khabar_inna_takes_raf",
        ],
    ),

    # --- istithnāʾ -----------------------------------------------------------
    ReasoningTemplate(
        family="istithna", subgroup="illa",
        name="illa_canonical",
        transformation_logic="istithna_illa: particle→harf_other(mabni); mustathna→nasb",
        justification=(
            "{particle_surface}: حرف استثناء مبني على السكون. "
            "{target_word}: مستثنى منصوب."
        ),
        derivation_chain=[
            "rule:illa_particle",
            "rule:mustathna_takes_nasb_in_positive_clause",
        ],
        semantic_disambiguation=(
            "إذا كان الاستثناء منقطعاً (المستثنى ليس من جنس المستثنى منه) "
            "فالمستثنى واجب النصب. وإذا كان متصلاً وكان الكلام مثبتاً فالمستثنى "
            "أيضاً واجب النصب."
        ),
    ),
    ReasoningTemplate(
        family="istithna", subgroup="istithna_noun",
        name="istithna_noun_canonical",
        transformation_logic="istithna_noun: particle→nasb(mustathna); next→jarr(mudaaf_ilayh)",
        justification=(
            "{particle_surface}: مستثنى منصوب وهو مضاف. "
            "{target_word}: مضاف إليه مجرور."
        ),
        derivation_chain=[
            "rule:istithna_via_noun",
            "rule:noun_takes_nasb_as_mustathna",
            "rule:idafa_governs_jarr",
        ],
    ),

    # --- mawṣūl --------------------------------------------------------------
    ReasoningTemplate(
        family="mawsool", subgroup="definite_relative",
        name="definite_mawsool_canonical",
        transformation_logic="mawsool_definite: pronoun→mabni; clause→relative",
        justification=(
            "{particle_surface}: اسم موصول مبني، له صلة موصول من جملة فعلية "
            "أو اسمية. محله الإعرابي يتبع موقعه في الجملة الكبرى."
        ),
        derivation_chain=[
            "rule:definite_relative_is_mabni",
            "rule:mawsool_inherits_case_from_governing_role",
        ],
        semantic_disambiguation=(
            "محل {particle_surface} الإعرابي يتحدد من موقعه: قد يكون فاعلاً، "
            "مفعولاً، نعتاً، أو خبراً، حسب ما يطلبه السياق."
        ),
    ),
    ReasoningTemplate(
        family="mawsool", subgroup="indefinite_relative",
        name="indefinite_mawsool_canonical",
        transformation_logic="mawsool_indefinite: pronoun→mabni; clause→relative",
        justification=(
            "{particle_surface}: اسم موصول عام (نكرة)، مبني، صلته جملة. "
            "محله الإعرابي حسب موقعه."
        ),
        derivation_chain=[
            "rule:indefinite_relative_is_mabni",
            "rule:mawsool_inherits_case_from_governing_role",
        ],
    ),

    # --- iḍāfa ---------------------------------------------------------------
    ReasoningTemplate(
        family="idafa", subgroup="any",
        name="idafa_canonical",
        transformation_logic="idafa: head→mudaaf; dependent→jarr(mudaaf_ilayh)",
        justification=(
            "{head_noun}: مضاف. {dependent_noun}: مضاف إليه مجرور."
        ),
        derivation_chain=[
            "rule:idafa_construction",
            "rule:mudaaf_ilayh_takes_jarr",
        ],
    ),
    ReasoningTemplate(
        family="idafa_multi", subgroup="any",
        name="idafa_multi_canonical",
        transformation_logic="idafa_multi: chain of mudaaf↔mudaaf_ilayh, all middle nouns are mudaaf",
        justification=(
            "هذه إضافة متعددة. {head_noun}: مضاف، وما يليه مضاف إليه ومضاف "
            "في آنٍ واحد، حتى الكلمة الأخيرة من السلسلة، التي تكون مضافاً "
            "إليه فقط."
        ),
        derivation_chain=[
            "rule:idafa_construction",
            "rule:multi_level_idafa_chain",
            "rule:mudaaf_ilayh_takes_jarr",
        ],
    ),

    # --- quranic_proxy -------------------------------------------------------
    ReasoningTemplate(
        family="quranic_proxy", subgroup="qad_idh",
        name="qad_idh_canonical",
        transformation_logic="qad_idh: particle→harf_tahqiq/dharf(mabni)",
        justification=(
            "{particle_surface}: حرف تحقيق (إذا أتى قبل ماضٍ) أو حرف توقع "
            "(إذا أتى قبل مضارع)، مبني."
        ),
        derivation_chain=[
            "rule:qad_idh_particle",
        ],
        discourse_notes=(
            "في القرآن، {particle_surface} يكثر استعماله للتحقيق والتأكيد، "
            "وقد يقع أيضاً ظرفاً زمانياً (إذ)."
        ),
    ),
    ReasoningTemplate(
        family="quranic_proxy", subgroup="lamma",
        name="lamma_canonical",
        transformation_logic="lamma: dharf_zaman/harf(mabni)",
        justification=(
            "{particle_surface}: ظرف زمان متضمن معنى الشرط، مبني، يستلزم "
            "فعلاً ماضياً تالياً."
        ),
        derivation_chain=[
            "rule:lamma_temporal",
            "rule:lamma_implies_past_verb",
        ],
    ),
    ReasoningTemplate(
        family="quranic_proxy", subgroup="kullama",
        name="kullama_canonical",
        transformation_logic="kullama: dharf_zaman_repetitive(mabni)",
        justification=(
            "{particle_surface}: ظرف زمان للتكرار، مبني، يدل على الاستمرار "
            "والتكرر."
        ),
        derivation_chain=[
            "rule:kullama_temporal_repetitive",
        ],
    ),
    ReasoningTemplate(
        family="quranic_proxy", subgroup="hatta",
        name="hatta_canonical",
        transformation_logic="hatta: harf_jarr_or_atf_or_ibtidaa",
        justification=(
            "{particle_surface}: حرف جر (يجر الاسم بعده)، أو حرف عطف (يدل "
            "على الغاية)، أو حرف ابتداء، حسب السياق."
        ),
        derivation_chain=[
            "rule:hatta_polysemy",
        ],
        semantic_disambiguation=(
            "تمييز نوع {particle_surface} يعتمد على ما يأتي بعدها (اسم، فعل، "
            "أو جملة) وعلى المعنى المراد."
        ),
    ),
]


_INDEX: Dict[Tuple[str, str], ReasoningTemplate] = {
    (t.family, t.subgroup): t for t in ALL_TEMPLATES
}


def get_template(family: str, subgroup: str) -> Optional[ReasoningTemplate]:
    """Return the canonical template for ``(family, subgroup)`` or
    ``None`` when no template is registered.

    Falls back to ``(family, "any")`` for families that don't
    distinguish subgroups (e.g. iḍāfa).
    """
    t = _INDEX.get((family, subgroup))
    if t is not None:
        return t
    return _INDEX.get((family, "any"))


def supported_families() -> List[str]:
    return sorted({t.family for t in ALL_TEMPLATES})
