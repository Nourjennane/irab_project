"""I'rāb-Guided Arabic Diacritization — Streamlit demo for HF Spaces.

Loads a trained model from HF Hub (or local path) and exposes the Predictor.
"""

import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch

# The package is pip-installed from pyproject.toml (see app/requirements.txt)
from irab_tashkeel.inference.predictor import Predictor


# -------- Page config --------
st.set_page_config(
    page_title="I'rāb-Guided Arabic Diacritization",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# RTL + Arabic font CSS
st.markdown("""
<style>
    .arabic {
        font-family: "Noto Naskh Arabic", "Amiri", "Cairo", serif;
        font-size: 26px;
        direction: rtl;
        text-align: right;
        line-height: 2;
    }
    .word-card {
        display: inline-block;
        margin: 6px;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)


# -------- Model loading --------
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "your-username/irab-tashkeel-model")
LOCAL_MODEL_CKPT = os.environ.get("MODEL_CKPT")  # e.g. runs/medium/best.pt


@st.cache_resource(show_spinner="Loading model …")
def load_predictor() -> Predictor:
    """Load Predictor from local path (MODEL_CKPT) or HF Hub (HF_MODEL_REPO)."""
    if LOCAL_MODEL_CKPT and Path(LOCAL_MODEL_CKPT).exists():
        return Predictor.from_checkpoint(LOCAL_MODEL_CKPT)

    # Fall back to HF Hub
    try:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id=HF_MODEL_REPO, filename="model.pt")
        return Predictor.from_checkpoint(ckpt)
    except Exception as e:
        st.error(
            f"Couldn't load model. Set $MODEL_CKPT to a local file, or "
            f"$HF_MODEL_REPO to your Hub repo. Error: {e}"
        )
        st.stop()


def role_color(role: str) -> str:
    palette = {
        "fiil":         "#FEEAE6",
        "N_marfu":      "#E1F5EE",
        "N_mansub":     "#FAEEDA",
        "ism_majrur":   "#EEEDFE",
        "mudaf_ilayh":  "#EEEDFE",
        "harf_jarr":    "#F1EFE8",
        "harf_atf":     "#F1EFE8",
        "harf_nafy":    "#F1EFE8",
        "mabni_noun":   "#FBEAF0",
        "other":        "#F7F7F9",
    }
    return palette.get(role, "#F7F7F9")


# -------- Sidebar --------
with st.sidebar:
    st.title("📖 I'rāb + Tashkīl")
    st.markdown("**Explainable Arabic diacritization**")
    st.markdown("---")
    st.markdown("""
A hybrid neural + rule-based system that:
- Adds diacritics (**tashkīl**)
- Labels each word's grammatical role (**i'rāb**)
- Detects orthographic and grammatical errors
- Explains each decision in Arabic & English
""")
    st.markdown("---")

    backend = st.radio(
        "Backend",
        ["Claude RAG (Yarob+distilled)", "Per-word decoder (offline)"],
        index=0,
        help=(
            "Claude RAG (best quality so far on Gazelle: 67.2% case / 68.8% role / "
            "44.8% marker). Needs ANTHROPIC_API_KEY in env. "
            "Per-word decoder is your trained baseline (32.8% case)."
        ),
    )
    use_claude = backend.startswith("Claude")
    if use_claude and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY not set — Claude RAG won't work. Restart Streamlit with the env var.")

    rag_k = 5
    if use_claude:
        rag_k = st.slider("RAG few-shot k", 1, 8, 5)

    st.markdown("---")
    show_confidence = st.checkbox("Show confidence scores", value=True)
    show_en = st.checkbox("Show English labels", value=True)
    conf_threshold = st.slider("Low-confidence flag threshold", 0.0, 1.0, 0.6, 0.05)

    st.markdown("---")
    st.markdown("[GitHub](https://github.com/) · Built with character-level Transformer + QAC + Tashkeela + I3rab")


# -------- Main --------
st.title("I'rāb-Guided Arabic Diacritization")
st.caption("Tashkīl with a why — every diacritic comes with a grammatical justification")

@st.cache_resource(show_spinner="Loading Yarob+distilled retrieval pool…")
def load_claude_rag_pool():
    from irab_tashkeel.inference.llm_baselines import load_combined_fewshots
    return load_combined_fewshots()


import re
import unicodedata as _ud

_DIAC = "ً-ْٰ"
_DIAC_RE = re.compile(rf"[{_DIAC}]")


def _strip_diacritics(s: str) -> str:
    return _DIAC_RE.sub("", _ud.normalize("NFC", s or ""))


def _compare_diacritics(user_marked: str, claude_marked: str):
    """Return (agrees: bool, user_last_diac: str, claude_last_diac: str).

    Compares only the *last surface diacritic* on each word — that's the case
    ending. Intra-word root diacritics are mostly invariant; we don't flag
    those.
    """
    def last_diac(w):
        # Walk from end; first non-diacritic letter; capture trailing diacritics
        diacs = []
        for ch in reversed(_ud.normalize("NFC", w)):
            if _DIAC_RE.fullmatch(ch):
                diacs.append(ch)
            else:
                break
        return "".join(reversed(diacs))
    u, c = last_diac(user_marked), last_diac(claude_marked)
    return (u == c), u, c


def _diac_name(d: str) -> str:
    """Arabic name for a diacritic glyph (for the explanation text)."""
    names = {
        "ُ": "الضمة",     "َ": "الفتحة",     "ِ": "الكسرة",
        "ٌ": "تنوين الضم","ً": "تنوين الفتح","ٍ": "تنوين الكسر",
        "ْ": "السكون",    "ّ": "الشدة",
    }
    if not d:
        return "بدون حركة"
    parts = [names.get(ch, ch) for ch in d]
    return " + ".join(parts)


_CASE_TO_VOWEL = {
    "rafʿ":  "ُ",
    "naṣb":  "َ",
    "jarr":  "ِ",
    "jazm":  "ْ",
}
_TANWIN_FOR_CASE = {
    "rafʿ":  "ٌ",
    "naṣb":  "ً",
    "jarr":  "ٍ",
}


def _enforce_case_on_diac(diac_word: str, case: str | None, marker: str | None) -> tuple[str, bool]:
    """If Claude's diacritized form contradicts its own case label, fix it.

    Returns (possibly_fixed, was_fixed_flag).

    Conservative: only touches the last surface diacritic. Skips:
      - mabni / unknown case (no surface vowel to enforce)
      - dual/plural-suffix markers (الياء، الواو، الألف) — those are special
        and not a single short-vowel.
      - words ending in ة (we still rewrite the vowel that follows the ة)
    """
    if not diac_word or not case:
        return diac_word, False
    if case not in _CASE_TO_VOWEL:
        return diac_word, False
    # If marker is one of the "special signs" (long-vowel forms), leave alone.
    if marker:
        m = marker.strip()
        if any(s in m for s in ("الواو", "الألف", "الياء", "النون", "حذف")):
            return diac_word, False

    base = _ud.normalize("NFC", diac_word)
    use_tanwin = bool(marker and "تنوين" in marker)
    target = _TANWIN_FOR_CASE[case] if (use_tanwin and case in _TANWIN_FOR_CASE) else _CASE_TO_VOWEL[case]

    # Walk from end, find the last non-diacritic letter, then peel off
    # any trailing diacritics on it (could be 0-2 chars: e.g. shadda+vowel).
    chars = list(base)
    if not chars:
        return diac_word, False
    end_diacs: list[str] = []
    while chars and _DIAC_RE.fullmatch(chars[-1]):
        end_diacs.append(chars.pop())
    end_diacs.reverse()
    if not chars:
        return diac_word, False
    # If there's a shadda at end-1, keep it; replace just the vowel after.
    keep_prefix = []
    has_shadda = False
    for d in end_diacs:
        if d == "ّ":
            keep_prefix.append(d)
            has_shadda = True
        else:
            # drop existing vowel/sukun/tanwin
            pass
    new_diacs = keep_prefix + [target]
    fixed = "".join(chars) + "".join(new_diacs)
    return fixed, (fixed != base)


def _explain_correction(user_marked: str, item) -> str:
    role = (item.role or "").strip()
    case = (item.case or "").strip()
    marker = (item.marker or "").strip()
    parts = []
    if role:
        parts.append(f"إعرابها: {role}")
    if marker:
        parts.append(f"وعلامتها {marker}")
    elif case:
        case_ar = {"rafʿ":"الرفع","naṣb":"النصب","jarr":"الجر","jazm":"الجزم","mabni":"البناء"}.get(case, case)
        parts.append(f"الحالة: {case_ar}")
    rationale = "، ".join(parts) if parts else "الإعراب أعلاه يوضح السبب"
    _, u_last, c_last = _compare_diacritics(user_marked, getattr(item, "diacritized", "") or "")
    return (
        f"كتبت آخرها بـ«{_diac_name(u_last)}»، "
        f"والصحيح أن تكون بـ«{_diac_name(c_last)}» — "
        f"لأن {rationale}."
    )


def claude_rag_predict(sentence, k, raw_user_input=None):
    """Adapter: shape Claude-RAG output into the same `result` shape the
    per-word renderer expects, AND attach a list of per-word corrections
    when the user's input had diacritics that disagree with Claude's.
    """
    from irab_tashkeel.inference.llm_baselines import claude_fewshot_rag
    from irab_tashkeel.data.schema import PredictionResult

    pool = load_claude_rag_pool()
    bare = _strip_diacritics(sentence).strip()
    items = claude_fewshot_rag(bare, pool, k=k, model="claude-haiku-4-5")

    # Tokenize the raw user input (preserving diacritics) to compare
    # against Claude's per-word output.
    user_words_marked = (raw_user_input or sentence).strip().split()
    corrections = []
    self_fixes = []  # cases where we overrode Claude's own diacritization
    words = []
    diacritized_parts = []
    for i, it in enumerate(items):
        raw_diac = getattr(it, "diacritized", None) or it.word
        # Self-consistency repair: if Claude's diacritization disagrees with
        # its own case label, override the last vowel using the case.
        fixed_diac, was_fixed = _enforce_case_on_diac(raw_diac, it.case, it.marker)
        if was_fixed:
            self_fixes.append({
                "index": i,
                "claude_form": raw_diac,
                "fixed_form":  fixed_diac,
                "case": it.case,
                "role": it.role,
            })
        word_diac = fixed_diac
        diacritized_parts.append(word_diac)

        # Compare with the user's typed diacritization for this word, if any.
        u_marked = user_words_marked[i] if i < len(user_words_marked) else ""
        had_user_diac = bool(_DIAC_RE.search(u_marked))
        if had_user_diac:
            agrees, u_last, c_last = _compare_diacritics(u_marked, word_diac)
            if not agrees:
                corrections.append({
                    "index": i,
                    "user_form":   u_marked,
                    "fixed_form":  word_diac,
                    "explanation": _explain_correction(u_marked, it),
                })

        words.append({
            "index": i,
            "surface": it.word,
            "diacritized": word_diac,
            "role": it.role or "other",
            "role_ar": it.role or "—",
            "role_en": (it.case or "—"),
            "irab_text": it.irab,
            "diac_confidence": 1.0,
            "irab_confidence": 1.0,
            "low_confidence": False,
        })

    result = PredictionResult(
        input_text=sentence,
        diacritized=" ".join(diacritized_parts),
        words=words, errors=[], tier=1, tier_flags=[],
    )
    # Stash corrections + self-fixes on the result for the renderer to pick up.
    result.corrections = corrections   # type: ignore[attr-defined]
    result.self_fixes = self_fixes     # type: ignore[attr-defined]
    return result


if not use_claude:
    predictor = load_predictor()
    predictor.confidence_threshold = conf_threshold
else:
    predictor = None

examples = {
    "Simple verbal sentence": "ذهب الطالب إلى المدرسة",
    "Nominal sentence":       "العلم نور والجهل ظلام",
    "Iḍāfa construction":     "كتاب الطالب جديد",
    "Sentence with errors":   "ذهب الطالب الى المدرسه",
    "With adjective":         "قرأت كتابا مفيدا عن اللغة العربية",
    "kāna sentence (Tier 2)": "كان الطالب مجتهدا في دروسه",
    "Relative clause (Tier 3)": "قرأت الكتاب الذي أعجبني",
    "Custom":                 "",
}

col_ex, col_in = st.columns([1, 3])
with col_ex:
    example_choice = st.selectbox("Load example", list(examples.keys()))
with col_in:
    default = examples[example_choice]
    user_input = st.text_area(
        "Enter Arabic text (undiacritized or partially diacritized)",
        value=default, height=80,
    )

if st.button("Diacritize + analyze", type="primary") and user_input.strip():
    with st.spinner("Running inference…"):
        if use_claude:
            result = claude_rag_predict(user_input, k=rag_k, raw_user_input=user_input)
        else:
            result = predictor.predict(user_input)

    # ---- Diacritized output ----
    st.subheader("Diacritized text")
    st.markdown(f"<div class='arabic'>{result.diacritized}</div>", unsafe_allow_html=True)

    # ---- Self-consistency repairs (model overrode its own diacritization) ----
    self_fixes = getattr(result, "self_fixes", None) or []
    if self_fixes:
        st.subheader(f"🔧 Self-consistency repairs ({len(self_fixes)})")
        st.caption(
            "Claude's diacritization disagreed with its own grammatical analysis. "
            "We override the last vowel using the case label."
        )
        for sf in self_fixes:
            st.markdown(
                f"<div style='direction:rtl; padding:10px; background:#fff8e1; "
                f"border-left:4px solid #f59e0b; border-radius:6px; color:#111; margin-bottom:8px'>"
                f"<span style='font-family:\"Noto Naskh Arabic\",serif; font-size:22px'>"
                f"<span style='text-decoration:line-through; color:#888'>{sf['claude_form']}</span> "
                f"&nbsp;⟶&nbsp; <b>{sf['fixed_form']}</b></span>"
                f"<div style='font-size:13px; color:#555; margin-top:4px'>"
                f"الإعراب: {sf.get('role','—')} (case={sf.get('case','?')})"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # ---- Diacritization corrections (Claude RAG only) ----
    corrections = getattr(result, "corrections", None) or []
    if corrections:
        st.subheader(f"📝 Diacritization corrections ({len(corrections)})")
        for c in corrections:
            with st.container():
                col_l, col_r = st.columns([1, 1])
                with col_l:
                    st.markdown(
                        f"<div style='direction:rtl; padding:8px; background:#fff5f5; "
                        f"border-left:4px solid #c44; border-radius:6px; color:#111'>"
                        f"<b>كتبت:</b><br>"
                        f"<span style='font-family:\"Noto Naskh Arabic\",serif; font-size:24px'>{c['user_form']}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col_r:
                    st.markdown(
                        f"<div style='direction:rtl; padding:8px; background:#f0fff4; "
                        f"border-left:4px solid #2a8; border-radius:6px; color:#111'>"
                        f"<b>الصواب:</b><br>"
                        f"<span style='font-family:\"Noto Naskh Arabic\",serif; font-size:24px'>{c['fixed_form']}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"<div style='direction:rtl; padding:8px; color:#444; "
                    f"font-family:\"Noto Naskh Arabic\",serif; font-size:14px; "
                    f"margin-bottom:12px'>{c['explanation']}</div>",
                    unsafe_allow_html=True,
                )

    # ---- Tier info ----
    if result.tier > 1:
        tier_messages = {
            2: "⚠️ Tier 2 detected (kāna/inna/jussive). Neural model handles case reassignment.",
            3: "⚠️ Tier 3 detected (relative/conditional). Neural model carries the full load.",
        }
        st.info(f"{tier_messages[result.tier]}  Flags: `{', '.join(result.tier_flags) or '—'}`")

    # ---- Errors ----
    if result.errors:
        neural_errors = [e for e in result.errors if e.get("start", -1) >= 0]
        ortho_errors = [e for e in result.errors if e.get("start", -1) < 0]

        if ortho_errors:
            st.subheader("🔧 Orthographic corrections (rule-based)")
            for err in ortho_errors:
                st.success(f"**{err['text']}** → **{err.get('corrected', '?')}** — {err['description']}")

        if neural_errors:
            st.subheader("⚠️ Detected errors (neural)")
            for err in neural_errors:
                st.warning(f"**{err['text']}** — {err['description']}")

    # ---- Per-word i'rab cards (grid, RTL) ----
    st.subheader("Per-word analysis")
    cols_per_row = 4
    words_rtl = list(reversed(result.words))
    for row_start in range(0, len(words_rtl), cols_per_row):
        cols = st.columns(cols_per_row)
        for ci, wi in enumerate(range(row_start, min(row_start + cols_per_row, len(words_rtl)))):
            w = words_rtl[wi]
            with cols[ci]:
                bg = role_color(w["role"])
                low_conf = w["irab_confidence"] < conf_threshold
                border = "#ff7f6e" if low_conf else "#e0e0e0"

                html = f"""
                <div style="background:{bg}; padding:12px; border-radius:8px;
                            border:1px solid {border}; direction:rtl; text-align:center; margin-bottom:8px; color:#111">
                  <div style="font-family:'Noto Naskh Arabic',serif; font-size:28px; font-weight:500; color:#111">
                    {w['diacritized']}
                  </div>
                  <div style="color:#444; font-size:14px; margin-top:6px">{w['role_ar']}</div>
                """
                if show_en:
                    html += f"<div style='color:#888; font-size:11px; direction:ltr'>{w['role_en']}</div>"
                irab_text = w.get("irab_text", "")
                if irab_text:
                    html += (
                        "<div style=\"margin-top:10px; padding:8px 6px; "
                        "background:rgba(0,0,0,0.04); border-radius:6px; "
                        "font-family:'Noto Naskh Arabic',serif; font-size:14px; "
                        "color:#222; line-height:1.6\">"
                        f"{irab_text}</div>"
                    )
                if show_confidence:
                    html += (
                        "<div style=\"margin-top:8px; font-size:10px; "
                        "color:#666; direction:ltr\">"
                        f"diac {w['diac_confidence']*100:.0f}% · "
                        f"irab {w['irab_confidence']*100:.0f}%</div>"
                    )
                if low_conf:
                    html += "<div style='color:#c44; font-size:10px; margin-top:4px'>⚠ low confidence</div>"
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

    # ---- Summary ----
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    avg_diac = np.mean([w["diac_confidence"] for w in result.words]) if result.words else 0
    avg_irab = np.mean([w["irab_confidence"] for w in result.words]) if result.words else 0
    c1.metric("Words", len(result.words))
    c2.metric("Avg diac conf", f"{avg_diac*100:.1f}%")
    c3.metric("Avg i'rab conf", f"{avg_irab*100:.1f}%")
    c4.metric("Tier", result.tier)

    # ---- Raw JSON (collapsible) ----
    with st.expander("Raw model output (JSON)"):
        st.json(result.to_json())

# Footer
st.markdown("---")
st.caption(
    "Trained on QAC + Tashkeela + I3rab + synthetic errors. "
    "Model is approximate; i'rāb labels are coarse-grained (11 classes)."
)
