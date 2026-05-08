"""Qualitative trace rendering for the structured i'rāb predictor.

Per-word table that surfaces:

* the model's structured prediction
* per-head softmax confidence
* which symbolic constraints fired
* the rendered Arabic prose
* the gold prose if a gold reference is available

Two output formats are supported:

* ``render_markdown``  — used in REPORT.md and pasted into READMEs
* ``render_latex``     — used in the appendix table of the paper PDF

Both consume :class:`SentenceIrab` predictions and an optional list of gold
prose strings (one per word) for side-by-side display.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from ..structured.word_irab import SentenceIrab, WordIrab
from .template_renderer import render_word


def _conf_str(c: Optional[float]) -> str:
    if c is None:
        return "—"
    return f"{c:.2f}"


def _fmt_constraints(names: Sequence[str]) -> str:
    if not names:
        return ""
    # Compact display: "prep→jarr; inna ism→nasb"
    pretty = {
        "prep_to_jarr":       "prep→jarr",
        "inna_ism_to_nasb":   "inna ism→nasb",
        "inna_khabar_to_raf": "inna khabar→raf",
        "kana_ism_to_raf":    "kana ism→raf",
        "kana_khabar_to_nasb": "kana khabar→nasb",
        "idafa_stub":         "iḍāfa",
    }
    return "; ".join(pretty.get(n, n) for n in names)


def render_markdown(
    sent: SentenceIrab,
    gold_prose: Optional[Sequence[str]] = None,
    *,
    show_confidence: bool = True,
    show_constraints: bool = True,
    title: Optional[str] = None,
) -> str:
    """Render a single sentence's predictions as a Markdown table."""
    lines: List[str] = []
    if title:
        lines.append(f"### {title}\n")
    lines.append(f"**Sentence:** `{sent.sentence}`\n")
    header = ["#", "word", "case", "role", "marker", "POS"]
    if show_confidence:
        header.append("conf")
    if show_constraints:
        header.append("constraints")
    header += ["rendered i'rāb"]
    if gold_prose is not None:
        header.append("gold i'rāb")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")

    for i, w in enumerate(sent.items):
        row = [str(i + 1), w.word, w.case or "—", w.role or "—",
               w.marker or "—", w.pos or "—"]
        if show_confidence:
            min_c = w.min_confidence()
            row.append(_conf_str(min_c))
        if show_constraints:
            row.append(_fmt_constraints(w.constraints_fired))
        rendered = w.irab_prose or render_word(w)
        row.append(rendered or "—")
        if gold_prose is not None:
            g = gold_prose[i] if i < len(gold_prose) else ""
            row.append(g or "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_latex(
    sent: SentenceIrab,
    gold_prose: Optional[Sequence[str]] = None,
    *,
    title: Optional[str] = None,
) -> str:
    """Render a single sentence's predictions as a LaTeX longtable.

    Uses the standard ``arabxetex`` / ``polyglossia`` Arabic typesetting; the
    paper template already loads xelatex with the Amiri font.
    """
    cols = "rlllllll"
    has_gold = gold_prose is not None
    if has_gold:
        cols += "l"
    lines: List[str] = []
    if title:
        lines.append(f"\\paragraph*{{{title}}}\\mbox{{}}\\\\")
    lines.append("\\begin{tabular}{" + cols + "}")
    header = ["#", "word", "case", "role", "marker", "POS", "min-conf", "constraints"]
    if has_gold:
        header += ["gold i'rāb"]
    lines.append(" & ".join(header) + " \\\\\\midrule")
    for i, w in enumerate(sent.items):
        row = [str(i + 1), f"\\textarabic{{{w.word}}}", w.case or "—", w.role or "—",
               w.marker or "—", w.pos or "—",
               _conf_str(w.min_confidence()),
               _fmt_constraints(w.constraints_fired) or "—"]
        if has_gold:
            g = gold_prose[i] if i < len(gold_prose) else ""
            row.append(f"\\textarabic{{{g}}}" if g else "—")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def render_qualitative_set(
    sentences: Iterable[SentenceIrab],
    gold_proses: Optional[Sequence[Sequence[str]]] = None,
    *,
    fmt: str = "markdown",
    titles: Optional[Sequence[str]] = None,
) -> str:
    """Render multiple sentences back-to-back."""
    out: List[str] = []
    g_iter = list(gold_proses) if gold_proses is not None else None
    titles_list = list(titles) if titles else [None] * 1024
    for i, s in enumerate(sentences):
        gold = g_iter[i] if g_iter is not None and i < len(g_iter) else None
        title = titles_list[i] if i < len(titles_list) else None
        if fmt == "markdown":
            out.append(render_markdown(s, gold, title=title))
        elif fmt == "latex":
            out.append(render_latex(s, gold, title=title))
        else:
            raise ValueError(f"Unknown fmt={fmt}")
    return "\n".join(out)
