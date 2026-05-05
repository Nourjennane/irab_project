"""Generate the three paper figures from prediction JSONLs.

  fig_capacity.pdf       — open-weight scaling null + headline reference lines
  fig_cross_register.pdf — Gazelle vs MASAQ subset role-F1 with two-sample CIs
  fig_per_construction.pdf — fully-correct rate per Gazelle construction tag

Run from repo root:
  PYTHONPATH=. .venv/bin/python docs/paper/generate_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1 — capacity null + closed-system reference
# ---------------------------------------------------------------------------
def fig_capacity():
    sysm = [
        ("AraT5v2-base\n(296M)", 0.296, 24.6, 17.9, 32.8, "#3b82f6"),
        ("mT5-base\n(580M)",     0.580, 18.7, 11.9, 25.4, "#94a3b8"),
        ("AraGPT2-large\n(792M)",0.792, 26.1, 19.4, 34.3, "#3b82f6"),
        ("AceGPT-13B\n(13B)",    13.0,  25.4, 17.9, 32.8, "#3b82f6"),
    ]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for name, params, fully, lo, hi, color in sysm:
        ax.errorbar(params, fully, yerr=[[fully-lo],[hi-fully]],
                    fmt='o', color=color, ecolor=color, capsize=4,
                    markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        ax.annotate(name, (params, fully), textcoords="offset points",
                    xytext=(8, -2), fontsize=8)

    # Sonnet RAG headline
    ax.axhline(32.1, color="#dc2626", ls="--", lw=1.5,
               label="Claude Sonnet 4.5 + RAG (closed)")
    ax.fill_between([0.1, 30], 24.6, 40.3, color="#dc2626", alpha=0.07)
    # Stanza UD baseline
    ax.axhline(5.2, color="#6b7280", ls=":", lw=1.2,
               label="Stanza Arabic UD baseline")

    ax.set_xscale("log")
    ax.set_xlim(0.2, 25)
    ax.set_ylim(0, 45)
    ax.set_xlabel("Parameters (billions, log scale)")
    ax.set_ylabel("fully-correct words on Gazelle (%)")
    ax.set_title("Capacity null: 44× scale-up adds nothing measurable\n(blue = Arabic-pretrained; grey = multilingual)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_capacity.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — cross-register drop
# ---------------------------------------------------------------------------
def fig_cross_register():
    # (system, gazelle_F1, masaq_F1, delta, ci_lo, ci_hi)
    data = [
        ("Stanza\n(UD baseline)", 10.4, 17.6, -7.2, -11.8,  +1.9),
        ("mT5-base FT",           32.8, 18.6, +14.2, +1.5, +26.3),
        ("AraT5v2-base FT",       58.9, 24.3, +34.7,+18.7, +48.9),
        ("AraGPT2-large FT",      58.1, 20.2, +37.9,+21.3, +53.3),
        ("AceGPT-13B FT*",        60.4, 22.2, +38.2,+22.8, +52.7),
        ("Sonnet RAG",            75.7, 14.1, +61.7,+48.8, +75.3),
        ("Sonnet zero-shot**",    78.1, 11.0, +67.0,+52.1, +78.0),
    ]
    names  = [d[0] for d in data]
    deltas = [d[3] for d in data]
    los    = [d[3]-d[4] for d in data]
    his    = [d[5]-d[3] for d in data]
    colors = ['#6b7280' if d[3] < 0 else '#3b82f6' for d in data]
    colors[-2] = '#dc2626'  # Sonnet RAG: highlight
    colors[-1] = '#f87171'  # Sonnet zero-shot

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.barh(names, deltas, xerr=[los, his],
                   color=colors, ecolor='black', capsize=3,
                   edgecolor='black', linewidth=0.4)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel("Cross-register Δ role-F1 (Gazelle − MASAQ subset, pp)")
    ax.set_title("Largest drop comes from the strongest system\n(★ = paired-significant; UD parser is register-stable)")
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Annotate ★ for paired-significant
    for i, (name, gaz, masaq, d, lo, hi) in enumerate(data):
        if (lo > 0 and hi > 0) or (lo < 0 and hi < 0):
            ax.text(d + (3 if d > 0 else -3), i, "★",
                    va='center', ha='left' if d>0 else 'right',
                    fontsize=11, color='black', fontweight='bold')

    fig.text(0.01, -0.03, "*partial MASAQ (n=210); **first 400 verses (n=657)",
             fontsize=7, color='#666')
    fig.tight_layout()
    fig.savefig(OUT / "fig_cross_register.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — per-construction breakdown (Gazelle, fully-correct rate)
# ---------------------------------------------------------------------------
def fig_per_construction():
    # rows = construction tag, cols = system
    tags = ["NOMINAL\n(n=18)", "VERBAL\n(n=61)", "PREPOSITIONAL\n(n=37)",
            "PARTICLE_MOOD\n(n=30)", "EXCEPTION\n(n=9)", "KANA_SISTERS\n(n=7)"]
    systems = ["Stanza", "AraT5v2-base FT", "AraGPT2-large", "AceGPT-13B FT", "Sonnet RAG"]
    # data[row][col] in %
    data = [
        [16.7, 50.0, 50.0, 50.0, 72.2],  # NOMINAL
        [ 3.3, 29.5, 29.5, 29.5, 32.8],  # VERBAL
        [ 8.1, 29.7, 35.1, 35.1, 29.7],  # PREPOSITIONAL
        [ 3.3, 10.0, 13.3, 13.3, 23.3],  # PARTICLE_MOOD
        [ 0.0,  0.0,  0.0,  0.0,  0.0],  # EXCEPTION
        [ 0.0,  0.0,  0.0,  0.0,  0.0],  # KANA_SISTERS
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    import numpy as np
    arr = np.array(data)
    im = ax.imshow(arr, cmap='YlGnBu', vmin=0, vmax=80, aspect='auto')

    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, rotation=25, ha='right', fontsize=8)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags, fontsize=8)
    ax.set_title("Per-construction fully-correct rate (%)\nEXCEPTION + KANA_SISTERS fail across all systems")

    # Annotate cells
    for i in range(len(tags)):
        for j in range(len(systems)):
            v = arr[i, j]
            color = 'white' if v > 40 else 'black'
            ax.text(j, i, f"{v:.0f}", ha='center', va='center',
                    fontsize=8, color=color,
                    fontweight='bold' if v == 0 else 'normal')

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("fully-correct (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_construction.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_capacity()
    print(f"wrote {OUT/'fig_capacity.pdf'}")
    fig_cross_register()
    print(f"wrote {OUT/'fig_cross_register.pdf'}")
    fig_per_construction()
    print(f"wrote {OUT/'fig_per_construction.pdf'}")


if __name__ == "__main__":
    main()
