"""Shared visual language for the figures / Şekiller için ortak görsel dil:
navy title + grey subtitle, watermark = orange, structure/labels = navy, control = grey.

Where two data sources (generation-time / post-hoc) must be told apart:
  generation-time data → orange tones (the watermark colour)
  post-hoc data        → blue tones
  control              → grey
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from analyze.i18n import L

NAVY = "#1B3A5C"
ORANGE = "#E07B00"
SLATE = "#6B7785"
GREY = "#8C95A3"
GREY_L = "#ECEFF3"
PANEL = "#F5F8FB"
ORANGE_L = "#FDEBD3"
BLUE = "#2F6DA8"
GRID = "#D9DEE5"

# ρ = 10%, 50%, 100% (light → dark)
GEN = ["#F2B35E", "#E8901C", "#B85E00"]          # generation-time data (orange)
POST = ["#8DB3DA", "#4F86BD", "#1F4E86"]         # post-hoc data (blue)

THR2, THR6 = 2.0, 6.0                              # p = 0.01, p = 10⁻⁶ (on the −log₁₀ scale)

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 13,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": NAVY,
    "axes.titlecolor": NAVY,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
})

W = 9.2          # inches; 1380 px at 150 dpi
DPI = 200


def frame(height, title, subtitle):
    """Figure with a title + subtitle strip. Returns `top` = the upper bound (figure fraction) usable by axes."""
    fig = plt.figure(figsize=(W, height), facecolor="white")
    fig.text(0.025, 1 - 0.30 / height, title, fontsize=21, fontweight="bold", color=NAVY,
             ha="left", va="center")
    fig.text(0.025, 1 - 0.78 / height, subtitle, fontsize=13, color=SLATE, ha="left", va="center")
    top = 1 - 1.15 / height
    return fig, top


def footnote(fig, text, y=0.012):
    fig.text(0.025, y, text, fontsize=10, color=SLATE, ha="left", va="bottom")


def save(fig, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=DPI, facecolor="white")
    fig.savefig(outdir / f"{name}.svg", facecolor="white")
    plt.close(fig)


def chance_band(ax, hi=THR2, lo=0.0, label=True, xtext=None, fs=10.5):
    ax.axhspan(lo, hi, color=GREY_L, zorder=0)
    if label:
        x = xtext if xtext is not None else 0.99
        ax.text(x, (lo + hi) / 2 if lo > 0 else hi * 0.5, L("şans bandı", "chance band"), transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=fs, color=SLATE)


def thresholds(ax, x_text=0.012, fs=10.5):
    ax.axhline(THR2, color="#7A8696", lw=1.0, ls=(0, (2, 2)), zorder=1)
    ax.axhline(THR6, color=ORANGE, lw=1.1, ls=(0, (5, 3)), zorder=1)
    ax.text(x_text, THR2, "p = 0.01 ", transform=ax.get_yaxis_transform(), fontsize=fs, color="#5C6877",
            va="bottom")
    ax.text(x_text, THR6, "p = 10⁻⁶ ", transform=ax.get_yaxis_transform(), fontsize=fs, color=ORANGE,
            va="bottom")


def rbox(ax, x, y, w, h, fc="white", ec=NAVY, lw=1.6, r=0.012, ls="-", z=2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       ls=ls, zorder=z, alpha=alpha, transform=ax.transData)
    ax.add_patch(p)
    return p
