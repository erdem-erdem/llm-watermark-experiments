"""Blog şekilleri / Blog figures — group B: E1.2 (before/after), E3.1, E4.1, 5.0 (chain), opening mini-demo.

  python -m analyze.blog_figures_b                       # all figures, both languages → figures/{tr,en}/blog/
  python -m analyze.blog_figures_b --lang en E3_1 E4_1   # selected figures, English only

All values come from real result files in results/ (no estimates); a missing input raises an error.
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from analyze import blog_tokens as bt
from analyze import blog_values as bv
from analyze.blog_style import (BLUE, GEN, GREY, GREY_L, GRID, NAVY, ORANGE, ORANGE_L, PANEL, POST, SLATE, THR2,
                                THR6, W, frame, save)
from analyze.i18n import L, fig_dir, get_lang, parse_lang_arg, set_lang
from analyze.blog_values import fnum

IN = bv.RES / "blog"        # data inputs (e1_examples.json …); figures go to figures/<lang>/main


def _pct(r):
    """'10' → '%10' (tr) / '10%' (en)."""
    return L(f"%{r}", f"{r}%")


def _thousands(x):
    """22793 → '22 793' (tr) / '22,793' (en)."""
    return f"{x:,.0f}".replace(",", L(" ", ","))


def _rho(name):
    """Contamination ratio of an A-* model from its name (A-gen-50 → 50)."""
    return int(name.rsplit("-", 1)[1]) if name.rsplit("-", 1)[-1].isdigit() else None


def _pts_axes(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W * 72)
    ax.set_ylim(fig.get_figheight() * 72, 0)
    ax.axis("off")
    return ax


def _fmt_p(nl):
    """−log₁₀p → 'p = …' biçimi."""
    if nl < 2:
        return f"p = {10 ** (-nl):.2f}"
    e = int(np.floor(nl))
    m = 10 ** (-(nl - e)) * 10
    if nl < 3:
        return f"p = {10 ** (-nl):.3f}"
    return f"p ≈ {m:.0f}×10⁻{str(e + 1).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}" if m >= 9.5 else \
        f"p ≈ {m:.1f}×10⁻{str(e + 1).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}"


def _colorbar(ax, x, y, w=110, h=9, fs=9):
    n = 40
    for i in range(n):
        ax.add_patch(Rectangle((x + i * w / n, y), w / n + 0.3, h, fc=bt.tok_color(4 * i / (n - 1)), ec="none"))
    ax.text(x - 5, y + h / 2, L("düşük skor", "low score"), fontsize=fs, color=SLATE, ha="right", va="center")
    ax.text(x + w + 5, y + h / 2, L("yüksek skor", "high score"), fontsize=fs, color=SLATE, ha="left", va="center")


# ============================================================================= E1.2
def fig_E1_2(V):
    d = bv.read_json(IN / "e1_examples.json")
    if d is None:
        raise FileNotFoundError("results/blog/e1_examples.json missing — run `python -m analyze.blog_examples` first")
    pairs = d["pairs"]
    scratch = plt.figure(figsize=(W, 3))
    m = bt.Measurer(scratch)
    fs, lh = 9.6, 13.6
    M, G = 16.0, 12.0
    colw = (W * 72 - 2 * M - G) / 2
    inner = colw - 18
    rows = []
    for p in pairs:
        left = bt.plain_tokens(p["source"])
        right = [t for t in p["tokens"] if not t["piece"].startswith("<|")]
        li, lhgt = bt.layout(m, left, inner, fs, lh)
        ri, rhgt = bt.layout(m, right, inner, fs, lh)
        rows.append((p, li, ri, max(lhgt, rhgt)))
    plt.close(scratch)
    head = 28.0       # kutu başlığı
    strip = 26.0
    gap = 26.0
    total = 1.2 * 72 + sum(24 + head + r[3] + 12 + strip + gap for r in rows) + 34
    H = total / 72
    fig, top = frame(H, L("Önce ve Sonra: Watermark'lanmış Yeniden Yazım", "Before and After: Watermarked Rewriting"),
                     L("Sağdaki metinde tokenlar watermark skoruna göre renklendirilmiştir (koyu turuncu = yüksek skor).",
                       "Tokens in the right-hand text are colored by watermark score (dark orange = high score)."))
    ax = _pts_axes(fig)
    y = 1.2 * 72
    names = {"high": L("Örnek A — yüksek anlam benzerliği", "Example A — high semantic similarity"),
             "mid": L("Örnek B — daha düşük anlam benzerliği", "Example B — lower semantic similarity")}
    for p, li, ri, h in rows:
        ax.text(M, y + 4, names[p["slot"]], fontsize=12.5, fontweight="bold", color=NAVY, va="top")
        y += 24
        bh = head + h + 12
        for ci, (title, items, fc, ec) in enumerate(((L("Özgün cevap (InstrucTurca)", "Original answer (InstrucTurca)"), li, "white", GREY),
                                                      (L("Watermark'lı yeniden yazım (4B, T = 1.2)",
                                                         "Watermarked rewrite (4B, T = 1.2)"), ri, "#FFFBF5", ORANGE))):
            x0 = M + ci * (colw + G)
            ax.add_patch(FancyBboxPatch((x0, y), colw, bh, boxstyle="round,pad=0,rounding_size=7", fc=fc, ec=ec, lw=1.6))
            ax.text(x0 + 9, y + 8, title, fontsize=10, fontweight="bold", color=NAVY if ci == 0 else ORANGE, va="top")
            bt.draw(ax, items, x0 + 9, y + head, fs, lh)
        y += bh + 6
        stat = (f"{L('anlam benzerliği', 'semantic similarity')} {p['similarity']:.2f}   ·   "
                f"{L('uzunluk oranı', 'length ratio')} {p['len_ratio']:.2f}   ·   "
                f"{L('puanlanan token', 'scored tokens')} N = {p['N']}   ·   −log₁₀(p) = {p['neglog10p']:.1f}")
        ax.add_patch(FancyBboxPatch((M, y + 3), W * 72 - 2 * M, strip - 6, boxstyle="round,pad=0,rounding_size=5",
                                    fc=PANEL, ec="none"))
        ax.text(W * 72 / 2, y + strip / 2, stat, fontsize=10.5, color=NAVY, ha="center", va="center", fontweight="bold")
        y += strip + gap
    _colorbar(ax, W * 72 - 215, y - gap + 2)
    save(fig, fig_dir("main"), "fig_E1_2")


# ============================================================================= E3.1
def fig_E3_1(V):
    e3 = bv.load_e3()
    fig, top = frame(5.9, L("Eğitimde Görülen Watermark: Kirlenme Oranı ve İz", "Watermark Seen in Training: Contamination Ratio and Trace"),
                     L("Öğrencinin, eğitimde gördüğü watermark'lı veri üzerinde okuma modu ile ölçülen izi; puanlanan token arttıkça.",
                       "Reading-mode trace of the watermarked training data, as scored tokens accumulate."))
    ax = fig.add_axes([0.105, 0.145, 0.585, top - 0.145 - 0.04])
    ymax = max(e3[k]["final"] for k in bv.E3_PARAMS) * 3.0
    ax.axhspan(0, THR2, color=GREY_L, zorder=0)
    ax.axhline(THR2, color="#7A8696", lw=1.0, ls=(0, (2, 2)))
    ax.axhline(THR6, color=ORANGE, lw=1.1, ls=(0, (5, 3)))
    order = ["A-post-100", "A-post-50", "A-post-10", "A-gen-100", "A-gen-50", "A-gen-10", "A-rew", "A0"]
    tone = {10: 0, 50: 1, 100: 2}
    ends = []
    for k in order:
        d = e3[k]
        n, y = d["curve_n"], np.maximum(d["curve_y"], 0.02)
        kind = "gen" if k.startswith("A-gen") else "post" if k.startswith("A-post") else k
        if kind in ("A0", "A-rew"):
            col, ls, lw = GREY, "-" if kind == "A0" else (0, (1, 1.6)), 2.0
        elif kind == "gen":
            col, ls, lw = GEN[tone[_rho(k)]], "-", 2.6
        else:
            col, ls, lw = POST[tone[_rho(k)]], (0, (5, 2.2)), 2.6
        ax.plot(n, y, color=col, ls=ls, lw=lw, zorder=3 if kind in ("gen", "post") else 2)
        if kind in ("gen", "post"):
            ends.append((d, col))
    for d, col in ends:
        val = d["final"]
        txt = _thousands(val) if val >= 100 else f"{val:.1f}"
        ax.annotate(txt, (d["curve_n"][-1], max(d["curve_y"][-1], 0.02)), xytext=(6, 0),
                    textcoords="offset points", fontsize=9.5, color=col, va="center", fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=10, linscale=0.9)
    ax.set_xlim(22, 6e6)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 2, 6, 10, 100, 1000, 10000][: 7 if ymax > 1.2e4 else 6])
    ax.set_yticklabels([str(t) if t < 1000 else _thousands(t) for t in [0, 2, 6, 10, 100, 1000, 10000]][: 7 if ymax > 1.2e4 else 6])
    ax.set_xticks([30, 100, 1000, 10000, 100000, 1000000])
    ax.set_xticklabels(["30", "100", _thousands(1000), L("10 bin", "10k"), L("100 bin", "100k"), L("1 milyon", "1M")])
    ax.minorticks_off()
    ax.text(5.6e6, 0.9, L("şans bandı", "chance band"), fontsize=10, color=SLATE, va="center", ha="right")
    ax.text(5.6e6, THR2 * 1.12, "p = 0.01", fontsize=10, color="#5C6877", va="bottom", ha="right")
    ax.text(5.6e6, THR6 * 1.1, "p = 10⁻⁶", fontsize=10, color=ORANGE, va="bottom", ha="right")
    ax.set_xlabel(L("puanlanan benzersiz token sayısı  (logaritmik eksen)", "number of unique scored tokens  (log axis)"), fontsize=12)
    ax.set_ylabel("−log₁₀(p)", fontsize=12)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    # lejant
    H = [Line2D([], [], color="none")]
    handles, labels = [], []
    def hdr(t):
        handles.append(Line2D([], [], color="none"))
        labels.append(t)
    hdr(L("Üretim verisi (düz)", "Generation-time data (solid)"))
    for i, r in enumerate(("%10", "%50", "%100")):
        handles.append(Line2D([], [], color=GEN[i], lw=2.6)); labels.append(f"ρ = {_pct(r[1:])}")
    hdr(L("Post-hoc veri (kesikli)", "Post-hoc data (dashed)"))
    for i, r in enumerate(("%10", "%50", "%100")):
        handles.append(Line2D([], [], color=POST[i], lw=2.6, ls=(0, (5, 2.2)))); labels.append(f"ρ = {_pct(r[1:])}")
    hdr(L("Kontroller (gri)", "Controls (grey)"))
    handles.append(Line2D([], [], color=GREY, lw=2.0)); labels.append(L("A₀ (hiç watermark'lı veri yok)", "A₀ (no watermarked data)"))
    handles.append(Line2D([], [], color=GREY, lw=2.0, ls=(0, (1, 1.6)))); labels.append(L("A-yeniden-yazım", "A-rewrite"))
    lg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.03, 1.02), frameon=False, fontsize=10.5,
                   handlelength=2.2, labelspacing=0.55)
    for t in lg.get_texts():
        if t.get_text().endswith(("(düz)", "(kesikli)", "(gri)", "(solid)", "(dashed)", "(grey)")):
            t.set_fontweight("bold")
            t.set_color(NAVY)
    fig.text(0.725, 0.145, L("Eğri sonundaki sayı: ρ'ya göre\ntüm eğitim verisinde ulaşılan −log₁₀(p).",
                             "Number at the end of each curve: −log₁₀(p)\nreached on all training data, per ρ."),
             fontsize=9.5, color=SLATE, va="bottom")
    save(fig, fig_dir("main"), "fig_E3_1")


# ============================================================================= E4.1
def fig_E4_1(V):
    e3 = bv.load_e3()
    e4 = bv.load_e4(e3)
    fig, top = frame(5.7, L("Sinyal Zinciri: İz Hangi Halkada Kayboluyor?", "The Signal Chain: At Which Link Does the Trace Vanish?"),
                     L("Dört halkanın her birinde ölçülen iz gücü, −log₁₀(p) (logaritmik eksen); gri = kontrol.",
                       "Trace strength measured at each of the four links, −log₁₀(p) (log axis); grey = control."))
    ax = fig.add_axes([0.105, 0.185, 0.875, top - 0.185 - 0.115])
    R3, R4 = e4["ring3"], e4["ring4"]
    groups = [
        (L("Halka 1", "Link 1"), L("Watermark'lı veri", "Watermarked data"),
         [(L("D_gen\n(metin)", "D_gen\n(text)"), e4["D_gen"], NAVY)]),
        (L("Halka 2", "Link 2"), L("Aracı model", "Intermediate model"),
         [(L("A-üretim-100\n(okuma)", "A-gen-100\n(reading)"), e4["A-gen-100"], NAVY)]),
        (L("Halka 3", "Link 3"), L("Aracının cevapları", "Intermediate's answers"),
         [("greedy", R3["agent_greedy"], GEN[2]), ("T = 0.7", R3["agent_sampled"], GEN[0]),
          (L("A₀ greedy\n(kontrol)", "A₀ greedy\n(control)"), R3["a0_greedy"], GREY)]),
        (L("Halka 4", "Link 4"), L("B öğrencisi", "Student B"),
         [("B-greedy", R4["B-greedy"], GEN[2]), (L("B-örnekleme", "B-sampling"), R4["B-sample"], GEN[0]),
          (L("B′\n(kontrol)", "B′\n(control)"), R4["B-control"], GREY)]),
    ]
    x = 0.0
    xs, labels, centers = [], [], []
    ax.axhspan(0.05, THR2, color=GREY_L, zorder=0)
    ax.axhline(THR2, color="#7A8696", lw=1.0, ls=(0, (2, 2)))
    ax.axhline(THR6, color=ORANGE, lw=1.1, ls=(0, (5, 3)))
    for gi, (g, sub, bars) in enumerate(groups):
        x0 = x
        for lab, d, col in bars:
            v = max(d["y"], 0.06)
            ax.bar(x, v, width=0.78, color=col, zorder=3, edgecolor="white", linewidth=1.0, alpha=1.0)
            lab_v = _thousands(d["y"]) if d["y"] >= 100 else f"{d['y']:.1f}"
            ax.text(x, v * 1.22, lab_v, ha="center", va="bottom", fontsize=10.5,
                    fontweight="bold", color=NAVY if col != GREY else SLATE, zorder=5)
            xs.append(x)
            labels.append(lab)
            x += 1.0
        centers.append((x0 + x - 1) / 2)
        if gi < len(groups) - 1:
            x += 0.75
    ax.set_yscale("log")
    ax.set_ylim(0.1, 8e4)
    ax.set_xlim(-0.7, x - 0.3)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.1, 1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["0.1", "1", "10", "100", _thousands(1000), _thousands(10000)])
    ax.minorticks_off()
    ax.set_ylabel("−log₁₀(p)", fontsize=12)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(x - 0.35, THR2 * 1.08, "p = 0.01", fontsize=9.5, color="#5C6877", ha="right", va="bottom")
    ax.text(x - 0.35, THR6 * 1.08, "p = 10⁻⁶", fontsize=9.5, color=ORANGE, ha="right", va="bottom")
    ax.text((xs[1] + xs[2]) / 2, 0.125, L("şans bandı", "chance\nband"), fontsize=9.5, color=SLATE, ha="center", va="bottom", linespacing=1.0)
    for (g, sub, _), c in zip(groups, centers):
        ax.text(c, 1.085, g, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=12, fontweight="bold", color=NAVY)
        ax.text(c, 1.02, sub, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10.5, color=SLATE)
    for a, b in zip(centers[:-1], centers[1:]):
        xm = (a + b) / 2 + (0.0 if abs(b - a) > 1.5 else 0.0)
        ax.axvline((max(xs_i for xs_i in xs if xs_i <= xm) + min(xs_i for xs_i in xs if xs_i > xm)) / 2, color=GRID, lw=1.0,
                   ls=(0, (2, 3)), zorder=0)
    save(fig, fig_dir("main"), "fig_E4_1")


# ============================================================================= 5.0 zincir
def _node(ax, x, y, w, h, title, sub=None, ec=NAVY, fc="white", dashed=False, lw=2.0, tcol=NAVY, fs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=lw,
                                ls=(0, (4, 2.5)) if dashed else "-", zorder=3))
    cy = y + h / 2 + (1.8 if sub else 0)
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=tcol, zorder=4, linespacing=1.15)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 3.4, sub, ha="center", va="center", fontsize=9.3, color=SLATE, zorder=4, linespacing=1.1)


def _arrow(ax, p0, p1, label=None, lpos=0.5, dy=2.4, col=SLATE, rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16, lw=2.0, color=col, zorder=2,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        lx = p0[0] + (p1[0] - p0[0]) * lpos
        ly = p0[1] + (p1[1] - p0[1]) * lpos + dy
        ax.text(lx, ly, label, ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=ORANGE, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))


def _qbadge(ax, cx, cy, txt=None, col=ORANGE):
    txt = txt or L("radyoaktif mi?", "radioactive?")
    ax.add_patch(FancyBboxPatch((cx - 8.2, cy - 2.2), 16.4, 4.4, boxstyle="round,pad=0,rounding_size=2.2", fc="white",
                                ec=col, lw=1.6, ls=(0, (3, 2)), zorder=4))
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=9.8, fontweight="bold", color=col, zorder=5)


def fig_5_0(V):
    H = 4.9
    fig, top = frame(H, L("Deney Zinciri: Veriden Öğrenciye, Öğrenciden Öğrenciye", "The Experiment Chain: From Data to Student to Student"),
                     L("Dört deney tek zincirin halkalarıdır; hepsi aynı 5.000 soruyu kullanır.",
                       "The four experiments are links of a single chain; all use the same 5,000 questions."))
    ax = fig.add_axes([0.01, 0.03, 0.98, top - 0.03 - 0.02])
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.text(0.5, 53.6, L("Deney 1–2: watermark'lı veri üret", "Experiments 1–2: produce watermarked data"), fontsize=10.5, color=SLATE, va="center")
    ax.text(60.0, 53.6, L("Deney 3: eğit", "Experiment 3: train"), fontsize=10.5, color=SLATE, va="center")
    ax.text(91.0, 53.6, L("Deney 4: aklama", "Experiment 4: laundering"), fontsize=10.5, color=SLATE, va="center")
    # düğümler
    _node(ax, 0.5, 20, 20, 15, L("Türkçe soru-\ncevap kümesi", "Turkish Q&A\ndataset"), L("InstrucTurca\n5.000 çift", "InstrucTurca\n5,000 pairs"), ec=NAVY, fc=PANEL, fs=11)
    _node(ax, 30, 36, 21, 12, L("Post-hoc\nwatermark'lı veri", "Post-hoc\nwatermarked data"), L("yeniden yazım (4B)", "rewriting (4B)"), ec=BLUE, fc="#EEF4FB", fs=11)
    _node(ax, 30, 5.5, 21, 12, L("Üretim sırasında\nwatermark'lı veri", "Generation-time\nwatermarked data"), L("9B, anahtarla", "9B, keyed"), ec=ORANGE, fc=ORANGE_L, fs=11)
    _node(ax, 61, 20, 17, 15, L("Öğrenci A", "Student A"), "Qwen3.5-2B", ec=ORANGE, fc="white", dashed=True)
    _node(ax, 92, 20, 15, 15, L("A'nın\ncevapları", "A's\nanswers"), L("watermark'sız", "unwatermarked"), ec=NAVY, fc="white", fs=11)
    _node(ax, 112, 20, 14, 15, L("Öğrenci B", "Student B"), "Qwen3.5-2B", ec=SLATE, fc="white", dashed=True, fs=10.5)
    # oklar
    _arrow(ax, (21.0, 30), (29.6, 41), L("Deney 1", "Exp. 1"), 0.40, dy=1.8)
    _arrow(ax, (21.0, 25), (29.6, 14), L("Deney 2", "Exp. 2"), 0.40, dy=-6.8)
    _arrow(ax, (51.4, 41), (60.6, 31.5), None)
    _arrow(ax, (51.4, 13), (60.6, 23.5), None)
    ax.text(56.0, 28.0, L("Deney 3", "Exp. 3"), fontsize=10.5, fontweight="bold", color=ORANGE, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"), zorder=6)
    _arrow(ax, (78.4, 27.5), (91.6, 27.5), None)
    ax.text(85.0, 29.2, L("cevap üret", "generate"), fontsize=9.5, color=ORANGE, ha="center", va="bottom", fontweight="bold")
    _arrow(ax, (107.4, 27.5), (111.6, 27.5), None)
    ax.text(109.5, 29.2, L("eğit", "train"), fontsize=9.5, color=ORANGE, ha="center", va="bottom", fontweight="bold")
    # radyoaktif mi?
    _qbadge(ax, 69.5, 15.6)
    _qbadge(ax, 118.5, 15.6, col=SLATE)
    ax.text(69.5, 11.3, L("Deney 3'te ölçülür", "measured in Exp. 3"), fontsize=9, color=SLATE, ha="center")
    ax.text(118.5, 11.3, L("Deney 4'te ölçülür", "measured in Exp. 4"), fontsize=9, color=SLATE, ha="center")
    save(fig, fig_dir("main"), "fig_5_0")


# ============================================================================= mini: opening
def fig_M_acilis(V):
    VARIANT = "v0"          # fixed in advance: the first (default) instruction; not picked by looking at results
    toks_rows = bv.read_csv(bv.RES / "mini" / "opening_tokens.csv")
    curve_rows = bv.read_csv(bv.RES / "mini" / "opening_curve.csv")
    if not toks_rows or not curve_rows:
        raise FileNotFoundError("results/mini/opening_tokens.csv / opening_curve.csv missing — "
                                "run `python -m gen.gen_opening` (needs a GPU) to produce them")
    items, rng_all = [], {}
    for seg, name in (("soz", L("Açılış sözü", "Opening line")), ("paragraf", L("İlk paragraf", "First paragraph"))):
        rows = [r for r in toks_rows if r["variant"] == VARIANT and r["segment"] == seg]
        n_sc = sum(1 for r in rows if r["scored"] == "1")      # N, special tokens included, as recorded
        toks = [{"piece": r["piece"], "score": (bv.fnum(r["score"]) if r["scored"] == "1" else None)}
                for r in rows if not r["piece"].startswith("<|")]
        cur = [r for r in curve_rows if r["variant"] == VARIANT and r["segment"] == seg]
        tgt = bv.fnum(cur[-1]["neglog10p"])
        items.append((name, toks, n_sc, tgt))
        vs = [bv.fnum([r for r in curve_rows if r["variant"] == v and r["segment"] == seg][-1]["neglog10p"])
              for v in sorted({r["variant"] for r in curve_rows})]
        rng_all[seg] = [min(vs), max(vs)]
    (IN / "mini_opening.json").write_text(json.dumps(
        {"variant": VARIANT, "quote": {"N": items[0][2], "y": items[0][3]},
         "para": {"N": items[1][2], "y": items[1][3]}, "range": rng_all}, ensure_ascii=False, indent=1), encoding="utf-8")
    scratch = plt.figure(figsize=(W, 3))
    m = bt.Measurer(scratch)
    fs, lh = 11.5, 17.0
    inner = W * 72 - 2 * 16 - 20
    lay = [bt.layout(m, t, inner, fs, lh) for _, t, _, _ in items]
    plt.close(scratch)
    H = (1.25 * 72 + sum(34 + l[1] + 14 + 30 for l in lay) + 28) / 72
    fig, top = frame(H, L("Açılışın İncelenmesi: Tek Söz Kanıt Değil, Paragraf Kanıt", "Examining the Opening: One Line Is Not Proof, a Paragraph Is"),
                     L("Aynı watermark'lı metnin tokenları skora göre renklendirildi (koyu turuncu = yüksek skor).",
                       "Tokens of the same watermarked text, colored by score (dark orange = high score)."))
    ax = _pts_axes(fig)
    y = 1.2 * 72
    for (name, toks, n, tgt), (its, h) in zip(items, lay):
        ax.add_patch(FancyBboxPatch((16, y), W * 72 - 32, 34 + h + 12, boxstyle="round,pad=0,rounding_size=8", fc="white",
                                    ec=ORANGE if tgt >= 4 else GREY, lw=1.8))
        ax.text(28, y + 10, name, fontsize=12.5, fontweight="bold", color=NAVY, va="top")
        verdict = L("kanıt yok", "no evidence") if tgt < 2 else L("güçlü kanıt", "strong evidence")
        ax.text(W * 72 - 28, y + 10, f"N = {n} token   ·   −log₁₀(p) = {tgt:.1f}   ·   {_fmt_p(tgt)}   →   {verdict}",
                fontsize=10.5, color=ORANGE if tgt >= 4 else SLATE, fontweight="bold", va="top", ha="right")
        bt.draw(ax, its, 28, y + 34, fs, lh)
        y += 34 + h + 12 + 30
    _colorbar(ax, W * 72 - 215, y - 22)
    save(fig, fig_dir("main"), "fig_M_acilis")


FIGS = {"E1_2": fig_E1_2, "E3_1": fig_E3_1, "E4_1": fig_E4_1, "5_0": fig_5_0, "M_acilis": fig_M_acilis}


def main(argv=None):
    langs, rest = parse_lang_arg(list(sys.argv[1:] if argv is None else argv))
    unknown = [n for n in rest if n not in FIGS]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGS)}")
    for lang in langs:
        set_lang(lang)
        for name in (rest or FIGS):
            FIGS[name]({})
            print(f"[{get_lang()}] {name} -> figures/{get_lang()}/main/fig_{name}.png")


if __name__ == "__main__":
    main()
