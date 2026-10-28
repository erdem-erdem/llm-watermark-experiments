"""Final figures, in Turkish and English (figures/<lang>/main/*.png|svg).
Nihai şekiller, Türkçe ve İngilizce.

Data: analyze/blog_values.py (real results under results/ only).

  python -m analyze.blog_figures                          # all figures, both languages
  python -m analyze.blog_figures E0 E1_1 E2_1             # selected figures
  python -m analyze.blog_figures --lang en                # one language (tr | en | both)
"""

from __future__ import annotations

import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats

from analyze import blog_values as bv
from analyze.i18n import L, fig_dir, parse_lang_arg, set_lang
from analyze.blog_style import (BLUE, DPI, GEN, GREY, GREY_L, GRID, NAVY, ORANGE, ORANGE_L, PANEL, POST, SLATE,
                                THR2, THR6, W, chance_band, footnote, frame, rbox, save, thresholds)
from analyze.blog_values import fnum, tr

def topic_name(tp):
    return {"yaratici": L("yaratıcı", "creative"), "olgusal": L("olgusal", "factual"),
            "matematik_kod": L("matematik / kod", "math / code")}[tp]


def pct(v):
    """Percent sign position follows the language: %87 (tr) / 87% (en)."""
    return L(f"%{v:.0f}", f"{v:.0f}%")


def _t(x, nd=1):
    return tr(x, nd)


# ============================================================================= E0
def fig_E0(V):
    e0 = V["e0"]
    fig, top = frame(4.9, L("Testin Kendisini Test Etmek: Temiz Modelde p Düzgün Dağılır", "Testing the Test: p Is Uniform on a Clean Model"),
                     L("Watermark'sız bir modelde p-değerleri eşit dağılmalı; ortalama p ≈ 0.5 olmalı.",
                       "On a model without a watermark, p-values should be uniform; the mean p should be ≈ 0.5."))
    panels = [
        (L("(a) Temiz model A₀", "(a) Clean model A₀"), L("doğru anahtar", "correct key"), e0["p_a0"], NAVY),
        ("(b) " + (L("Tüm modeller", "All models") if e0["wrong_all_models"] else L("Temiz model A₀", "Clean model A₀")),
         L("beş yanlış anahtar", "five wrong keys"), e0["p_wrong"], NAVY),
        (L("(c) Tekilleştirme kapalı", "(c) Deduplication off"), L("A₀, doğru anahtar", "A₀, correct key"), e0["p_nodedup"], SLATE),
    ]
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.985, bottom=0.25, top=top - 0.10, wspace=0.22)
    bins = np.linspace(0, 1, 11)
    for j, (title, sub, p, col) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        w = np.ones_like(p) / len(p)
        ax.hist(p, bins=bins, weights=w, color=GREY, edgecolor="white", linewidth=1.2, alpha=0.9, zorder=2)
        ax.axhline(0.1, color=ORANGE, lw=1.8, ls=(0, (4, 3)), zorder=3)
        ks = stats.kstest(p, "uniform").pvalue
        ax.set_ylim(0, 0.22)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel(L("p-değeri", "p-value"), fontsize=12)
        if j == 0:
            ax.set_ylabel(L("testlerin payı", "share of tests"), fontsize=12)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{title}\n", fontsize=13, fontweight="bold", color=NAVY, loc="left", pad=2)
        ax.text(0.0, 1.02, sub, transform=ax.transAxes, fontsize=11, color=SLATE, va="bottom")
        ax.text(0.97, 0.965, L(f"ortalama p = {np.mean(p):.2f}\nKS p = {ks:.3g}\nn = {len(p)} test",
                                     f"mean p = {np.mean(p):.2f}\nKS p = {ks:.3g}\nn = {len(p)} tests"), transform=ax.transAxes,
                ha="right", va="top", fontsize=11, color=NAVY,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
    fig.text(0.075, 0.062, L("Turuncu kesikli çizgi: eşit dağılım beklentisi (her kutuda %10). Her test ≈20 bin puanlanan tokenlık bir parçadır.",
               "Orange dashed line: expected uniform level (10% per bin). Each test is a chunk of ≈20k scored tokens."),
             fontsize=10.5, color=SLATE)
    save(fig, fig_dir("main"), "fig_E0")


# ============================================================================= E1.1
SIZES = ["0.8B", "2B", "4B", "9B"]
SIZE_COL = {"0.8B": "#A9C6E4", "2B": "#6F9CCB", "4B": "#3C73AE", "9B": "#1B3F70"}
T_MARK = {0.7: "o", 1.0: "s", 1.2: "^"}


def fig_E1_1(V):
    T = V["e1"]["T"]
    sel = V["e1"]["sel"]
    fig, top = frame(5.9, L("Anlam ile İz Arasındaki Denge: Güçlü İz, Anlama Mal Olur", "Meaning vs. Trace: A Strong Trace Costs Meaning"),
                     L("Her nokta bir (model, sıcaklık) ayarı: medyan iz gücü ve medyan anlam benzerliği.",
                       "Each point is one (model, temperature) setting: median trace strength vs. mean semantic similarity."))
    ax = fig.add_axes([0.085, 0.14, 0.60, top - 0.14 - 0.03])
    ax.add_patch(Rectangle((THR6, 0.95), 600, 0.06, fc=ORANGE_L, ec="none", alpha=0.9, zorder=0))
    ax.text(THR6 * 1.12, 0.9985, L("hedef bölge: tespit edilebilir (p < 10⁻⁶)\nve anlam benzerliği ≥ 0.95", "target zone: detectable (p < 10⁻⁶)\nand similarity ≥ 0.95"), fontsize=10.5,
            color=ORANGE, va="top", fontweight="bold")
    ax.axhline(0.85, color=GREY, lw=1.0, ls=(0, (3, 3)))
    ax.text(0.58, 0.8465, L("benzerlik 0.85 (KARAR-8 alt sınırı)", "similarity 0.85 (pre-set lower bound)"), fontsize=9.5, color=SLATE, va="top")
    for r in T:
        size = r["model"].split("-")[-1]
        t = float(r["T"])
        x, y = fnum(r["median_neglog10p"]), fnum(r["mean_similarity"])
        passr = fnum(r["length_pass_rate"])
        ax.scatter(x, y, s=70 + 520 * passr, marker=T_MARK[t], color=SIZE_COL[size], edgecolor="white", linewidth=1.4, zorder=4)
    s_model, s_T = sel["model"].split("-")[-1], float(sel["T"])
    srow = next(r for r in T if r["model"].endswith(s_model) and float(r["T"]) == s_T)
    sx, sy = fnum(srow["median_neglog10p"]), fnum(srow["mean_similarity"])
    ax.scatter(sx, sy, s=1250, marker="*", facecolor="none", edgecolor=ORANGE, linewidth=2.4, zorder=5)
    ax.annotate(L(f"seçilen ayar\n{s_model}, T = {s_T:g}", f"selected setting\n{s_model}, T = {s_T:g}"), (sx, sy), xytext=(sx * 0.18, sy - 0.026), fontsize=11, color=ORANGE,
                fontweight="bold", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.3), ha="center")
    ax.set_xscale("log")
    ax.set_xlim(0.55, 600)
    ax.set_ylim(0.835, 1.0)
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax.set_xlabel(L("medyan iz gücü  −log₁₀(p)   (logaritmik eksen)", "median trace strength  −log₁₀(p)   (log axis)"), fontsize=12)
    ax.set_ylabel(L("ortalama anlam benzerliği", "mean semantic similarity"), fontsize=12)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    # lejant paneli
    lx = fig.add_axes([0.72, 0.14, 0.265, top - 0.14 - 0.03])
    lx.axis("off")
    lx.set_xlim(0, 1)
    lx.set_ylim(0, 1)
    lx.text(0, 0.96, L("Model boyutu (renk)", "Model size (colour)"), fontsize=11.5, fontweight="bold", color=NAVY, va="top")
    for i, s in enumerate(SIZES):
        lx.scatter(0.06, 0.86 - i * 0.075, s=160, color=SIZE_COL[s], edgecolor="white", linewidth=1.2)
        lx.text(0.16, 0.86 - i * 0.075, f"Qwen3.5-{s}", fontsize=11, color=NAVY, va="center")
    lx.text(0, 0.52, L("Sıcaklık (işaret)", "Temperature (marker)"), fontsize=11.5, fontweight="bold", color=NAVY, va="top")
    for i, t in enumerate((0.7, 1.0, 1.2)):
        lx.scatter(0.06, 0.42 - i * 0.075, s=150, marker=T_MARK[t], color=GREY, edgecolor="white", linewidth=1.2)
        lx.text(0.16, 0.42 - i * 0.075, f"T = {t:.1f}", fontsize=11, color=NAVY, va="center")
    lx.text(0, 0.17, L("İşaret boyutu: yeniden yazımın\nuzunluk filtresinden geçme oranı\n(en büyük ≈ %87, en küçük ≈ %27)",
                      "Marker size: share of rewrites\npassing the length filter\n(largest ≈ 87%, smallest ≈ 27%)"),
            fontsize=10, color=SLATE, va="top")
    save(fig, fig_dir("main"), "fig_E1_1")


# ============================================================================= E2.1
def fig_E2_1(V):
    cs = V["e2"]["curves"]
    fig, top = frame(5.3, L("Uzunluk ve Konu: İz, Token Sayısıyla Birikir", "Length and Topic: The Trace Accumulates with Tokens"),
                     L("Üretim sırasında işaretlenmiş cevaplarda medyan iz gücü, puanlanan token sayısına göre.",
                       "Median trace strength in answers watermarked during generation, by number of scored tokens."))
    ax = fig.add_axes([0.085, 0.17, 0.66, top - 0.17 - 0.04])
    ns = sorted({int(r["n"]) for r in cs})
    cols = {"yaratici": GEN[2], "olgusal": GEN[1], "matematik_kod": GEN[0]}
    marks = {"yaratici": "o", "olgusal": "s", "matematik_kod": "^"}
    ax.axhspan(0, THR2, color=GREY_L, zorder=0)
    ax.axhline(THR2, color="#7A8696", lw=1.0, ls=(0, (2, 2)))
    ax.axhline(THR6, color=ORANGE, lw=1.1, ls=(0, (5, 3)))
    ax.text(ns[-1] * 0.995, THR2 + 0.15, "p = 0.01", fontsize=10.5, color="#5C6877", va="bottom", ha="right")
    ax.text(ns[-1] * 0.995, THR6 - 0.15, "p = 10⁻⁶", fontsize=10.5, color=ORANGE, va="top", ha="right")
    for tp in ("yaratici", "olgusal", "matematik_kod"):
        rows = sorted([r for r in cs if r["set"] == "D_gen" and r["topic"] == tp], key=lambda r: int(r["n"]))
        x = [int(r["n"]) for r in rows]
        y = [fnum(r["median"]) for r in rows]
        lo = [fnum(r["q25"]) for r in rows]
        hi = [fnum(r["q75"]) for r in rows]
        if tp == "yaratici":
            ax.fill_between(x, lo, hi, color=ORANGE, alpha=0.13, lw=0, zorder=1)
        ax.plot(x, y, color=cols[tp], lw=2.6, marker=marks[tp], ms=7.5, mec="white", mew=1.2, zorder=4, label=topic_name(tp))
    rows = sorted([r for r in cs if r["set"] == "D_clean" and r["topic"] == "hepsi"], key=lambda r: int(r["n"]))
    ax.plot([int(r["n"]) for r in rows], [fnum(r["median"]) for r in rows], color=GREY, lw=2.2, ls="-", marker="o", ms=5.5,
            mec="white", zorder=3, label=L("watermark'sız (temiz) veri", "no-watermark (clean) data"))
    ax.set_xlim(0, ns[-1] * 1.0)
    ax.set_ylim(0, 19)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlabel(L("puanlanan token sayısı", "number of scored tokens"), fontsize=12)
    ax.set_ylabel(L("medyan −log₁₀(p)", "median −log₁₀(p)"), fontsize=12)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(ns[-1] * 0.995, 1.0, L("şans bandı", "chance band"), fontsize=10, color=SLATE, va="center", ha="right")
    ax.text(0.02, 0.97, L("turuncu bant: yaratıcı konularda %25–%75 aralığı", "orange band: 25th–75th percentile, creative topics"), transform=ax.transAxes, fontsize=9.5, color=ORANGE, va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), frameon=False, fontsize=11)
    # sağ panel: n=400'de tespit oranları
    rx = fig.add_axes([0.80, 0.17, 0.19, top - 0.17 - 0.04])
    rx.axis("off")
    rx.set_xlim(0, 1)
    rx.set_ylim(0, 1)
    rx.text(0, 0.98, L("400 tokende\ntespit oranı", "Detection rate\nat 400 tokens"), fontsize=11.5, fontweight="bold", color=NAVY, va="top")
    rx.text(0, 0.80, "(p < 10⁻⁶)", fontsize=10, color=SLATE, va="top")
    y = 0.68
    for tp in ("yaratici", "olgusal", "matematik_kod"):
        r = next(r for r in cs if r["set"] == "D_gen" and r["topic"] == tp and int(r["n"]) == 400)
        rx.text(0, y, pct(100 * fnum(r['detect_rate'])), fontsize=17, fontweight="bold", color=cols[tp], va="center")
        rx.text(0.50, y, f"{topic_name(tp)}\n({int(r['count'])} " + L("cevap", "answers") + ")", fontsize=9.5, color=SLATE, va="center")
        y -= 0.19
    r = next(r for r in cs if r["set"] == "D_clean" and r["topic"] == "hepsi" and int(r["n"]) == 400)
    rx.text(0, y, pct(100 * fnum(r['detect_rate'])), fontsize=17, fontweight="bold", color=GREY, va="center")
    rx.text(0.50, y, L("temiz veri", "clean data") + f"\n({int(r['count'])} " + L("cevap", "answers") + ")", fontsize=9.5, color=SLATE, va="center")
    save(fig, fig_dir("main"), "fig_E2_1")


# ============================================================================= E2.2
def fig_E2_2(V):
    tok = V["e2"]["tok"]
    fig, top = frame(5.1, L("Üretim Sırasında mı, Sonradan mı? Token Başına İz", "During Generation or After? Trace per Token"),
                     L("Cevap başına ortalama watermark skoru: şans düzeyi 1.0; yükseldikçe iz güçlenir.",
                       "Mean watermark score per answer: chance level is 1.0; the higher, the stronger the trace."))
    ax = fig.add_axes([0.085, 0.14, 0.62, top - 0.14 - 0.04])
    order = [("D_gen", L("Üretim verisi\n(9B, T = 0.8)", "Generation-time\ndata (9B, T = 0.8)"), ORANGE),
             ("D_post", L("Post-hoc veri\n(4B, T = 1.2)", "Post-hoc data\n(4B, T = 1.2)"), BLUE),
             ("D_clean", L("Temiz veri\n(watermark'sız)", "Clean data\n(no watermark)"), GREY)]
    data = []
    for s, _, _ in order:
        v = np.array([fnum(r["mean_score"]) for r in tok if r["set"] == s])
        data.append(v[np.isfinite(v)])
    ax.set_yscale("log")
    for i, (d, (_, _, c)) in enumerate(zip(data, order)):
        ld = np.log10(np.clip(d, 0.3, 30))
        grid = np.linspace(np.log10(0.55), np.log10(10.5), 300)
        kde = stats.gaussian_kde(ld, bw_method=0.18)
        dens = kde(grid)
        wid = 0.40 * dens / dens.max()
        ax.fill_betweenx(10 ** grid, i - wid, i + wid, color=c, alpha=0.88, edgecolor="white", linewidth=1.0, zorder=2)
    for i, d in enumerate(data):
        q1, md, q3 = np.percentile(d, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color=NAVY, lw=4.2, solid_capstyle="round", zorder=5)
        ax.scatter(i, md, s=70, color="white", edgecolor=NAVY, linewidth=1.8, zorder=6)
        ax.text(i + 0.43, md, L("medyan", "median") + f"\n{md:.2f}", fontsize=10.5, color=NAVY, va="center", zorder=7)
    ax.axhline(1.0, color=NAVY, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.62, 1.0, L("şans düzeyi ", "chance level "), fontsize=10, color=NAVY, va="bottom", ha="left")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([o[1] for o in order], fontsize=11.5)
    ax.set_ylim(0.55, 10.5)
    ax.set_yticks([0.6, 1, 2, 4, 8])
    ax.set_yticklabels(["0.6", "1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(-0.65, 2.95)
    ax.set_ylabel(L("token başına skor (log ekseni)", "score per token (log axis)"), fontsize=12)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    # sağ metin paneli
    rx = fig.add_axes([0.73, 0.14, 0.26, top - 0.14 - 0.04])
    rx.axis("off")
    rx.set_xlim(0, 1)
    rx.set_ylim(0, 1)
    ratio = (np.mean(data[1]) - 1) / (np.mean(data[0]) - 1)
    rx.text(0, 0.97, L("Post-hoc iz,\ntoken başına", "Post-hoc trace,\nper token"), fontsize=12, fontweight="bold", color=NAVY, va="top")
    rx.text(0, 0.78, f"≈ {ratio:.1f}×", fontsize=34, fontweight="bold", color=BLUE, va="top")
    rx.text(0, 0.55, L("daha güçlü: şanstan sapma\n(skor − 1) üretim verisinde\n"
                       f"{np.mean(data[0]) - 1:.2f}, post-hoc veride\n{np.mean(data[1]) - 1:.2f}.",
                       "stronger: the deviation from\nchance (score − 1) is "
                       f"{np.mean(data[0]) - 1:.2f}\nin generation-time data,\n{np.mean(data[1]) - 1:.2f} in post-hoc data."),
            fontsize=10.5, color=SLATE, va="top")
    rx.text(0, 0.17, L("Dikkat: iki veri farklı model ve\nsıcaklıkla üretildi; karşılaştırma\nadil bir A/B değildir.",
                       "Caution: the two datasets used\ndifferent models and temperatures;\nthis is not a fair A/B comparison."), fontsize=9.5,
            color="#B3261E", va="top")
    save(fig, fig_dir("main"), "fig_E2_2")


FIGS = {"E0": fig_E0, "E1_1": fig_E1_1, "E2_1": fig_E2_1, "E2_2": fig_E2_2}


def main(argv):
    from analyze import blog_figures_b
    FIGS.update(blog_figures_b.FIGS)
    langs, argv = parse_lang_arg(argv)
    unknown = [k for k in argv if k not in FIGS]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGS)}")
    for lang in langs:
        set_lang(lang)
        V = bv.load_all()
        for k in argv or list(FIGS):
            FIGS[k](V)
            print("ok", lang, k)


if __name__ == "__main__":
    main(sys.argv[1:])
