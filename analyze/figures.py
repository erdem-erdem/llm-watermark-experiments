"""Diagnostic figures of every stage, in Turkish and English (figures/<lang>/extra/*.png|svg).
Her aşamanın tanı şekilleri, Türkçe ve İngilizce.

Fixed palette: watermark = dark blue, control = grey, post-hoc = orange.
Every figure is a 300 dpi PNG + SVG; the numbers behind it are written next to the inputs as CSV
(results/<stage>/fig_*.csv). A figure whose input CSV is missing is skipped.

  python -m analyze.figures [--lang tr|en|both]      # default: both
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze.i18n import L, fig_dir, parse_lang_arg, set_lang
from wm.util import (RESULTS, dump_csv, log_cmd, parse_args, read_csv,
                     record_duration, update_manifest)

R = RESULTS

WM = "#1a3a6b"      # koyu mavi — watermark'lı
CTRL = "#8c8c8c"    # gri — kontrol / watermark'sız
POSTHOC = "#e07b39"  # turuncu — post-hoc yeniden yazım
ACC = "#2e7d32"     # yeşil — vurgu

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.titlesize": 12,
    "axes.labelsize": 11, "figure.dpi": 100, "savefig.dpi": 300})

THR = 6.0   # -log10(p) tespit eşiği
THR1 = 2.0  # p = 0.01 eşiği


def _save(fig, out: Path):
    """Write figures/<lang>/extra/<name>.png|svg (name = file stem of `out`)."""
    base = fig_dir("extra") / out.stem
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _exists(p: Path) -> bool:
    return p.exists()


def fig_pvals_hist():
    """E0: histogram of chunk p-values + uniformity."""
    src = R / "E0" / "pvals_a0.csv"
    if not _exists(src):
        return
    rows = read_csv(src)
    nl = np.asarray([float(r["neglog10p"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nl, bins=15, color=WM, alpha=0.85, edgecolor="white")
    ax.axvline(np.mean(nl), color=ACC, ls="--",
               label=L(f"ortalama = {np.mean(nl):.2f}", f"mean = {np.mean(nl):.2f}"))
    ax.set_xlabel(L("-log₁₀(p)  (beklenen ≈ 0.10)", "-log₁₀(p)  (expected ≈ 0.10)"))
    ax.set_ylabel(L("parça sayısı", "number of chunks"))
    ax.set_title(L("E0 — Temiz model kalibrasyonu (A₀, ana anahtar)", "E0 — Clean-model calibration (A₀, main key)"))
    ax.legend()
    _save(fig, R / "E0" / "fig_E0_pvals")
    dump_csv([{"mean_neglog10p": float(np.mean(nl)),
               "median_neglog10p": float(np.median(nl)),
               "std_neglog10p": float(np.std(nl)), "n_parts": len(nl)}],
             R / "E0" / "fig_E0_pvals.csv",
             fieldnames=["mean_neglog10p", "median_neglog10p",
                         "std_neglog10p", "n_parts"])


def fig_wrong_keys():
    """E0: key discrimination — main key vs. wrong keys."""
    src = R / "E0" / "wrong_keys.csv"
    if not _exists(src):
        return
    rows = read_csv(src)
    keys = [r["key"] for r in rows]
    vals = [float(r["neglog10p"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [WM if k == "main" else CTRL for k in keys]
    ax.bar(range(len(keys)), vals, color=colors)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30)
    ax.set_ylabel("-log₁₀(p)")
    ax.set_title(L("E0 — Anahtar ayrımı (okuma modu)", "E0 — Key discrimination (reading mode)"))
    _save(fig, R / "E0" / "fig_E0_wrongkeys")
    dump_csv([{k: r.get(k) for k in ("key", "mode", "model", "neglog10p", "N")}
              for r in rows], R / "E0" / "fig_E0_wrongkeys.csv",
             fieldnames=["key", "mode", "model", "neglog10p", "N"])


def fig_e1_sweep():
    """E1: sweep results with the selected setting marked."""
    src = R / "E1" / "T_E1.csv"
    raw = R / "E1" / "sweep.csv"
    if not _exists(src) or not _exists(raw):
        return
    summ = read_csv(src)
    rawrows = read_csv(raw)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = sorted({s["model"] for s in summ})
    Ts = sorted({float(s["T"]) for s in summ})
    width = 0.8 / len(Ts)
    for ti, T in enumerate(Ts):
        xs, ys, cols = [], [], []
        for mi, m in enumerate(models):
            s = [x for x in summ if x["model"] == m and float(x["T"]) == T]
            if not s:
                continue
            s = s[0]
            xs.append(mi - 0.4 + width * (ti + 0.5))
            ys.append(float(s["median_neglog10p"]))
            cols.append(POSTHOC if s["eligible"] == "True" else CTRL)
        ax.bar(xs, ys, width=width * 0.9, color=cols, label=f"T={T}")
    # seçili ayarı işaretle
    import json
    sel_path = R / "E1" / "auto_selection.json"
    if _exists(sel_path):
        sel = json.loads(sel_path.read_text(encoding="utf-8"))["auto_selected"]
        mi = models.index(sel["model"]) if sel["model"] in models else None
        if mi is not None:
            ax.annotate(L(f"seçili\n{sel['model']} T={sel['T']}", f"selected\n{sel['model']} T={sel['T']}"),
                        xy=(mi, float(sel["median_neglog10p"])),
                        xytext=(mi - 0.4, float(sel["median_neglog10p"]) * 1.15),
                        arrowprops={"arrowstyle": "->", "color": ACC},
                        color=ACC, fontsize=10)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel(L("medyan -log₁₀(p)", "median -log₁₀(p)"))
    ax.set_title(L("E1 — Post-hoc yeniden yazım taraması", "E1 — Post-hoc rewrite sweep"))
    ax.legend(title=L("sıcaklık", "temperature"))
    _save(fig, R / "E1" / "fig_E1_1")
    dump_csv(summ, R / "E1" / "fig_E1_1.csv",
             fieldnames=list(summ[0].keys()))


def fig_e2_curves():
    """E2: D_gen prefix curves — median -log10p vs. n + thresholds."""
    src = R / "E2" / "curves.csv"
    if not _exists(src):
        return
    rows = read_csv(src)
    fig, ax = plt.subplots(figsize=(7, 5))
    ns = sorted({int(float(r["n"])) for r in rows})
    med = []
    q25, q75 = [], []
    for n in ns:
        v = [float(r["neglog10p"]) for r in rows if int(float(r["n"])) == n]
        med.append(np.median(v))
        q25.append(np.percentile(v, 25))
        q75.append(np.percentile(v, 75))
    ax.plot(ns, med, color=WM, lw=2, marker="o", ms=4,
            label=L("D_gen (wm, ana anahtar)", "D_gen (wm, main key)"))
    ax.fill_between(ns, q25, q75, color=WM, alpha=0.18, label=L("%25–%75", "25th–75th pct."))
    ax.axhline(THR, color=CTRL, ls="--", lw=1,
               label=L(f"tespit eşiği (-log₁₀p = {THR:.0f})", f"detection threshold (-log₁₀p = {THR:.0f})"))
    ax.axhline(THR1, color=CTRL, ls=":", lw=1, label="p = 0.01")
    ax.set_xlabel(L("puanlanan token sayısı (n)", "number of scored tokens (n)"))
    ax.set_ylabel("-log₁₀(p)")
    ax.set_title(L("E2 — İlk n token ile tespit eğrisi", "E2 — Detection curve on the first n tokens"))
    ax.legend(fontsize=9)
    _save(fig, R / "E2" / "fig_E2_1_curves")
    dump_csv([{"n": n, "median": m, "q25": a, "q75": b}
              for n, m, a, b in zip(ns, med, q25, q75)],
             R / "E2" / "fig_E2_1_curves.csv",
             fieldnames=["n", "median", "q25", "q75"])


def fig_e2_dist():
    """E2: per-sample -log10p distribution, D_gen vs. D_clean."""
    src = R / "E2" / "detection.csv"
    if not _exists(src):
        return
    rows = read_csv(src)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.linspace(-2, max(20, max(float(r["neglog10p"]) for r in rows)), 40)
    for s, c in (("D_gen", WM), ("D_clean", CTRL)):
        v = [float(r["neglog10p"]) for r in rows if r["set"] == s]
        ax.hist(v, bins=bins, alpha=0.65, color=c,
                label=f"{s} (n={len(v)})")
    ax.axvline(THR, color=CTRL, ls="--", lw=1, label=L("tespit eşiği", "detection threshold"))
    ax.set_xlabel("-log₁₀(p)")
    ax.set_ylabel(L("örnek sayısı", "number of samples"))
    ax.set_title(L("E2 — Tespit p-değeri dağılımı", "E2 — Distribution of detection p-values"))
    ax.legend()
    _save(fig, R / "E2" / "fig_E2_2_dist")
    dump_csv([{"set": s, "n": len([r for r in rows if r["set"] == s]),
               "median": float(np.median([float(r["neglog10p"]) for r in rows
                                          if r["set"] == s]))}
              for s in ("D_gen", "D_clean")],
             R / "E2" / "fig_E2_2_dist.csv", fieldnames=["set", "n", "median"])


def fig_e2_topic():
    """E2: medians by topic."""
    src = R / "E2" / "medians_by_topic.csv"
    if not _exists(src):
        return
    rows = read_csv(src)
    topics = sorted({r["topic"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(topics))
    w = 0.35
    for i, (s, c) in enumerate((("D_gen", WM), ("D_clean", CTRL))):
        v = []
        for t in topics:
            rr = [r for r in rows if r["topic"] == t and r["set"] == s]
            v.append(float(rr[0]["median_neglog10p"]) if rr else 0)
        ax.bar(x + (i - 0.5) * w, v, w, color=c, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(topics)
    ax.set_ylabel(L("medyan -log₁₀(p)", "median -log₁₀(p)"))
    ax.set_title(L("E2 — Konu bazlı tespit gücü", "E2 — Detection strength by topic"))
    ax.legend()
    _save(fig, R / "E2" / "fig_E2_3_topic")
    dump_csv(rows, R / "E2" / "fig_E2_3_topic.csv",
             fieldnames=["topic", "set", "median_neglog10p"])


def fig_e3_curves():
    """E3: detection curves per training condition (20 points)."""
    src = R / "E3" / "detect_curves.csv"
    if not _exists(src):
        return
    rows = read_csv(src)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    palette = {}
    order = []
    for r in rows:
        m = r["model"]
        if m not in palette:
            order.append(m)
            palette[m] = (CTRL if m in ("A0", "A-rew")
                          else POSTHOC if m.startswith("A-post")
                          else WM)
    for m in order:
        v = sorted([(int(float(r["n"])), float(r["neglog10p"]))
                    for r in rows if r["model"] == m])
        ax.plot([a for a, _ in v], [b for _, b in v], marker="o", ms=3,
                lw=1.6, color=palette[m], label=m)
    ax.set_xscale("log")
    ax.axhline(THR, color=CTRL, ls="--", lw=1, label=L("tespit eşiği", "detection threshold"))
    ax.set_xlabel(L("puanlanan token sayısı (n)", "number of scored tokens (n)"))
    ax.set_ylabel("-log₁₀(p)")
    ax.set_title(L("E3 — Eğitim koşuluna göre radyoaktivite eğrileri", "E3 — Radioactivity curves by training condition"))
    ax.legend(fontsize=8, ncol=2)
    _save(fig, R / "E3" / "fig_E3_1_curves")
    dump_csv(rows, R / "E3" / "fig_E3_1_curves.csv",
             fieldnames=["model", "n", "neglog10p"])


def fig_e3_loss():
    """E3: training loss curves (all in one plot)."""
    loss_dir = R / "E3" / "loss"
    if not loss_dir.is_dir():
        return
    files = sorted(loss_dir.glob("*.csv"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for f in files:
        rows = read_csv(f)
        steps = [int(float(r["step"])) for r in rows if r["step"] not in ("", None)]
        vals = [float(r["loss"]) for r in rows if r["loss"] not in ("", None)]
        name = f.stem
        c = CTRL if name in ("A0", "A-rew") else \
            POSTHOC if name.startswith("A-post") else WM
        ax.plot(steps, vals, lw=1.5, color=c, alpha=0.85, label=name)
    ax.set_xlabel(L("adım", "step"))
    ax.set_ylabel(L("kayıp", "loss"))
    ax.set_title(L("E3 — Eğitim kayıp eğrileri", "E3 — Training loss curves"))
    ax.legend(fontsize=8, ncol=2)
    _save(fig, R / "E3" / "fig_E3_2_loss")


def fig_e4_chain():
    """E4: the 4-ring chain — main key vs. median of wrong keys."""
    src = R / "E4" / "chain.csv"
    if not _exists(src):
        return
    rows = read_csv(src)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows))
    main = [float(r["neglog10p"]) for r in rows]
    wrong = [float(r["wrong_key_median"]) if r.get("wrong_key_median")
             not in ("", None) else 0.0 for r in rows]
    ax.bar(x - 0.2, main, 0.4, color=WM, label=L("ana anahtar", "main key"))
    ax.bar(x + 0.2, wrong, 0.4, color=CTRL, label=L("yanlış anahtarlar (medyan)", "wrong keys (median)"))
    ax.set_xticks(x)
    ax.set_xticklabels([L("H", "R") + f"{r['ring']}\n{(r.get('cond') or r['what'][:14]).replace('_', chr(10))}" for r in rows],
                       fontsize=7)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("-log₁₀(p)")
    ax.set_title(L("E4 — Radyoaktivite zinciri (4 halka)", "E4 — Radioactivity chain (4 rings)"))
    ax.legend()
    _save(fig, R / "E4" / "fig_E4_1_chain")
    dump_csv(rows, R / "E4" / "fig_E4_1_chain.csv",
             fieldnames=["ring", "what", "cond", "key", "neglog10p", "N",
                         "median_per_sample", "wrong_key_median"])


def fig_evals():
    """E2: math-olympiad quality check — wm vs. no-wm conditions (legacy file name: aime_*)."""
    src = R / "E2" / "aime_summary.json"
    if not _exists(src):
        return
    import json
    s = json.loads(src.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([L("watermark'lı", "watermarked"), L("watermark'sız", "unwatermarked")],
                  [s["wm_mean"], s["nowm_mean"]],
                  yerr=[s["wm_std"], s["nowm_std"]],
                  color=[WM, CTRL], capsize=5)
    ax.set_ylabel(L("doğruluk", "accuracy"))
    n_prob = int(s["n_problems"])
    ax.set_title(L(f"Matematik olimpiyatı (TÜBİTAK) — Kalite kontrolü ({n_prob} problem)",
                   f"Math olympiad (TÜBİTAK) — Quality check ({n_prob} problems)"))
    ax.set_ylim(0, max(0.05, s["nowm_mean"] + s["nowm_std"] + 0.08))
    for b, v, sd in zip(bars, [s["wm_mean"], s["nowm_mean"]], [s["wm_std"], s["nowm_std"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + sd + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    _save(fig, R / "E2" / "fig_E2_aime")
    dump_csv([{"condition": "wm", "mean": s["wm_mean"], "std": s["wm_std"]},
              {"condition": "nowm", "mean": s["nowm_mean"], "std": s["nowm_std"]}],
             R / "E2" / "fig_E2_aime.csv",
             fieldnames=["condition", "mean", "std"])


def fig_chain_diagram():
    """Experiment chain diagram (no data)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    boxes = [
        (L("E0\nkalibras-\nyon", "E0\ncalibration"), CTRL),
        (L("E1\npost-hoc\ntarama", "E1\npost-hoc\nsweep"), POSTHOC),
        ("E2\nD_gen /\nD_clean", WM),
        (L("E3\n8 eğitim\nkoşulu", "E3\n8 training\nconditions"), WM),
        (L("E4\najan\nzinciri", "E4\nagent\nchain"), WM),
    ]
    for i, (label, c) in enumerate(boxes):
        x = 0.06 + i * 0.19
        ax.add_patch(plt.Rectangle((x, 0.35), 0.15, 0.35, fill=True,
                                   color=c, alpha=0.9, lw=0))
        ax.text(x + 0.075, 0.52, label, color="white", ha="center",
                va="center", fontsize=9.5, fontweight="bold")
        if i:
            ax.annotate("", xy=(x - 0.004, 0.525), xytext=(x - 0.036, 0.525),
                        xycoords="axes fraction",
                        arrowprops={"arrowstyle": "->", "color": "#333",
                                    "lw": 2})
    ax.text(0.5, 0.14, L("Veri: InstrucTurca 5.500 örnek → D_gen (wm) → post-hoc "
                         "yeniden yazım (E1) → SFT koşuları (E3) → ajan halkaları "
                         "(E4)",
                         "Data: InstrucTurca 5,500 samples → D_gen (wm) → post-hoc\n"
                         "rewrite (E1) → SFT runs (E3) → agent rings (E4)"),
            ha="center", fontsize=9, color="#333")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, R / "E5_chain" / "fig_chain")


def fig_mini_token_colors():
    """Mini experiment: example 1.1 token colouring (scored tokens highlighted)."""
    src = R / "mini" / "example_1_1.csv"
    if not _exists(src):
        return
    rows = read_csv(src)
    first = rows[0].get("pair") if rows else None
    rows = [r for r in rows if r.get("pair") == first]       # first pair only
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis("off")
    tokens = [r.get("piece", r.get("token", "")) for r in rows]
    flagged = [r.get("scored") in ("1", "True", "true") for r in rows]
    x = 0
    for tok, fl in zip(tokens, flagged):
        ax.text(x, 0.5, tok, fontsize=9,
                color=WM if fl else "#333",
                backgroundcolor="#dce7f5" if fl else "none")
        x += 0.011 * max(1, len(tok) + 1)
    ax.set_xlim(0, max(x, 1))
    ax.set_ylim(0, 1)
    _save(fig, R / "mini" / "fig_mini_tokens")


def fig_mini_curve():
    """Mini experiment: first-n-tokens curve."""
    src = R / "mini" / "fig13_curve.csv"
    if not _exists(src):
        return
    rows = read_csv(src)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([float(r["n"]) for r in rows], [float(r["neglog10p"]) for r in rows],
            color=WM, marker="o", ms=4)
    ax.axhline(THR, color=CTRL, ls="--", lw=1, label=L("tespit eşiği", "detection threshold"))
    ax.set_xlabel(L("puanlanan token sayısı", "number of scored tokens"))
    ax.set_ylabel("-log₁₀(p)")
    ax.set_title(L("Mini — Örnek metinde ilk n token", "Mini — First n tokens of an example text"))
    ax.legend()
    _save(fig, R / "mini" / "fig13_curve")


ALL_FIGS = [fig_pvals_hist, fig_wrong_keys, fig_e1_sweep, fig_e2_curves,
            fig_e2_dist, fig_e2_topic, fig_e3_curves, fig_e3_loss,
            fig_e4_chain, fig_evals, fig_chain_diagram,
            fig_mini_token_colors, fig_mini_curve]


def main():
    langs, rest = parse_lang_arg(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest              # wm.util.parse_args does not know --lang
    parse_args("figure generation / figür üretimi")
    log_cmd(sys.argv)
    t0 = time.time()
    done = []
    for lang in langs:
        set_lang(lang)
        for f in ALL_FIGS:
            try:
                f()
                done.append(f"{lang}:{f.__name__}")
            except Exception as e:
                print(f"  skipped ({lang}:{f.__name__}): {e}")
    print(f"figures: {len(done)}/{len(ALL_FIGS) * len(langs)} produced")
    update_manifest(**{"figures.done": done})
    record_duration("figures", time.time() - t0)


if __name__ == "__main__":
    main()
