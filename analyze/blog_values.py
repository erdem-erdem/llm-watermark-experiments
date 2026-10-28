"""Single data source for the figures: reads ONLY real result files under results/.
Şekiller için TEK veri kaynağı: yalnızca results/ altındaki gerçek sonuç dosyalarını okur.

There are no estimates or placeholders: if a required file is missing a FileNotFoundError names the
file and the stage that produces it (see RUN_MANUAL.md).
Tahmin / yer tutucu yoktur: gerekli bir dosya eksikse hata, dosyayı ve onu üreten aşamayı söyler.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from analyze.i18n import L

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DATA = ROOT / "data"


# ----------------------------------------------------------------------------- helpers
def read_csv(p):
    """Lenient reader: missing file → []. Use `need_csv` for required inputs."""
    p = Path(p)
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _missing(p, stage):
    rel = Path(p).relative_to(ROOT) if ROOT in Path(p).parents else p
    return FileNotFoundError(f"missing results file: {rel} — produced by: {stage}")


def need_csv(p, stage):
    rows = read_csv(p)
    if not rows:
        raise _missing(p, stage)
    return rows


def need_json(p, stage):
    d = read_json(p)
    if d is None:
        raise _missing(p, stage)
    return d


def fnum(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def tr(x, nd=1):
    """Number format used in all figures: '.' decimal, space as thousands separator → 22 793."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", " ")
    return f"{x:.{nd}f}"


# ----------------------------------------------------------------------------- E3
# model → (ρ, source). Source strings are internal identifiers (compared in the figure code); use
# `src_label` for display.
E3_PARAMS = {
    "A0":         ("%0",   "—"),
    "A-gen-10":   ("%10",  "üretim"),
    "A-gen-50":   ("%50",  "üretim"),
    "A-gen-100":  ("%100", "üretim"),
    "A-post-10":  ("%10",  "post-hoc"),
    "A-post-50":  ("%50",  "post-hoc"),
    "A-post-100": ("%100", "post-hoc"),
    "A-rew":      ("%100", "yeniden yazım"),
}


def label_of(name: str) -> str:
    """Display name of a trained model in the active language."""
    return {
        "A0": "A₀",
        "A-gen-10": L("A-üretim-10", "A-gen-10"),
        "A-gen-50": L("A-üretim-50", "A-gen-50"),
        "A-gen-100": L("A-üretim-100", "A-gen-100"),
        "A-post-10": L("A-posthoc-10", "A-post-10"),
        "A-post-50": L("A-posthoc-50", "A-post-50"),
        "A-post-100": L("A-posthoc-100", "A-post-100"),
        "A-rew": L("A-yeniden-yazım", "A-rewrite"),
        "B-greedy": "B-greedy",
        "B-sample": L("B-örnekleme", "B-sample"),
        "B-control": "B′",
    }.get(name, name)


def src_label(src: str) -> str:
    return {"üretim": L("üretim", "generation"), "post-hoc": "post-hoc",
            "yeniden yazım": L("yeniden yazım", "rewrite"), "—": "—"}.get(src, src)


def _wrong_mean_p(r):
    vals = [fnum(r.get(f"wrong{i}")) for i in range(5)]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean([10 ** (-v) for v in vals])) if vals else float("nan")


def load_e3():
    """→ {model: {...}} with reading-mode result, curves and downstream (GPQA-TR, eval loss)."""
    stage = "python -m analyze.reading_rerun (scripts/rerun_reading.sh)"
    summ = need_csv(RES / "E3" / "reading_summary.csv", stage)
    curv = need_csv(RES / "E3" / "reading_curves.csv", stage)
    ev = {r["label"]: r for r in (read_json(RES / "E3" / "eval_loss.json") or {"results": []})["results"]}
    for r in (read_json(RES / "E4" / "eval_loss.json") or {"results": []})["results"]:
        ev.setdefault(r["label"], r)
    gp = {r["label"]: r for r in (read_json(RES / "E3" / "gpqa_tr.json") or {"results": []})["results"]}
    for r in (read_json(RES / "E4" / "gpqa_tr.json") or {"results": []})["results"]:
        gp.setdefault(r["label"], r)
    out = {}
    for name, (rho, src) in E3_PARAMS.items():
        test = "gen" if name == "A0" else ("post" if name == "A-rew" else "seen")
        r = next((x for x in summ if x["model"] == name and x["test"] == test), None)
        if r is None:
            raise _missing(RES / "E3" / "reading_summary.csv", f"{stage} — no row for {name}/{test}")
        c = [x for x in curv if x["model"] == name and x["test"] == test]
        e, g = ev.get(name), gp.get(name)
        nan = float("nan")
        out[name] = {
            "model": name, "label": label_of(name), "rho": rho, "src": src,
            "N": int(fnum(r["N"])), "final": fnum(r["neglog10p"]), "wrong_p": _wrong_mean_p(r),
            "m": fnum(r["mean_score_per_token"]),
            "curve_n": np.array([fnum(x["n"]) for x in c]), "curve_y": np.array([fnum(x["neglog10p"]) for x in c]),
            "curve_w": np.array([fnum(x["wrong_median"]) for x in c]),
            "gpqa": fnum(g["accuracy"]) if g else nan, "gpqa_lo": fnum(g["ci_lo"]) if g else nan,
            "gpqa_hi": fnum(g["ci_hi"]) if g else nan,
            "loss": fnum(e["mean_loss"]) if e else nan, "loss_lo": fnum(e["ci_lo"]) if e else nan,
            "loss_hi": fnum(e["ci_hi"]) if e else nan,
        }
    return out


# ----------------------------------------------------------------------------- E4
def load_e4(e3):
    stage = "python -m analyze.detect_all --stage E4 / analyze.reading_rerun"
    rows = need_csv(RES / "E4" / "chain.csv", stage)
    r1 = next((r for r in rows if r.get("ring") == "1"), None)
    if r1 is None:
        raise _missing(RES / "E4" / "chain.csv", f"{stage} — no ring 1 row")
    d = {"D_gen": {"y": fnum(r1["neglog10p"]), "N": int(fnum(r1["N"]))}}
    a = e3["A-gen-100"]                          # ring 2 = the same reading-mode measurement as E3
    d["A-gen-100"] = {"y": a["final"], "N": a["N"], "wrong": a["wrong_p"]}

    def ring(k, names):
        got = {r["cond"]: {"y": fnum(r["neglog10p"]), "N": int(fnum(r["N"])),
                           "wrong": fnum(r.get("wrong_key_median"))} for r in rows if r["ring"] == k}
        for n in names:
            if n not in got:
                raise _missing(RES / "E4" / "chain.csv", f"{stage} — no ring {k} row for {n}")
        return got

    d["ring3"] = ring("3", ("agent_greedy", "agent_sampled", "a0_greedy"))
    d["ring4"] = ring("4", ("B-greedy", "B-sample", "B-control"))
    ev = {r["label"]: r for r in (read_json(RES / "E4" / "eval_loss.json") or {"results": []})["results"]}
    gp = {r["label"]: r for r in (read_json(RES / "E4" / "gpqa_tr.json") or {"results": []})["results"]}
    for k in ("B-greedy", "B-sample", "B-control"):
        e, g = ev.get(k), gp.get(k)
        d["ring4"][k].update(loss=fnum(e["mean_loss"]) if e else float("nan"),
                             gpqa=fnum(g["accuracy"]) if g else float("nan"),
                             gpqa_lo=fnum(g["ci_lo"]) if g else float("nan"),
                             gpqa_hi=fnum(g["ci_hi"]) if g else float("nan"))
    return d


# ----------------------------------------------------------------------------- E0
def load_e0():
    stage = "python -m analyze.e0_complete"
    e0 = need_json(RES / "E0" / "E0.json", stage)
    a0 = [fnum(r["neglog10p"]) for r in need_csv(RES / "E0" / "pvals_a0_exact.csv", stage)]
    nod = [fnum(r["neglog10p"]) for r in need_csv(RES / "E0" / "pvals_a0_nodedup.csv", stage)]
    allw = read_csv(RES / "E0" / "pvals_parts_all.csv")
    if allw:           # all watermarked models × 5 wrong keys
        wrong = [fnum(r["p"]) for r in allw if r["key"].startswith("wrong")]
        real_all = True
    else:              # fallback: clean model A₀ only
        wrong = [fnum(r["p"]) for r in need_csv(RES / "E0" / "pvals_wrongkeys.csv", stage)]
        real_all = False
    p = lambda nl: [10 ** (-v) for v in nl]
    return {"e0": e0, "p_a0": np.array(p(a0)), "p_nodedup": np.array(p(nod)), "p_wrong": np.array(wrong),
            "wrong_all_models": real_all}


# ----------------------------------------------------------------------------- E1
def load_e1():
    stage = "python -m analyze.e1_report / analyze.e1_complete"
    return {"T": need_csv(RES / "E1" / "T_E1.csv", stage),
            "sweep": need_csv(RES / "E1" / "sweep.csv", stage),
            "sel": need_json(RES / "E1" / "selected.json", stage),
            "full": need_json(RES / "E1" / "full_summary.json", stage)}


# ----------------------------------------------------------------------------- E2
def load_e2():
    stage = "python -m analyze.e2_complete"
    return {"T": {r["set"]: r for r in need_csv(RES / "E2" / "T_E2.csv", stage)},
            "curves": need_csv(RES / "E2" / "curves_summary.csv", stage),
            "tok": need_csv(RES / "E2" / "token_mean_scores.csv", stage),
            "div": need_json(RES / "E2" / "diversity.json", stage),
            "summ": need_json(RES / "E2" / "E2_summary.json", stage)}


def load_aime():
    """Math-olympiad quality check (legacy 'aime' name; data = TÜBİTAK olympiad, see README)."""
    s = need_json(RES / "E2" / "aime_summary.json", "python -m eval.aime_tr / analyze.e2_complete")
    return {**{k: s[k] for k in ("wm_mean", "wm_std", "nowm_mean", "nowm_std", "n_problems")},
            "no_boxed": float(np.mean([v["no_boxed_rate"] for v in s["per_condition"].values()]))}


# ----------------------------------------------------------------------------- mini experiments
def load_mini():
    out = {"opening_real": (RES / "mini" / "opening_tokens.csv").exists()}
    out["bench"] = need_json(RES / "mini" / "benchmark_marking.json", "python -m gen.gen_mini")
    out["guess"] = read_csv(RES / "mini" / "guess_pairs.csv")
    out["same"] = read_json(RES / "mini" / "same_prompt_twice.json")
    out["fig13"] = read_json(RES / "mini" / "fig13_meta.json")
    return out


# ----------------------------------------------------------------------------- all
def load_all():
    e3 = load_e3()
    return {"e0": load_e0(), "e1": load_e1(), "e2": load_e2(), "aime": load_aime(), "e3": e3,
            "e4": load_e4(e3), "mini": load_mini()}


if __name__ == "__main__":
    load_all()
    print("all figure inputs present")
