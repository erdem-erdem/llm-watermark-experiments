"""Tablo CSV'leri — TABLO yer tutucularına giden veriler (T_E2..T_E4).

  python -m analyze.tables
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from wm.util import (RESULTS, dump_csv, log_cmd, parse_args, read_csv,
                     read_jsonl, record_duration, update_manifest)

R = RESULTS
THR = 6.0


def t_e2():
    # analyze.e2_complete tam tabloyu (D_gen/D_post/D_clean + çeşitlilik) yazdıysa
    # eski, eksik tabloyla ÜZERİNE YAZMA
    if (R / "E2" / "E2_summary.json").exists():
        return
    rows = []
    det = R / "E2" / "detection.csv"
    if det.exists():
        d = read_csv(det)
        for s in ("D_gen", "D_clean"):
            v = [float(r["neglog10p"]) for r in d if r["set"] == s]
            n400 = [float(r["neglog10p"]) for r in d
                    if r["set"] == s and float(r["N"]) >= 400]
            rows.append({"set": s, "n": len(v),
                         "median_neglog10p": float(np.median(v)),
                         "mean_neglog10p": float(np.mean(v)),
                         "detect_rate_median": float(np.median(v) >= THR),
                         "n_N>=400": len(n400),
                         "rate_at_n400": float(np.mean([x >= THR for x in n400]))
                         if n400 else ""})
    aime = R / "E2" / "aime_summary.json"
    if aime.exists():
        s = json.loads(aime.read_text(encoding="utf-8"))
        rows.append({"set": "AIME_wm", "n": s["n_problems"],
                     "median_neglog10p": "",
                     "mean_neglog10p": "", "detect_rate_median": "",
                     "n_N>=400": "", "rate_at_n400": "",
                     "accuracy_mean": s["wm_mean"],
                     "accuracy_std": s["wm_std"]})
        rows.append({"set": "AIME_nowm", "n": s["n_problems"],
                     "median_neglog10p": "", "mean_neglog10p": "",
                     "detect_rate_median": "", "n_N>=400": "",
                     "rate_at_n400": "", "accuracy_mean": s["nowm_mean"],
                     "accuracy_std": s["nowm_std"]})
    if rows:
        dump_csv(rows, R / "E2" / "T_E2.csv",
                 fieldnames=["set", "n", "median_neglog10p", "mean_neglog10p",
                             "detect_rate_median", "n_N>=400", "rate_at_n400",
                             "accuracy_mean", "accuracy_std"])


def t_e3():
    rows = []
    curves = R / "E3" / "detect_curves.csv"
    if curves.exists():
        d = read_csv(curves)
        for m in sorted({r["model"] for r in d}):
            row = {"model": m}
            for n in (50, 100, 200, 400):
                v = [float(r["neglog10p"]) for r in d
                     if r["model"] == m and int(float(r["n"])) == n]
                row[f"neglog10p@n{n}"] = v[0] if v else ""
            rows.append(row)
    ev = R / "E3" / "eval_loss.json"
    if ev.exists():
        s = json.loads(ev.read_text(encoding="utf-8"))
        for r in s["results"]:
            row = [x for x in rows if x["model"] == r["label"]]
            tgt = row[0] if row else {"model": r["label"]}
            tgt["eval_loss"] = r["mean_loss"]
            tgt["eval_ppl"] = r["ppl"]
            if not row:
                rows.append(tgt)
    gp = R / "E3" / "gpqa_tr.json"
    if gp.exists():
        s = json.loads(gp.read_text(encoding="utf-8"))
        for r in s["results"]:
            row = [x for x in rows if x["model"] == r["label"]]
            tgt = row[0] if row else {"model": r["label"]}
            tgt["gpqa_acc"] = r["accuracy"]
            tgt["gpqa_ci"] = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
            if not row:
                rows.append(tgt)
    if rows:
        keys = sorted({k for r in rows for k in r})
        dump_csv(rows, R / "E3" / "T_E3.csv", fieldnames=keys)


def t_e4():
    src = R / "E4" / "chain.csv"
    if not src.exists():
        return
    rows = read_csv(src)
    dump_csv(rows, R / "E4" / "T_E4.csv",
             fieldnames=["ring", "what", "key", "neglog10p", "N",
                         "median_per_sample", "wrong_key_median"])
    # eval B modelleri
    ev = R / "E4" / "eval_loss.json"
    extra = []
    if ev.exists():
        s = json.loads(ev.read_text(encoding="utf-8"))
        for r in s["results"]:
            extra.append({"model": r["label"], "eval_loss": r["mean_loss"],
                          "eval_ppl": r["ppl"]})
    gp = R / "E4" / "gpqa_tr.json"
    if gp.exists():
        s = json.loads(gp.read_text(encoding="utf-8"))
        for r in s["results"]:
            row = [x for x in extra if x["model"] == r["label"]]
            tgt = row[0] if row else {"model": r["label"]}
            tgt["gpqa_acc"] = r["accuracy"]
            if not row:
                extra.append(tgt)
    if extra:
        dump_csv(extra, R / "E4" / "T_E4_evals.csv",
                 fieldnames=sorted({k for r in extra for k in r}))


def t_e0():
    src = R / "E0" / "pvals_a0.csv"
    if not src.exists():
        return
    rows = read_csv(src)
    nl = np.asarray([float(r["neglog10p"]) for r in rows])
    ns = np.asarray([int(float(r["N"])) for r in rows])
    dump_csv([{"metric": "mean_neglog10p", "value": float(np.mean(nl))},
              {"metric": "std_neglog10p", "value": float(np.std(nl))},
              {"metric": "mean_N", "value": float(np.mean(ns))},
              {"metric": "in_0.4_0.6_mean_p",
               "value": bool(0.4 <= np.mean(10.0 ** (-nl)) <= 0.6)},
              {"metric": "mean_p", "value": float(np.mean(10.0 ** (-nl)))},
              {"metric": "n_parts", "value": len(rows)}],
             R / "E0" / "T_E0.csv", fieldnames=["metric", "value"])


def main():
    argv = parse_args("tablo üretimi")
    log_cmd(sys.argv)
    t0 = time.time()
    t_e0()
    t_e2()
    t_e3()
    t_e4()
    update_manifest(**{"tables.done": True})
    record_duration("tables", time.time() - t0)
    print("tablolar tamam")


if __name__ == "__main__":
    main()