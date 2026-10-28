"""E0 tamamlama (CPU; E0/preds/a0.npz önbelleğinden) — kabul ölçütü ve eksik paneller.

Düzeltilen/eklenen:
  - Kabul ölçütü: p-değerlerinin ORTALAMASI (önceki hesap -log10p ortalamasının
    10^-x'ini alıyordu: 0.34; doğrusu mean(10^-x))  + KS testi (düzgün dağılıma)
  - (b) paneli: 5 yanlış anahtar × 50 parça p-değerleri (histogram + KS)
  - (c) paneli: tekilleştirme kapalı (yalnız kural (a)) — TEMİZ model A₀ üzerinde,
    ana anahtarla; açık/kapalı yan yana, 50 parça p-değeri

  python -m analyze.e0_complete
Çıktılar: E0/E0.json, E0/pvals_wrongkeys.csv, E0/pvals_a0_nodedup.csv
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy import stats

from wm.prf import R
from wm.reading_mode import load_predictions, neglog10p_of, score_reading
from wm.scoring import log10_p, per_token_score
from wm.util import (RESULTS, dump_csv, dump_json, log_cmd, parse_args,
                     record_duration, update_manifest)

E0 = RESULTS / "E0"
N_PARTS = 50


def _parts(samples):
    """Örnek indekslerini 50 eşit parçaya böl (parça başına ayrı test)."""
    idxs = sorted({int(i) for i, _ in samples})
    return [list(map(int, part)) for part in np.array_split(np.asarray(idxs), N_PARTS)]


def _ks(p):
    return float(stats.kstest(p, "uniform").pvalue)


def main():
    parse_args("E0 tamamlama")
    log_cmd(sys.argv)
    t0 = time.time()
    sel = json.loads((E0 / "selected_keys.json").read_text(encoding="utf-8"))
    main_key, wrong = sel["main_key"], sel["wrong_keys"]
    windows, preds, samples, k = load_predictions(E0 / "preds" / "a0.npz")
    parts = _parts(samples)

    # 1) ana anahtar: kural (a)-(d) spesifikasyona TAM uygun (c: cevap başına)
    rows = []
    for pi, part in enumerate(parts):
        sc, _ = score_reading(windows, preds, samples, main_key, k, include=part)
        rows.append({"part": pi, "N": len(sc), "neglog10p": neglog10p_of(sc)})
    dump_csv(rows, E0 / "pvals_a0_exact.csv", ["part", "N", "neglog10p"])
    p_main = 10.0 ** (-np.asarray([r["neglog10p"] for r in rows]))

    # 2) yanlış anahtarlar × parça (tekilleştirme açık)
    wrong_rows, p_wrong = [], []
    for wi, wk in enumerate(wrong):
        for pi, sub in enumerate(parts):
            sc, _ = score_reading(windows, preds, samples, wk, k, include=sub)
            nl = neglog10p_of(sc)
            p = 10.0 ** (-nl)
            p_wrong.append(p)
            wrong_rows.append({"key": f"wrong{wi}", "part": pi, "N": len(sc),
                               "neglog10p": nl, "p": p})
    dump_csv(wrong_rows, E0 / "pvals_wrongkeys.csv",
             ["key", "part", "N", "neglog10p", "p"])

    # 3) tekilleştirme KAPALI (yalnız kural a): temiz model, ana anahtar
    off_rows, p_off = [], []
    in_part = {}
    for i, (idx, _t) in enumerate(samples):
        in_part.setdefault(int(idx), []).append(i)
    for pi, part in enumerate(parts):
        sub = [i for idx in part for i in in_part[idx]]
        sc = [per_token_score(R(main_key, windows[i], int(preds[i]))) for i in sub]
        N = len(sc)
        nl = log10_p(float(np.sum(sc)), N)
        p_off.append(10.0 ** (-nl))
        off_rows.append({"part": pi, "N": N, "neglog10p": nl})
    dump_csv(off_rows, E0 / "pvals_a0_nodedup.csv", ["part", "N", "neglog10p"])

    p_wrong = np.asarray(p_wrong)
    p_off = np.asarray(p_off)
    mean_N_on = float(np.mean([r["N"] for r in rows]))
    mean_N_off = float(np.mean([r["N"] for r in off_rows]))
    out = {
        "n_parts": len(p_main),
        "scoring": "score_reading (kural c cevap başına, d küresel) — spesifikasyon uyumlu",
        "a0_main_dedup_on": {"mean_p": float(p_main.mean()),
                             "median_p": float(np.median(p_main)),
                             "ks_p": _ks(p_main), "mean_N_per_part": mean_N_on},
        "wrong_keys_dedup_on": {"n_tests": int(len(p_wrong)),
                                "mean_p": float(p_wrong.mean()),
                                "ks_p": _ks(p_wrong)},
        "a0_main_dedup_off": {"mean_p": float(p_off.mean()),
                              "median_p": float(np.median(p_off)),
                              "ks_p": _ks(p_off),
                              "mean_neglog10p": float(np.mean(
                                  [r["neglog10p"] for r in off_rows])),
                              "mean_N_per_part": mean_N_off},
        "pass_mean_p_0.4_0.6": bool(0.4 <= p_main.mean() <= 0.6
                                    and 0.4 <= p_wrong.mean() <= 0.6),
        "pass_ks_0.01": bool(_ks(p_main) > 0.01 and _ks(p_wrong) > 0.01),
        "note": "kabul ölçütü: ortalama p 0.4–0.6 ve KS p > 0.01 (hem ana hem yanlış anahtarlar, "
                "yalnız A₀; tüm A modelleri için analyze.reading_rerun)",
    }
    dump_json(out, E0 / "E0.json")
    update_manifest(**{"e0.complete": out})
    record_duration("e0_complete", time.time() - t0)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
