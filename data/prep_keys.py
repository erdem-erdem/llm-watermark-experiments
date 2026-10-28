"""Anahtar seçimi — SPEC.md Bölüm 3.6, E0 adım 1.

20 aday anahtar (sabit tohumdan) D_orig ve D_clean cevaplarında metin
tespitinden geçirilir; örnek p-değerlerinin tekdüze dağılıma uyumu KS ile
ölçülür; KS p en yüksek anahtar = ana anahtar; kalanlardan 5 yanlış anahtar.
results/E0/key_selection.csv'ye yazılır.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kstest

from wm.detect import detect_text
from wm.util import (RESULTS, ROOT, dump_csv, dump_json, log_cmd, load_config,
                     parse_args, read_jsonl, record_duration, update_manifest)

DATA_DIR = ROOT / "data"


def main():
    argv = parse_args("Anahtar seçimi")
    cfg = load_config(argv.config)
    log_cmd(sys.argv)
    t0 = __import__("time").time()
    smoke = argv.smoke

    # İDEMPOTENT: ana anahtar bir kez seçilir. Kalibrasyon verisi D_clean varsa
    # onu da içerir; yeniden seçim anahtarı DEĞİŞTİRİR ve o anahtarla zaten
    # işaretlenmiş verinin tespitini sessizce bozar.
    sel_path = RESULTS / "E0" / "selected_keys.json"
    if sel_path.exists() and not smoke and not argv.force:
        import json
        k = json.loads(sel_path.read_text(encoding="utf-8"))
        print(f"anahtar zaten seçilmiş — atlanıyor: ana anahtar "
              f"{int(k['main_key']):016x} (zorlamak için --force; "
              f"bağımlı tüm üretimleri geçersiz kılar)")
        return

    n_keys = 5 if smoke else cfg.get("n_candidates", 20)
    n_wrong = 5
    n_cal = 50 if smoke else cfg.get("n_calibration", 500)

    train = list(read_jsonl(DATA_DIR / "train.jsonl"))
    clean = list(read_jsonl(DATA_DIR / "D_clean.jsonl")) if (DATA_DIR / "D_clean.jsonl").exists() else []
    cal_recs = train[:n_cal]
    if clean:
        cal_recs = cal_recs + clean[:n_cal]

    rng = random.Random(cfg.get("seed", 20260814))
    keys = [rng.getrandbits(64) for _ in range(n_keys)]

    rows = []
    for key in keys:
        ps = []
        for rec in cal_recs:
            r = detect_text(rec["prompt_ids"], rec["answer_ids"], key, k=2)
            ps.append(10.0 ** (-r.neglog10p) if r.N else 0.5)
        ks_stat, ks_p = kstest(ps, "uniform")
        rows.append({"key": key, "ks_stat": ks_stat, "ks_p": ks_p,
                     "mean_p": float(np.mean(ps)), "n": len(ps)})
        print(f"  key={key:016x} KS p={ks_p:.3f} mean p={np.mean(ps):.3f}")

    rows.sort(key=lambda r: -r["ks_p"])
    main_key = rows[0]["key"]
    wrong_keys = [r["key"] for r in rows[1:1 + n_wrong]]
    out_dir = RESULTS / "E0"
    dump_csv(rows, out_dir / "key_selection.csv")
    dump_json({"main_key": main_key, "wrong_keys": wrong_keys,
               "selection_rule": "KS p en yüksek", "n_candidates": n_keys},
              out_dir / "selected_keys.json")
    update_manifest(**{"keys.main": main_key, "keys.wrong": wrong_keys,
                       "keys.n_candidates": n_keys})
    record_duration("key_selection", __import__("time").time() - t0)
    print(f"ana anahtar: {main_key:016x}; yanlış anahtarlar: "
          f"{[f'{k:016x}' for k in wrong_keys]}")


if __name__ == "__main__":
    main()