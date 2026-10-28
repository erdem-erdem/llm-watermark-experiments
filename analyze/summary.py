"""SUMMARY.md — koşu özeti (kabul kapıları, E3 tablosu, süreler, ortam).

  python -m analyze.summary
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from wm.util import (RESULTS, dump_json, log_cmd, parse_args, read_csv,
                     record_duration, update_manifest)

R = RESULTS
THR = 6.0


def _num(x, nd=2):
    if x in ("", None):
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _has(*paths):
    return [str(p) for p in paths if Path(p).exists()]



def _get(csv_path, key):
    try:
        rows = read_csv(csv_path)
        for r in rows:
            if r.get("metric") == key:
                return r["value"]
    except Exception:
        pass
    return None


def write_summary():
    lines = ["# Deney Özeti (SUMMARY)\n"]
    lines.append("## Kabul kapıları\n")
    # E0 kapısı
    e0j = R / "E0" / "E0.json"
    t0 = R / "E0" / "T_E0.csv"
    if e0j.exists():
        j = json.loads(e0j.read_text(encoding="utf-8"))
        a, w = j["a0_main_dedup_on"], j["wrong_keys_dedup_on"]
        lines.append(f"- **E0**: A₀/ana anahtar ortalama p = {a['mean_p']:.2f} "
                     f"(KS p = {a['ks_p']:.2f}); yanlış anahtarlar ortalama p = "
                     f"{w['mean_p']:.2f} (KS p = {w['ks_p']:.4f}) → "
                     f"{'GEÇTİ' if j['pass_mean_p_0.4_0.6'] and j['pass_ks_0.01'] else 'KISMEN/KONTROL ET'} "
                     f"(beklenen: ortalama 0.4–0.6, KS p > 0.01)\n")
    elif t0.exists():
        mp = _get(t0, "mean_neglog10p")
        ok = None
        if mp not in ("", None):
            p = 10 ** (-float(mp))
            ok = 0.4 <= p <= 0.6
        lines.append(f"- **E0**: parça p ortalaması "
                     f"{_num(10 ** (-float(mp))) if mp not in ('', None) else '—'} "
                     f"→ {'GEÇTİ' if ok else 'KONTROL ET'} (beklenen aralık 0.4–0.6)\n")
    # E1 seçimi
    sel = R / "E1" / "auto_selection.json"
    if sel.exists():
        s = json.loads(sel.read_text(encoding="utf-8"))["auto_selected"]
        lines.append(f"- **E1**: seçilen ayar `{s['model']}` T={s['T']} "
                     f"(medyan -log₁₀p = {_num(s['median_neglog10p'])}; "
                     f"kural: {s.get('rule', 'KARAR-8')})\n")
        if s.get("rule", "KARAR-8") != "KARAR-8":
            lines.append("  - ⚠ **SAPMA:** KARAR-8 (benzerlik ≥ 0.85 ve uzunluk "
                         "geçişi ≥ %90) hiçbir ayarla sağlanamadı; otomatik "
                         "\"verim\" fallback'i uygulandı (ayrıntı: "
                         "`E1/SWEEP_REPORT.md`). Yazıda belirtilmeli.\n")
    # E2 ana sayılar
    t2 = R / "E2" / "T_E2.csv"
    if t2.exists():
        for r in read_csv(t2):
            lines.append(f"- **E2** {r['set']}: n={r['n']}, "
                         f"medyan -log₁₀p={_num(r['median_neglog10p'])}, "
                         f"n=400 oran={_num(r.get('rate_at_n400'), 3)}\n")
    # E3/E4
    t3 = R / "E3" / "T_E3.csv"
    if t3.exists():
        lines.append("\n## E3 koşulları\n\n| model | -log₁₀p@n100 | "
                     "-log₁₀p@n400 | eval_loss | GPQA |\n|---|---|---|---|---|\n")
        for r in read_csv(t3):
            lines.append(f"| {r['model']} | {_num(r.get('neglog10p@n100'))} | "
                         f"{_num(r.get('neglog10p@n400'))} | "
                         f"{_num(r.get('eval_loss'), 4)} | "
                         f"{_num(r.get('gpqa_acc'), 3)} |\n")
    t4 = R / "E4" / "T_E4.csv"
    if t4.exists():
        lines.append("\n## E4 zinciri\n\n| halka | ne | -log₁₀p | "
                     "yanlış anahtar medyan |\n|---|---|---|---|\n")
        for r in read_csv(t4):
            lines.append(f"| H{r['ring']} | {r['what']} | {_num(r['neglog10p'])} | "
                         f"{_num(r.get('wrong_key_median'))} |\n")
    # süreler
    man = R / "run_manifest.json"
    if man.exists():
        d = json.loads(man.read_text(encoding="utf-8"))
        durs = d.get("durations", {})
        if durs:
            lines.append("\n## Süreler\n")
            for k, v in sorted(durs.items()):
                lines.append(f"- {k}: {v / 60:.1f} dk\n")
        lines.append("\n## Ortam\n")
        env = d.get("environment", {})
        for k, v in env.items():
            lines.append(f"- {k}: {v}\n")
    (R / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("SUMMARY.md yazıldı")


def main():
    argv = parse_args("koşu özeti")
    log_cmd(sys.argv)
    t0 = time.time()
    write_summary()
    update_manifest(**{"summary.done": True})
    record_duration("summary", time.time() - t0)


if __name__ == "__main__":
    main()