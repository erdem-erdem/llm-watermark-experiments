"""E1 sweep raporu — KARAR-8 otomatik seçim + GATE (§5-E1).

E1/sweep.csv'den:
  - fig_E1_1: (T, model) başına -log10(p) dağılımı (scatter/box)
  - T_E1.csv: ayar başına özet (medyan -log10p, benzerlik, uzunluk geçiş)
  - KARAR-8 seçimi: benzerlik >= 0.85 VE uzunluk geçiş >= %90 olanlar
    içinden en yüksek medyan -log10p
  - SWEEP_REPORT.md + E1/GATE.md (onay kilit noktası)

  python -m analyze.e1_report [--smoke]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from wm.util import (RESULTS, dump_csv, dump_json, log_cmd, parse_args,
                     read_csv, record_duration, update_manifest)

E1 = RESULTS / "E1"
SIM_MIN = 0.85
PASS_MIN = 0.90
DETECT_MIN = 6.0   # -log10(p) eşiği (configs/detect.yaml: threshold_neglog10p)


def summarize(rows):
    by = {}
    for r in rows:
        k = (r["model"], float(r["T"]))
        by.setdefault(k, []).append(r)
    out = []
    for (model, T), g in sorted(by.items()):
        nl = [float(r["neglog10p"]) for r in g if r["neglog10p"] not in ("", None)]
        sims = [float(r["similarity"]) for r in g
                if r.get("similarity") not in ("", None)]
        lrs = [float(r["length_ratio"]) for r in g
               if r.get("length_ratio") not in ("", None)]
        med = float(np.median(nl)) if nl else float("nan")
        mean_sim = float(np.mean(sims)) if sims else float("nan")
        ok = [1.0 if (0.75 <= lr <= 1.25) else 0.0 for lr in lrs]
        pass_rate = float(np.mean(ok)) if ok else float("nan")
        # satır düzeyi "verim": aynı anda benzer + uzunluk bandında + tespit edilebilir
        def _ok(r):
            try:
                return (str(r.get("filtered")).lower() != "true"
                        and float(r["similarity"]) >= SIM_MIN
                        and float(r["neglog10p"]) >= DETECT_MIN)
            except (ValueError, KeyError, TypeError):
                return False
        yield_rate = float(np.mean([1.0 if _ok(r) else 0.0 for r in g]))
        lq = (np.percentile(lrs, [10, 50, 90]) if lrs else [float("nan")] * 3)
        out.append({"model": model, "T": T, "n": len(g), "median_neglog10p": med,
                    "yield": yield_rate,
                    "lr_q10": float(lq[0]), "lr_median": float(lq[1]), "lr_q90": float(lq[2]),
                    "share_short": float(np.mean([lr < 0.75 for lr in lrs])) if lrs else float("nan"),
                    "share_long": float(np.mean([lr > 1.25 for lr in lrs])) if lrs else float("nan"),
                    "mean_neglog10p": float(np.mean(nl)) if nl else float("nan"),
                    "q25": float(np.percentile(nl, 25)) if nl else float("nan"),
                    "q75": float(np.percentile(nl, 75)) if nl else float("nan"),
                    "mean_similarity": mean_sim,
                    "length_pass_rate": pass_rate,
                    "eligible": bool(mean_sim >= SIM_MIN and pass_rate >= PASS_MIN)})
    return out


def select_setting(summary_rows):
    """Seçim kuralı (tamamen otomatik, belirlenimli).

    1) KARAR-8: benzerlik >= 0.85 VE uzunluk geçiş >= %90 → en yüksek medyan.
    2) Hiçbir ayar sağlamıyorsa FALLBACK ("verim"): satırların en büyük payının
       AYNI ANDA benzer (>=0.85), uzunluk bandında (0.75-1.25) ve tespit
       edilebilir (-log10p >= 6) olduğu ayar; eşitlikte yüksek medyan.
    Dönüş: (satır, kural_adı) ya da (None, None).
    """
    elig = [r for r in summary_rows if r["eligible"]]
    if elig:
        return max(elig, key=lambda r: r["median_neglog10p"]), "KARAR-8"
    cand = [r for r in summary_rows if r["mean_similarity"] >= SIM_MIN]
    if not cand:
        return None, None
    return (max(cand, key=lambda r: (r["yield"], r["median_neglog10p"])),
            "fallback_verim")


def write_report(summary_rows, sel, rule, smoke):
    med_txt = "\n".join(
        f"| {r['model']} | {r['T']} | {r['n']} | {r['median_neglog10p']:.2f} | "
        f"{r['mean_similarity']:.3f} | {100 * r['length_pass_rate']:.0f}% | "
        f"{100 * r['yield']:.0f}% | "
        f"{r['lr_q10']:.2f} / {r['lr_median']:.2f} / {r['lr_q90']:.2f} | "
        f"{100 * r['share_short']:.0f}% / {100 * r['share_long']:.0f}% | "
        f"{'uygun' if r['eligible'] else '—'} |" for r in summary_rows)
    if sel and rule == "KARAR-8":
        sel_txt = (f"**KARAR-8 seçimi:** model `{sel['model']}`, T={sel['T']} "
                   f"(medyan -log10p = {sel['median_neglog10p']:.2f}, "
                   f"benzerlik {sel['mean_similarity']:.3f}, "
                   f"geçiş %{100 * sel['length_pass_rate']:.0f}).")
    elif sel:
        best_pass = max(r["length_pass_rate"] for r in summary_rows)
        sel_txt = (
            "**SAPMA — KARAR-8 sağlanamadı** (benzerlik >= 0.85 VE uzunluk "
            f"geçişi >= %90 koşulunu hiçbir ayar karşılamıyor; en iyi geçiş "
            f"%{100 * best_pass:.0f}). Otomatik FALLBACK uygulandı: satırların "
            "en büyük payının aynı anda benzer (>=0.85), uzunluk bandında "
            "(0.75-1.25) ve tespit edilebilir (-log10p >= 6) olduğu ayar.\n\n"
            f"**Seçilen:** model `{sel['model']}`, T={sel['T']} — verim "
            f"%{100 * sel['yield']:.0f}, medyan -log10p = "
            f"{sel['median_neglog10p']:.2f}, benzerlik "
            f"{sel['mean_similarity']:.3f}, uzunluk geçişi "
            f"%{100 * sel['length_pass_rate']:.0f}. Yazı bu sapmayı belirtmelidir.")
    else:
        sel_txt = ("**Seçim yapılamadı:** hiçbir ayarın ortalama benzerliği "
                   ">= 0.85 değil.")
    Path(E1 / "SWEEP_REPORT.md").write_text(
        f"# E1 Tarama Raporu\n\n{sel_txt}\n\n"
        "| model | T | n | medyan -log10p | benzerlik | uzunluk geçiş | verim | "
        "uzunluk oranı q10/medyan/q90 | kısa% / uzun% | uygun |\n"
        f"|---|---|---|---|---|---|---|---|---|---|\n{med_txt}\n\n"
        "Notlar:\n- uzunluk oranı = çıktı karakter sayısı / kaynak karakter sayısı; "
        "band 0.75-1.25. 'kısa%' = bandın altı, 'uzun%' = bandın üstü.\n"
        "- verim = satırın aynı anda benzer (>=0.85) + bantta + -log10p >= 6 olma payı.\n"
        "- filtreli (uzunluk dışı / think_leak) satırlar İŞARETLENİR, eğitimden "
        "ÇIKARILMAZ; oranı raporlanır (Bölüm 5-E1.5).\n"
        "- Seçim otomatiktir (`E1/selected.json`, auto=true). Elle bir "
        "`selected.json` varsa üzerine yazılmaz.\n",
        encoding="utf-8")
    (E1 / "GATE.md").write_text(
        "# E1 SEÇİMİ (otomatik)\n\n" + sel_txt + "\n\n"
        "Pipeline durmaz: `results/E1/selected.json` otomatik yazıldı ve "
        "`run_e1_full.sh` devam eder. Elle değiştirmek için `selected.json`'u "
        "kendiniz yazın (varsa ezilmez) ve `bash scripts/run_e1_full.sh` çalıştırın.\n",
        encoding="utf-8")


def _is_auto(path) -> bool:
    import json
    try:
        return bool(json.loads(Path(path).read_text(encoding="utf-8")).get("auto"))
    except Exception:  # noqa: BLE001
        return False


def main():
    argv = parse_args("E1 tarama raporu")
    log_cmd(sys.argv)
    t0 = time.time()
    csv = E1 / "sweep.csv"
    parts = sorted((E1 / "sweep_parts").glob("sweep_*.csv")) \
        if (E1 / "sweep_parts").is_dir() else []
    if not csv.exists() and not parts:
        print(f"sweep.csv yok: {csv}")
        return
    rows = []
    for p in parts:
        rows.extend(read_csv(p))
    if csv.exists():
        rows.extend(read_csv(csv))
    # (id, model, T) başına tek satır (parça + birleşik çift yazımı önle)
    seen, uniq = set(), []
    for r in rows:
        k = (r.get("id"), r.get("model"), r.get("T"))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    rows = uniq
    if argv.smoke:
        rows = rows[:200]
    summary_rows = summarize(rows)
    dump_csv(summary_rows, E1 / "T_E1.csv",
             fieldnames=["model", "T", "n", "median_neglog10p", "mean_neglog10p",
                         "q25", "q75", "mean_similarity", "length_pass_rate",
                         "yield", "lr_q10", "lr_median", "lr_q90", "share_short",
                         "share_long", "eligible"])
    sel, rule = select_setting(summary_rows)
    write_report(summary_rows, sel, rule, argv.smoke)
    if sel:
        dump_json({"auto_selected": {"model": sel["model"], "T": sel["T"],
                                     "median_neglog10p": sel["median_neglog10p"],
                                     "yield": sel["yield"], "rule": rule,
                                     "approved": True}}, E1 / "auto_selection.json")
        # Tam otomatik akış: onay dosyası yazılır. Elle yazılmış bir dosya VARSA
        # ezilmez (kullanıcı kararı önceliklidir).
        sel_path = E1 / "selected.json"
        if sel_path.exists() and not _is_auto(sel_path):
            print(f"selected.json elle yazılmış — korunuyor: {sel_path}")
        else:
            dump_json({"model": sel["model"], "T": sel["T"], "approved": True,
                       "auto": True, "rule": rule,
                       "deviation_from_karar8": rule != "KARAR-8",
                       "yield": sel["yield"],
                       "median_neglog10p": sel["median_neglog10p"],
                       "mean_similarity": sel["mean_similarity"],
                       "length_pass_rate": sel["length_pass_rate"]}, sel_path)
        update_manifest(**{"e1.selection": {"model": sel["model"], "T": sel["T"],
                                            "rule": rule}})
        print(f"E1 seçimi ({rule}): {sel['model']} T={sel['T']}")
    record_duration("e1_report", time.time() - t0)
    print("E1 raporu tamam (results/E1/SWEEP_REPORT.md)")


if __name__ == "__main__":
    main()