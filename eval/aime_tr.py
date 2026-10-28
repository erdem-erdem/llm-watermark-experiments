"""Matematik kalite kontrolü (TÜBİTAK Bilim Olimpiyatı TR) — Bölüm 4.3, E2 adım 5.

Dataset: ytu-ce-cosmos/tubitak-science-olympiad-tr, split "ortaokul_matematik".
Soru alanı: question_latex; cevap alanı: answer_value.
YALNIZCA answer_value tamamen sayı olan satırlar kullanılır (doğrulanabilirlik);
süzülen TÜM satırlar kullanılır — örnekleme yok.

Satır sayısı × 8 koşul üretim: watermark'lı 4 anahtar (ana + 3 yanlış),
watermark'sız 4 tohum. T=0.8 top-p 0.95, max_new_tokens 4096, thinking kapalı.
Puanlama: son \\boxed{...} içeriği → sayı (int/float) → answer_value ile
karşılaştırma.
Ölçek (maliyet: satır × 8 koşul × 4096 token üretimi):
  configs/eval.yaml → aime_max_problems (varsayılan 100; 0/null = TÜM satırlar).
  Koşul bazlı paralel koşum: --only main,wrong1 ... (parça dosyaları
  results/E2/aime_parts/ altına yazılır ve her koşumda birleşik özet tazelenir).

Çıktılar: results/E2/aime_summary.json, aime_per_problem.csv (dosya adları
yer tutucu haritasıyla uyumlu kalsın diye korunmuştur; GPQA soruları gibi
hassas içerik dosyalara yazılmaz).

  python -m eval.aime_tr --config configs/eval.yaml [--smoke] [--only main,seed0]
"""

from __future__ import annotations

import json
import re
import sys
import time

import numpy as np

from gen.generate import DEFAULT_TEACHER, build_prompts
from wm.util import (RESULTS, dump_csv, dump_json, log_cmd, load_config,
                     parse_args, record_duration, update_manifest)

E2 = RESULTS / "E2"
# Talimat PROBLEMDEN SONRA gelir ve örnek biçim içerir (model uzun çözümün sonunda
# biçimi unutmasın). {problem} yer tutucusu .replace ile doldurulur.
AIME_INSTRUCTION = (
    "Aşağıdaki problemi adım adım çöz.\n\n{problem}\n\n"
    "ÖNEMLİ: Çözümün sonunda nihai cevabı yalnızca bir sayı olarak "
    "\\boxed{} içinde ver; örnek biçim: \\boxed{42}. "
    "Birim, açıklama ya da başka bir metin yazma.")

NUMERIC_RE = re.compile(r"^\s*-?\d+(?:[.,]\d+)?\s*$")


def is_numeric_answer(v) -> bool:
    """answer_value tamamen sayı mı? (int ya da ondalık; TR/US ayırıcı)"""
    if isinstance(v, (int, float)):
        return True
    if not isinstance(v, str):
        return False
    return bool(NUMERIC_RE.match(v))


def parse_num(v):
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def load_aime(cfg):
    """Olimpiyat verisini yükle → answer_value tamamen sayı olan satırlar.

    Ölçek sınırı: cfg["aime_max_problems"] > 0 ise satır sayısı ona indirilir
    (sabit tohumla karıştırılır — deterministik; 0/null = hepsi).
    """
    from datasets import load_dataset
    # config yalnızca "default"; ortaokul_matematik bir SPLIT'tir
    ds = load_dataset(cfg["aime_dataset"],
                      split=cfg.get("aime_split", "ortaokul_matematik"))
    rows = [r for r in ds
            if r.get("question_latex") and is_numeric_answer(r.get("answer_value"))]
    print(f"sayı-cevaplı satır: {len(rows)} / toplam {len(ds)}")
    if not rows:
        sys.exit("HATA: sayı-cevaplı satır yok — alan adlarını data/field_peek"
                 " tarzı kontrol edin (question_latex / answer_value).")
    cap = cfg.get("aime_max_problems", 100)
    if cap and len(rows) > cap:
        rng = np.random.default_rng(cfg.get("seed", 20260814))
        idx = sorted(rng.choice(len(rows), size=cap, replace=False).tolist())
        rows = [rows[i] for i in idx]
        print(f"aime_max_problems={cap}: {cap} problem seçildi "
             f"(0/null = tümü; configs/eval.yaml)")
    return rows


def _balanced_boxed(text: str):
    """Son \\boxed{...} içeriği, İÇ İÇE süslü parantezlerle (\\frac{1}{2},
    \\text{cm} ...). Eski regex ([^{}]*) iç içe parantezli her kutuyu
    'boxed yok' sayıyordu. Kapanmamış (kesilmiş) kutu için None."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth, out = start + len("\\boxed{"), 1, []
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(c)
        i += 1
    return None


_FRAC = re.compile(r"^(-?)\\d?frac\{(\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}$")


def parse_boxed_content(raw: str):
    """Kutu içeriğini sayıya çevir (birim/%/derece/kesir toleranslı); olmuyorsa None."""
    c = raw.replace("$", "")
    c = re.sub(r"\\text\{[^{}]*\}|\\mathrm\{[^{}]*\}", "", c)          # birimler
    c = re.sub(r"\\(?:,|;|:|!|\s)|\\%|%|\^\{?\\circ\}?|\\circ|\\cdot|\\left|\\right", "", c)
    c = c.replace(" ", "").replace("\u00a0", "").strip()
    m = _FRAC.match(c)
    if m:
        den = float(m.group(3))
        return (-1.0 if m.group(1) else 1.0) * float(m.group(2)) / den if den else None
    # virgül: binlik ayırıcıysa ("1,234") sil; ondalıksa ("3,14") noktaya çevir
    if "," in c:
        c = c.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", c) else c.replace(",", ".")
    try:
        return float(c)
    except ValueError:
        return None


def last_boxed_num(text: str):
    """(değer, boxed_var_mı). Kutu var ama sayı değilse (None, True)."""
    raw = _balanced_boxed(text)
    if raw is None:
        return None, False
    return parse_boxed_content(raw), True


def main():
    argv = parse_args("AIME 2025 TR kalite kontrolü")
    cfg = load_config(argv.config)
    log_cmd(sys.argv)
    t0 = time.time()
    smoke = argv.smoke

    problems = load_aime(cfg)
    if smoke:
        problems = problems[:3]
    n_prob = len(problems)
    print(f"Olimpiyat-TR: {n_prob} problem; alan adları: "
          f"{list(problems[0].keys())}")

    key_json = json.loads((RESULTS / "E0" / "selected_keys.json").read_text(
        encoding="utf-8"))
    main_key = key_json["main_key"]
    wrong_keys = key_json["wrong_keys"][:3]

    wm_keys = [("main", main_key)] + [(f"wrong{i+1}", k)
                                      for i, k in enumerate(wrong_keys)]
    nw_seeds = [cfg["seed"], cfg["seed"] + 41, cfg["seed"] + 42, cfg["seed"] + 43]
    conditions = ([("wm", kid, k) for kid, k in wm_keys] +
                  [("nowm", f"seed{i}", s) for i, s in enumerate(nw_seeds)])

    # koşul filtresi: GPU'ya bölünmüş koşumlar için (--only main,wrong1)
    only = None
    for i, a in enumerate(sys.argv):
        if a == "--only":
            only = set(sys.argv[i + 1].split(","))
    if only:
        conditions = [c for c in conditions if c[1] in only]
        if not conditions:
            sys.exit(f"bilinmeyen koşul(lar): {only} "
                     f"(main/wrong1-3/seed0-3)")

    from gen.generate import generate_batch
    prompt_records = []
    for i, p in enumerate(problems):
        problem_text = p.get("question_latex")
        prompt_records.append({
            # .format değil .replace: şablondaki \boxed{} süslü parantezleri
            # str.format ile çakışır (IndexError)
            "id": f"aime-{i:02d}", "prompt": AIME_INSTRUCTION.replace(
                "{problem}", problem_text)})
    if smoke:
        conditions = conditions[:2]

    teacher = cfg.get("teacher_model", DEFAULT_TEACHER)
    max_tok = cfg.get("aime_max_new_tokens", 4096)

    truth = {}
    for i, p in enumerate(problems):
        truth[f"aime-{i:02d}"] = parse_num(p["answer_value"])

    per_problem = []
    per_condition_acc = {}
    parts_dir = E2 / "aime_parts"
    for cond_name, kid, kv in conditions:
        if cond_name == "wm":
            recs = generate_batch(teacher, prompt_records, watermark=True,
                                  key=kv, key_id=kid, temperature=cfg["T"],
                                  top_p=cfg["top_p"], max_new_tokens=max_tok,
                                  seed=cfg["seed"], tp_size=cfg.get("tp_size", 1))
        else:
            recs = generate_batch(teacher, prompt_records, watermark=False,
                                  key=None, key_id=None, temperature=cfg["T"],
                                  top_p=cfg["top_p"], max_new_tokens=max_tok,
                                  seed=kv, tp_size=cfg.get("tp_size", 1))
        correct = {}
        no_boxed = 0
        for r in recs:
            val, has_boxed = last_boxed_num(r["answer"])
            if not has_boxed:
                no_boxed += 1
            tv = truth[r["id"]]
            ok = (val is not None and tv is not None
                  and abs(val - tv) <= 1e-6 * max(1.0, abs(tv)))
            correct[r["id"]] = ok
            n_tok = len(r.get("answer_ids") or [])
            per_problem.append({"problem_id": r["id"], "condition": kid,
                                "wm": cond_name == "wm", "correct": ok,
                                "has_boxed": has_boxed,
                                "boxed_numeric": val is not None,
                                "n_tokens": n_tok,
                                "truncated": n_tok >= max_tok - 1,
                                "tail": (r.get("answer") or "")[-120:].replace("\n", " ")})
        acc = float(np.mean(list(correct.values())))
        rows_k = [r for r in per_problem if r["condition"] == kid]
        per_condition_acc[kid] = {"accuracy": acc, "no_boxed_rate": no_boxed / len(recs),
                                  "truncated_rate": float(np.mean([r["truncated"] for r in rows_k])),
                                  "wm": cond_name == "wm"}
        print(f"  {kid}: doğruluk {acc:.3f}, boxed'sız %{100*no_boxed/len(recs):.0f}")
        # koşul parçası: GPU başına koşumlarda birleşirme için
        dump_json({"kid": kid, "condition": cond_name,
                   "wm": cond_name == "wm", "n": n_prob, "accuracy": acc,
                   "no_boxed_rate": no_boxed / len(recs),
                   "rows": [r for r in per_problem if r["condition"] == kid]},
                  parts_dir / f"{kid}.json")

    # ---- birleştirme: TÜM parçaları oku (paralel koşumlardan) ----
    per_condition_acc, per_problem_all = {}, []
    if parts_dir.is_dir():
        for pf in sorted(parts_dir.glob("*.json")):
            d = json.loads(pf.read_text(encoding="utf-8"))
            per_condition_acc[d["kid"]] = {"accuracy": d["accuracy"],
                                           "no_boxed_rate": d["no_boxed_rate"],
                                           "truncated_rate": float(np.mean(
                                               [bool(r.get("truncated")) for r in d["rows"]]))
                                           if d["rows"] else None,
                                           "wm": d["wm"]}
            per_problem_all.extend(d["rows"])

    wm_accs = [v["accuracy"] for v in per_condition_acc.values() if v["wm"]]
    nw_accs = [v["accuracy"] for v in per_condition_acc.values() if not v["wm"]]
    summary = {
        "n_problems": n_prob,
        "wm_mean": float(np.mean(wm_accs)), "wm_std": float(np.std(wm_accs)),
        "nowm_mean": float(np.mean(nw_accs)), "nowm_std": float(np.std(nw_accs)),
        "overlap": bool(set(wm_accs) & set(nw_accs)),
        "per_condition": per_condition_acc,
        "note": "n_problem küçük; farkın yorumu yazara aittir.",
    }
    dump_json(summary, E2 / "aime_summary.json")
    dump_csv(per_problem_all, E2 / "aime_per_problem.csv",
             fieldnames=["problem_id", "condition", "wm", "correct", "has_boxed",
                           "boxed_numeric", "n_tokens", "truncated", "tail"])
    update_manifest(**{"e2.aime": summary})
    record_duration("aime_eval", time.time() - t0)
    print(f"AIME: wm {summary['wm_mean']:.3f}±{summary['wm_std']:.3f} vs "
          f"nowm {summary['nowm_mean']:.3f}±{summary['nowm_std']:.3f} "
          f"({len(per_condition_acc)} koşul birleşik)")


if __name__ == "__main__":
    main()