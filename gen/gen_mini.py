"""Mini deneme: benchmark işaretleme (ÖRNEK 4.1) — TÜBİTAK Bilim Olimpiyatı TR.

Veri: ytu-ce-cosmos/tubitak-science-olympiad-tr, split "ortaokul_matematik"
(599 soru; answer_value tamamen sayı olan satırlardan `benchmark_n` (varsayılan 100; eval ile AYNI problemler) kadarı,
sabit tohumla; eval.aime_tr.load_aime ile aynı yükleyici).

Adımlar:
  1. Her problem Qwen3.5-9B ile ana anahtarla post-hoc YENİDEN YAZILIR
     (E1 talimatı + "Problemi çözme"). Sayılar ve $...$ matematik parçaları
     orijinalde nasılsa korunmalı: otomatik denetim, bozulanlar en fazla 3 kez
     yeniden üretilir (ek cümle: "Sayıları ve matematiksel ifadeleri değiştirme."),
     yine bozuk kalanlar RAPORLANIR.
  2. Her problemin −log10 p ve N değeri; 100 problemin BİRLİKTE (küresel
     tekilleştirmeli) N ve −log10 p değeri.
  3. Benchmark hâlâ aynı şeyi ölçüyor mu? Orijinal ve yeniden yazılmış problem
     watermark'sız çözülür (\\boxed{} talimatıyla, eval ile AYNI talimat/ayrıştırıcı);
     doğruluklar ve uyum raporlanır.

ÖNEMLİ: temperature=0 kullanılmaz (tek-sıcak dağılım → iz oluşmaz); yeniden yazım
T = cfg["rewrite_T"] (varsayılan 1.0), çözüm T = eval T (0.8).

Çıktılar (results/mini/): benchmark_marking.csv, benchmark_marking.json,
benchmark_marking.md (2 örnek çift).

  python -m gen.gen_mini --config configs/mini.yaml [--smoke]
"""

from __future__ import annotations

import json
import re
import sys
import time

import numpy as np

from eval.aime_tr import AIME_INSTRUCTION, last_boxed_num, load_aime, parse_num
from gen.generate import REWRITE_INSTRUCTION, generate_batch
from gen.mini_util import scored_tokens, summarize
from wm.detect import detect_dataset
from wm.util import (RESULTS, ROOT, dump_csv, dump_json, load_config, log_cmd,
                     parse_args, record_duration, update_manifest)

MINI = RESULTS / "mini"
EXTRA = " Sayıları ve matematiksel ifadeleri değiştirme."
NO_SOLVE = " Problemi ÇÖZME; yalnızca yeniden yaz."
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
MATH_RE = re.compile(r"\$[^$]+\$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def missing_items(original: str, rewrite: str):
    """Orijinaldeki sayılar ve $...$ parçalarından yeniden yazımda bulunmayanlar."""
    r_nums = set(NUM_RE.findall(rewrite))
    miss = [n for n in sorted(set(NUM_RE.findall(original))) if n not in r_nums]
    r_norm = _norm(rewrite)
    miss += [m for m in MATH_RE.findall(original) if _norm(m) not in r_norm]
    return miss


def solve(problems_text, truth, cfg, ecfg, tag):
    """Watermark'sız çözüm (eval ile aynı talimat/ayrıştırıcı). → {id: (ok, boxed)}"""
    recs = [{"id": i, "prompt": AIME_INSTRUCTION.replace("{problem}", t)}
            for i, t in problems_text.items()]
    out = generate_batch(ecfg.get("teacher_model", cfg["model"]), recs,
                         watermark=False, key=None, key_id=None,
                         temperature=ecfg["T"], top_p=ecfg["top_p"],
                         max_new_tokens=ecfg.get("aime_max_new_tokens", 4096),
                         seed=ecfg["seed"], tp_size=cfg.get("tp_size", 1))
    res = {}
    for r in out:
        val, boxed = last_boxed_num(r["answer"])
        tv = truth[r["id"]]
        ok = val is not None and tv is not None and abs(val - tv) <= 1e-6 * max(1.0, abs(tv))
        res[r["id"]] = (bool(ok), bool(boxed))
    print(f"  çözüm[{tag}]: doğruluk {np.mean([v[0] for v in res.values()]):.3f}, "
          f"boxed {np.mean([v[1] for v in res.values()]):.2f}")
    return res


def main():
    argv = parse_args("benchmark işaretleme mini denemesi")
    cfg = load_config(argv.config)
    ecfg = load_config(ROOT / "configs" / "eval.yaml")
    log_cmd(sys.argv)
    t0 = time.time()
    sel = json.loads((RESULTS / "E0" / "selected_keys.json").read_text(encoding="utf-8"))
    main_key = int(sel["main_key"])
    k = cfg.get("k", 2)
    MINI.mkdir(parents=True, exist_ok=True)

    n = 3 if argv.smoke else cfg.get("benchmark_n", 100)
    ecfg["aime_max_problems"] = n
    probs = load_aime(ecfg)
    texts = {f"prob-{i:02d}": p["question_latex"] for i, p in enumerate(probs)}
    truth = {f"prob-{i:02d}": parse_num(p["answer_value"]) for i, p in enumerate(probs)}

    # 1) yeniden yazım + sayı/ifade denetimi (<=3 yeniden deneme)
    final, attempts, miss = {}, {i: 0 for i in texts}, {}
    todo = list(texts)
    for attempt in range(4):                       # 1 ilk + 3 yeniden deneme
        if not todo:
            break
        instr = REWRITE_INSTRUCTION + NO_SOLVE + (EXTRA if attempt > 0 else "")
        recs = [{"id": i, "prompt": f"{instr}\n\n{texts[i]}"} for i in todo]
        batch = generate_batch(cfg["model"], recs, watermark=True, key=main_key,
                               key_id="main", temperature=cfg.get("rewrite_T", 1.0),
                               top_p=cfg.get("top_p", 0.95),
                               max_new_tokens=cfg.get("benchmark_max_new_tokens", 700),
                               seed=cfg.get("seed", 20260814) + attempt, k=k,
                               tp_size=cfg.get("tp_size", 1))
        nxt = []
        for r in batch:
            attempts[r["id"]] = attempt
            m = missing_items(texts[r["id"]], r["answer"])
            final[r["id"]] = r                      # son deneme saklanır
            miss[r["id"]] = m
            if m:
                nxt.append(r["id"])
        print(f"  deneme {attempt + 1}: {len(todo)} problem, bozulan {len(nxt)}")
        todo = nxt

    # 2) tespit: problem başına + 30'u birlikte
    recs_final = [final[i] for i in texts]
    comb, _ = detect_dataset(recs_final, main_key, k)
    rows = []
    for i in texts:
        r = final[i]
        st = scored_tokens(list(r["answer_ids"]), len(r["prompt_ids"]), main_key, k)
        s = summarize([x for _, x in st])
        rows.append({"id": i, "N": s["N"], "neglog10p": s["neglog10p"],
                     "retries": attempts[i], "numbers_ok": not miss[i],
                     "missing": ";".join(miss[i]),
                     "len_ratio": len(r["answer"]) / max(1, len(texts[i]))})

    # 3) çözülebilirlik: orijinal vs yeniden yazılmış (watermark'sız, boxed talimatlı)
    sol_o = solve(texts, truth, cfg, ecfg, "orijinal")
    sol_r = solve({i: final[i]["answer"] for i in texts}, truth, cfg, ecfg, "yeniden-yazım")
    for row in rows:
        i = row["id"]
        row["orig_correct"], row["orig_boxed"] = sol_o[i]
        row["rew_correct"], row["rew_boxed"] = sol_r[i]
    dump_csv(rows, MINI / "benchmark_marking.csv",
             ["id", "N", "neglog10p", "retries", "numbers_ok", "missing", "len_ratio",
              "orig_correct", "orig_boxed", "rew_correct", "rew_boxed"])

    ok_rows = [r for r in rows if r["numbers_ok"]]
    summary = {
        "n_problems": len(rows), "n_numbers_ok": len(ok_rows),
        "n_corrupt_after_retries": len(rows) - len(ok_rows),
        "combined_N": comb.N, "combined_neglog10p": comb.neglog10p,
        "median_per_problem_neglog10p": float(np.median([r["neglog10p"] for r in rows])),
        "median_N": float(np.median([r["N"] for r in rows])),
        "acc_original": float(np.mean([r["orig_correct"] for r in rows])),
        "acc_rewritten": float(np.mean([r["rew_correct"] for r in rows])),
        "agree_correct_both": float(np.mean([r["orig_correct"] == r["rew_correct"]
                                             for r in rows])),
        "T_rewrite": cfg.get("rewrite_T", 1.0), "model": cfg["model"],
        "dataset": ecfg["aime_dataset"], "split": ecfg.get("aime_split"),
    }
    dump_json(summary, MINI / "benchmark_marking.json")

    # 2 örnek çift (sayıları korunmuş, orta uzunlukta, iz belirgin)
    cand = sorted([r for r in ok_rows if 200 <= len(texts[r["id"]]) <= 600],
                  key=lambda r: -r["neglog10p"])[:2] or ok_rows[:2]
    md = ["# Benchmark işaretleme — örnek çiftler\n",
          f"{summary['n_problems']} problem birlikte: N = {comb.N}, "
          f"−log₁₀(p) = {comb.neglog10p:.1f}; sayıları bozulan: "
          f"{summary['n_corrupt_after_retries']}\n"]
    for r in cand:
        md.append(f"\n## {r['id']} — N={r['N']}, −log₁₀(p)={r['neglog10p']:.1f}\n"
                  f"**Orijinal**\n\n{texts[r['id']]}\n\n**Yeniden yazım**\n\n"
                  f"{final[r['id']]['answer'].strip()}\n")
    (MINI / "benchmark_marking.md").write_text("\n".join(md), encoding="utf-8", newline="")
    update_manifest(**{"mini.benchmark": summary})
    record_duration("gen_mini", time.time() - t0)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
