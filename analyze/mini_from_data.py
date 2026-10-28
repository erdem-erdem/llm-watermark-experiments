"""Mini denemeler — MEVCUT verilerden (GPU gerekmez).

  guess_pairs        MİNİ DENEME 2.1 "Görebiliyor musunuz?": aynı prompt'a
                     watermark'lı (D_gen) ve watermark'sız (D_clean) cevap yan yana,
                     sırası sabit tohumla karıştırılmış; cevap + p-değerleri ayrı.
  same_prompt_twice  aynı prompt, iki üretim: watermark'lı + aynı anahtar → birebir
                     aynı metin; watermark'sız + farklı tohum → farklı metin.
  fig13              GÖRSEL 1.3 tarzı: bir D_gen cevabı ve aynı prompt'un D_clean
                     cevabı için token başına skor + kümülatif −log10 p.
  example_1_1        ÖRNEK 1.1: E1 tam verisinden 2 kısa (kaynak, yeniden yazım) çifti;
                     anlam benzerliği (e5) bulunursa eklenir.

Seçim ölçütleri kodda açıktır; tohum sabit → deterministik. Hiçbir metin elle seçilmez.

  python -m analyze.mini_from_data
Çıktılar (results/mini/): guess_pairs.{md,csv}, same_prompt_twice.{md,json},
  fig13_tokens.csv, fig13_curve.csv, example_1_1.{md,csv}
"""

from __future__ import annotations

import json
import random
import sys
import time

import numpy as np

from analyze.e2_complete import _per_record
from gen.mini_util import cum_curve, scored_tokens, summarize, token_pieces
from wm.util import (RESULTS, ROOT, dump_csv, dump_json, log_cmd, parse_args,
                     read_jsonl, record_duration, update_manifest)

DATA = ROOT / "data"
MINI = RESULTS / "mini"
SEED = 20260814


def _keys():
    s = json.loads((RESULTS / "E0" / "selected_keys.json").read_text(encoding="utf-8"))
    return int(s["main_key"]), [int(x) for x in s["wrong_keys"]]


def _stat(rec, key):
    sc = _per_record(rec, key)
    return summarize(sc)


def _clean_text(r, lo=120, hi=420):
    """Okunabilir örnek: orta uzunluk, kod bloğu/URL yok."""
    a = r["answer"]
    return (lo <= len(r["answer_ids"]) <= hi and "```" not in a
            and "http" not in a and len(r["prompt"]) <= 220)


def guess_pairs(main_key, wrong_keys, tok):
    gen = {r["id"]: r for r in read_jsonl(DATA / "D_gen.jsonl")}
    clean = {r["id"]: r for r in read_jsonl(DATA / "D_clean.jsonl")}
    rng = random.Random(SEED)
    cands = []
    for i, g in gen.items():
        c = clean.get(i)
        if not c or g["topic"] == "matematik_kod" or not (_clean_text(g) and _clean_text(c)):
            continue
        if abs(len(g["answer_ids"]) - len(c["answer_ids"])) > 0.15 * len(c["answer_ids"]):
            continue
        cands.append(i)
    rng.shuffle(cands)
    picked, want = [], {"yaratici": 1, "olgusal": 2}
    for i in cands:                                   # ilk uygun olanlar (karışık sırayla)
        sg = _stat(gen[i], main_key)
        sc = _stat(clean[i], main_key)
        if 8 <= sg["neglog10p"] <= 60 and sc["neglog10p"] < 1.0 and want.get(gen[i]["topic"], 0) > 0:
            want[gen[i]["topic"]] -= 1
            picked.append((i, sg, sc))
        if len(picked) == 3:
            break
    rows, md = [], ["# MİNİ DENEME 2.1 — Görebiliyor musunuz?\n"]
    for n, (i, sg, sc) in enumerate(picked, 1):
        wm_first = rng.random() < 0.5
        a, b = (gen[i], clean[i]) if wm_first else (clean[i], gen[i])
        wr = [_stat(gen[i], wk)["neglog10p"] for wk in wrong_keys[:2]]
        md.append(f"\n## Soru {n}  ·  konu: {gen[i]['topic']}\n**Prompt:** {gen[i]['prompt']}\n\n"
                  f"**Cevap A**\n\n{a['answer'].strip()}\n\n**Cevap B**\n\n{b['answer'].strip()}\n\n"
                  f"_Cevap:_ **{'A' if wm_first else 'B'}** watermark'lı — "
                  f"N={sg['N']}, −log₁₀(p)={sg['neglog10p']:.1f} (yanlış anahtarlarla "
                  f"{wr[0]:.2f} ve {wr[1]:.2f}); "
                  f"**{'B' if wm_first else 'A'}** watermark'sız — N={sc['N']}, "
                  f"−log₁₀(p)={sc['neglog10p']:.2f}.\n")
        rows.append({"q": n, "id": i, "topic": gen[i]["topic"],
                     "wm_is": "A" if wm_first else "B",
                     "wm_N": sg["N"], "wm_neglog10p": sg["neglog10p"],
                     "wm_wrong1": wr[0], "wm_wrong2": wr[1],
                     "clean_N": sc["N"], "clean_neglog10p": sc["neglog10p"]})
    dump_csv(rows, MINI / "guess_pairs.csv",
             ["q", "id", "topic", "wm_is", "wm_N", "wm_neglog10p", "wm_wrong1",
              "wm_wrong2", "clean_N", "clean_neglog10p"])
    (MINI / "guess_pairs.md").write_text("\n".join(md), encoding="utf-8", newline="")
    print(f"  guess_pairs: {len(rows)} çift")


def same_prompt_twice(main_key):
    def by_id(path):
        d = {}
        for r in read_jsonl(path):
            d.setdefault(r["id"], []).append(r)
        return {i: v for i, v in d.items() if len(v) >= 2}
    wm, nowm = by_id(DATA / "diversity_wm.jsonl"), by_id(DATA / "diversity_nowm.jsonl")
    rng = random.Random(SEED + 1)
    ids = [i for i in wm if i in nowm and _clean_text(wm[i][0]) and _clean_text(nowm[i][0], 120, 420)
           and wm[i][0]["topic"] != "matematik_kod"]
    rng.shuffle(ids)
    out, md = [], ["# Aynı prompt, iki üretim\n"]
    for n, i in enumerate(ids[:3], 1):
        w1, w2 = wm[i][:2]
        n1, n2 = nowm[i][:2]
        row = {"id": i, "prompt": w1["prompt"],
               "wm_identical": w1["answer_ids"] == w2["answer_ids"],
               "wm_run1": _stat(w1, main_key), "wm_run2": _stat(w2, main_key),
               "nowm_identical": n1["answer_ids"] == n2["answer_ids"],
               "nowm_run1": _stat(n1, main_key), "nowm_run2": _stat(n2, main_key)}
        out.append(row)
        md.append(f"\n## Prompt {n}\n{w1['prompt']}\n\n"
                  f"**Watermark'lı, aynı anahtar** — birebir aynı mı: **{row['wm_identical']}** "
                  f"(−log₁₀p {row['wm_run1']['neglog10p']:.1f} / {row['wm_run2']['neglog10p']:.1f})\n\n"
                  f"1) {w1['answer'].strip()[:500]}\n\n2) {w2['answer'].strip()[:500]}\n\n"
                  f"**Watermark'sız, farklı tohum** — birebir aynı mı: **{row['nowm_identical']}** "
                  f"(−log₁₀p {row['nowm_run1']['neglog10p']:.2f} / {row['nowm_run2']['neglog10p']:.2f})\n\n"
                  f"1) {n1['answer'].strip()[:500]}\n\n2) {n2['answer'].strip()[:500]}\n")
    dump_json({"examples": out,
               "overall_same_rate_wm": float(np.mean([v[0]["answer_ids"] == v[1]["answer_ids"]
                                                      for v in wm.values()])),
               "overall_same_rate_nowm": float(np.mean([v[0]["answer_ids"] == v[1]["answer_ids"]
                                                        for v in nowm.values()]))},
              MINI / "same_prompt_twice.json")
    (MINI / "same_prompt_twice.md").write_text("\n".join(md), encoding="utf-8", newline="")
    print(f"  same_prompt_twice: {len(out)} prompt")


def fig13(main_key, tok):
    gen = {r["id"]: r for r in read_jsonl(DATA / "D_gen.jsonl")}
    clean = {r["id"]: r for r in read_jsonl(DATA / "D_clean.jsonl")}
    rng = random.Random(SEED + 2)
    ids = [i for i, g in gen.items() if g["topic"] == "yaratici" and i in clean
           and 150 <= len(g["answer_ids"]) <= 250 and 150 <= len(clean[i]["answer_ids"]) <= 250
           and "```" not in g["answer"] and "```" not in clean[i]["answer"]]
    rng.shuffle(ids)
    pick = None
    for i in ids:
        sg, sc = _stat(gen[i], main_key), _stat(clean[i], main_key)
        if 6 <= sg["neglog10p"] <= 15 and sc["neglog10p"] < 1.0:    # eşiği geçen ama aşırı olmayan
            pick = (i, sg, sc)
            break
    if not pick:
        print("  fig13: uygun örnek bulunamadı")
        return
    i = pick[0]
    tok_rows, curve_rows = [], []
    for name, rec in (("D_gen", gen[i]), ("D_clean", clean[i])):
        ids_ = list(rec["answer_ids"])
        st = scored_tokens(ids_, len(rec["prompt_ids"]), main_key)
        sc = {p: s for p, s in st}
        for pos, piece in enumerate(token_pieces(tok, ids_)):
            tok_rows.append({"set": name, "record_id": i, "pos": pos, "piece": piece,
                             "scored": int(pos in sc), "score": sc.get(pos, "")})
        for n, nl in cum_curve([s for _, s in st]):
            curve_rows.append({"set": name, "record_id": i, "n": n, "neglog10p": nl})
    dump_csv(tok_rows, MINI / "fig13_tokens.csv",
             ["set", "record_id", "pos", "piece", "scored", "score"])
    dump_csv(curve_rows, MINI / "fig13_curve.csv", ["set", "record_id", "n", "neglog10p"])
    dump_json({"id": i, "prompt": gen[i]["prompt"], "D_gen": pick[1], "D_clean": pick[2]},
              MINI / "fig13_meta.json")
    print(f"  fig13: {i}  D_gen {pick[1]['neglog10p']:.1f} (N={pick[1]['N']})  "
          f"D_clean {pick[2]['neglog10p']:.2f}")


def example_1_1(main_key, tok):
    post = [r for r in read_jsonl(DATA / "D_post.jsonl") if not r["filtered"]]
    rng = random.Random(SEED + 3)
    pool = []
    for r in post:
        sc = len(r["source_answer"])
        lr = len(r["answer"]) / max(1, sc)
        if 200 <= sc <= 450 and 0.9 <= lr <= 1.15 and r["topic"] != "matematik_kod"                 and "```" not in r["answer"] and "```" not in r["source_answer"]:
            pool.append((r, lr))
    rng.shuffle(pool)                       # önce karıştır, sonra İLK uygun iki çifti al
    cands = []
    for r, lr in pool:                      # (en yüksek iz seçilmez — seçici olmayalım)
        st = _stat(r, main_key)
        if st["N"] >= 40 and 6 <= st["neglog10p"] <= 60:
            cands.append((r, st, lr))
        if len(cands) == 2:
            break
    sim_model = None
    try:
        from sentence_transformers import SentenceTransformer
        sim_model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
    except Exception as e:  # noqa: BLE001
        print(f"  (benzerlik modeli yüklenemedi: {type(e).__name__}; benzerlik boş bırakıldı)")
    rows, md = [], ["# ÖRNEK 1.1 — Önce ve sonra (E1 seçilen ayar: 4B, T=1.2)\n"]
    for n, (r, st, lr) in enumerate(cands, 1):
        sim = ""
        if sim_model is not None:
            e = sim_model.encode([r["source_answer"], r["answer"]], normalize_embeddings=True)
            sim = float(np.dot(e[0], e[1]))
        ids_ = list(r["answer_ids"])
        stt = scored_tokens(ids_, len(r["prompt_ids"]), main_key)
        scm = {p: s for p, s in stt}
        for pos, piece in enumerate(token_pieces(tok, ids_)):
            rows.append({"pair": n, "id": r["id"], "pos": pos, "piece": piece,
                         "scored": int(pos in scm), "score": scm.get(pos, "")})
        md.append(f"\n## Çift {n} · {r['id']}\n**Özgün:** {r['source_answer'].strip()}\n\n"
                  f"**Watermark'lı yeniden yazım:** {r['answer'].strip()}\n\n"
                  f"benzerlik = {sim if sim == '' else round(sim, 3)} · uzunluk oranı = {lr:.2f} · "
                  f"N = {st['N']} · −log₁₀(p) = {st['neglog10p']:.1f}\n")
    dump_csv(rows, MINI / "example_1_1.csv",
             ["pair", "id", "pos", "piece", "scored", "score"])
    (MINI / "example_1_1.md").write_text("\n".join(md), encoding="utf-8", newline="")
    print(f"  example_1_1: {len(cands)} çift")


def main():
    parse_args("mini denemeler (veriden)")
    log_cmd(sys.argv)
    t0 = time.time()
    MINI.mkdir(parents=True, exist_ok=True)
    main_key, wrong = _keys()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    guess_pairs(main_key, wrong, tok)
    same_prompt_twice(main_key)
    fig13(main_key, tok)
    example_1_1(main_key, tok)
    update_manifest(**{"mini.from_data": True})
    record_duration("mini_from_data", time.time() - t0)


if __name__ == "__main__":
    main()
