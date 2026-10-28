"""Tüm tespit (detection) koşularının orkestrasyonu — E0/E2/E3/E4.

Ürettiği dosyalar:
  E0/pvals_a0.csv          — A₀ temiz model, ana anahtar, D_orig cevapları
                             (50 parça p-değer histogramı için)
  E0/wrong_keys.csv        — A modelleri x 5 yanlış anahtar (okuma modu)
  E0/dedup_demo.csv        — tekilleştirme kapalı/açık karşılaştırması
  E2/curves.csv            — D_gen ön-ek eğrileri (n ∈ {25..400}, konu bazlı)
  E2/detection.csv         — D_gen / D_clean örnekleme düzeyi -log10p
  E2/token_scores.csv      — token başına skor dağılımları (D_gen vs D_clean)
  E3/detect_curves.csv     — 8 eğitim koşulu x 20 nokta eğri
  E3/reading_mode.csv      — okuma modu (eğitim verisi tekrarına karşı)
  E4/chain.csv             — 4 halka: D_gen metin, A-gen-100 okuma,
                             ajan çıktı metin, B okuma; ana anahtar + yanlışlar

  python -m analyze.detect_all --config configs/detect.yaml [--smoke]
                            [--stage E0,E2,E3,E4] [--only A-gen-10,A-post-50]
  --only: E3 koşu filtresi — GPU başına koşumlarda parça dosyaları
          (results/E3/detect_curves_parts/) yazar; her koşum TÜM parçaları
          birleştirip detect_curves.csv'yi tazeler.
  ölçek sınırı (configs/detect.yaml): e3_detect_n_prompts (0/null = tüm
  D_gen; önbellek adı prompt sayısını içerir, sınır değişince yeniden
  üretilir).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from wm.detect import detect_dataset, detect_prefix_curves
from wm.reading_mode import (compute_predictions, collect_prompt_windows,
                             load_predictions, rescore_predictions)
from wm.util import (RESULTS, ROOT, dump_csv, dump_json, log_cmd, load_config,
                     parse_args, read_csv, read_jsonl, record_duration, update_manifest)

DATA = ROOT / "data"
E0, E2, E3, E4 = (RESULTS / d for d in ("E0", "E2", "E3", "E4"))


def _keys():
    import json
    sel = json.loads((E0 / "selected_keys.json").read_text(encoding="utf-8"))
    return sel["main_key"], sel["wrong_keys"]


def _topics(records):
    """Konu etiketleri; eksikse 'bilinmiyor'."""
    return [r.get("topic") or "bilinmiyor" for r in records]


# ---------------- E0 ----------------

def run_e0(cfg, smoke):
    main_key, wrong_keys = _keys()
    train = list(read_jsonl(DATA / "train.jsonl"))
    if smoke:
        train = train[:50]
    # A₀ eğitimdeki biçimle okunur: şablonlu prompt (cevaba bitiş token'ı EKLENMEZ —
    # yalnız cevap token'ları puanlanır)
    from train.chat_format import load_template_tokenizer, to_chat_records
    train = to_chat_records(train, load_template_tokenizer(), add_end=False)
    preds_dir = E0 / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    npz = preds_dir / "a0.npz"
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        cfg["a0_model"], dtype=torch.bfloat16).cuda().eval()
    if not npz.exists():
        compute_predictions(model, train, k=cfg.get("k", 2),
                            out_path=npz)
    del model
    import torch as _t
    _t.cuda.empty_cache()
    windows, preds, samples, k = load_predictions(npz)

    # 1) A₀ + ana anahtar: parça başına p-değerleri (kalibrasyon)
    pvals = []
    n_parts = min(50, max(1, len(set(int(i) for i in samples[:, 0])))) \
        if len(samples) else 1
    by_sample = {}
    for i, (idx, _pos) in enumerate(samples):
        by_sample.setdefault(int(idx), []).append(i)
    idxs = sorted(by_sample)
    parts = np.array_split(np.asarray(idxs), n_parts)
    for part in parts:
        sub = [i for idx in part for i in by_sample[int(idx)]]
        r = rescore_predictions([windows[i] for i in sub], preds[sub], main_key, k)
        pvals.append({"part": len(pvals), "N": r["N"],
                      "neglog10p": r["neglog10p"]})
    dump_csv(pvals, E0 / "pvals_a0.csv", fieldnames=["part", "N", "neglog10p"])

    # 2) yanlış anahtarlar: A₀ okuma modu (kalibrasyon; düşük -log10p beklenir)
    rows = []
    for wi, wk in enumerate(wrong_keys):
        r = rescore_predictions(windows, preds, wk, k,
                                prompt_overlap_windows=collect_prompt_windows(train, k))
        rows.append({"key": f"wrong{wi}", "mode": "reading", "model": cfg["a0_model"],
                     "neglog10p": r["neglog10p"], "N": r["N"]})
    r = rescore_predictions(windows, preds, main_key, k,
                            prompt_overlap_windows=collect_prompt_windows(train, k))
    rows.append({"key": "main", "mode": "reading", "model": cfg["a0_model"],
                 "neglog10p": r["neglog10p"], "N": r["N"]})

    # 3) tekilleştirme gösterimi: metin modu, kural (d) kapalı/açık
    gen = list(read_jsonl(DATA / "D_gen.jsonl"))
    if smoke:
        gen = gen[:20]
    res_full, _ = detect_dataset(gen, main_key)
    # kural (d) kapalı: kayıt bazında ayrı ayrı (küresel küme yok)
    per = []
    from wm.detect import detect_text
    for rec in gen:
        d = detect_text(rec["prompt_ids"], rec["answer_ids"], main_key)
        per.append(d.neglog10p)
    demo = {"dedup_on_total_neglog10p": res_full.neglog10p,
            "dedup_on_N": res_full.N,
            "dedup_off_mean_neglog10p": float(np.mean(per)),
            "dedup_off_median_neglog10p": float(np.median(per)),
            "dedup_off_n": len(per),
            "note": "kural (d) kapalı olduğunda tekrar eden pencereler skoru "
                    "şişirir — Blogda gösterim için."}
    dump_json(demo, E0 / "dedup_demo.json")
    for r_ in rows:
        r_["a0_model"] = cfg["a0_model"]
    dump_csv(rows, E0 / "wrong_keys.csv",
             fieldnames=["key", "mode", "model", "neglog10p", "N", "a0_model"])
    update_manifest(**{"e0.detection": {"n_parts": len(pvals)}})


# ---------------- E2 ----------------

def run_e2(cfg, smoke):
    main_key, wrong_keys = _keys()
    d_gen = list(read_jsonl(DATA / "D_gen.jsonl"))
    d_clean = list(read_jsonl(DATA / "D_clean.jsonl"))
    if smoke:
        d_gen, d_clean = d_gen[:50], d_clean[:50]

    # örnek düzeyi tespit (küresel kural d)
    res_gen, per_gen = detect_dataset(d_gen, main_key)
    res_clean, per_clean = detect_dataset(d_clean, main_key)
    rows = [{"record_id": r.get("id", i), "set": "D_gen", "topic": t,
             "N": n, "neglog10p": nl}
            for i, (r, (n, nl), t) in enumerate(zip(d_gen, per_gen, _topics(d_gen)))]
    rows += [{"record_id": r.get("id", i), "set": "D_clean", "topic": t,
              "N": n, "neglog10p": nl}
             for i, (r, (n, nl), t) in enumerate(zip(d_clean, per_clean,
                                                     _topics(d_clean)))]
    dump_csv(rows, E2 / "detection.csv", fieldnames=[
        "record_id", "set", "topic", "N", "neglog10p"])

    # n=400 eşik oranları (Blog eşiği -log10p = 6)
    thr = cfg.get("threshold_neglog10p", 6.0)
    gen4 = [r for r in rows if r["set"] == "D_gen" and r["N"] >= 400]
    cl4 = [r for r in rows if r["set"] == "D_clean" and r["N"] >= 400]
    rates = {
        "D_gen_detection_rate_at_n400":
            float(np.mean([r["neglog10p"] >= thr for r in gen4])) if gen4 else None,
        "D_gen_n_at_n400": len(gen4),
        "D_clean_false_alarm_rate_at_n400":
            float(np.mean([r["neglog10p"] >= thr for r in cl4])) if cl4 else None,
        "D_clean_n_at_n400": len(cl4),
        "threshold_neglog10p": thr,
    }
    dump_json(rates, E2 / "rates.json")

    # konu bazlı medyanlar
    topics = sorted({r["topic"] for r in rows})
    med = [{"topic": t, "set": s,
            "median_neglog10p": float(np.median([r["neglog10p"] for r in rows
                                                 if r["topic"] == t and r["set"] == s]))}
           for t in topics for s in ("D_gen", "D_clean")]
    dump_csv(med, E2 / "medians_by_topic.csv",
             fieldnames=["topic", "set", "median_neglog10p"])

    # ön-ek eğrileri (her D_gen kaydı; ns = {25,50,100,200,300,400})
    curve_rows = []
    for r in d_gen:
        cs = detect_prefix_curves(r["answer_ids"], len(r["prompt_ids"]),
                                  main_key, k=cfg.get("k", 2),
                                  ns=[25, 50, 100, 200, 300, 400])
        for c in cs:
            c["record_id"] = r.get("id")
            c["topic"] = r.get("topic")
            curve_rows.append(c)
    dump_csv(curve_rows, E2 / "curves.csv",
             fieldnames=["record_id", "topic", "n", "N", "neglog10p"])

    # token başına skorlar (dağılım figürü için)
    tok_rows = []
    from wm.sampler import text_detection_scores
    from wm.prf import R
    from wm.scoring import per_token_score
    for rec, st in ((d_gen[0], "D_gen"), (d_clean[0], "D_clean")):
        for pos, window in text_detection_scores(rec["answer_ids"],
                                                 len(rec["prompt_ids"]),
                                                 main_key, cfg.get("k", 2)):
            tok_rows.append({"set": st, "record_id": rec.get("id"),
                             "pos": pos,
                             "score": per_token_score(R(main_key, window,
                                                        int(rec["answer_ids"][pos])))})
    dump_csv(tok_rows, E2 / "token_scores.csv",
             fieldnames=["set", "record_id", "pos", "score"])

    # yanlış anahtar ortalama (okuma modu hızlı: metin modu örneklemesi)
    wrong_means = {}
    for wi, wk in enumerate(wrong_keys):
        _, per = detect_dataset(d_gen[:20] if not smoke else d_gen, wk)
        wrong_means[f"wrong{wi}"] = float(np.median([p[1] for p in per]))
    dump_json(wrong_means, E2 / "wrong_key_medians.json")
    update_manifest(**{"e2.detection": {
        "D_gen_total_neglog10p": res_gen.neglog10p,
        "D_clean_total_neglog10p": res_clean.neglog10p,
        **rates}})


# ---------------- E3 ----------------

def run_e3(cfg, smoke, only=None):
    main_key, wrong_keys = _keys()
    gen_rows = list(read_jsonl(DATA / "D_gen.jsonl"))
    # ölçek sınırı: her koşu için üretilecek prompt sayısı (maliyet: 8 koşu
    # × n × 512 token). 0/null = tüm D_gen (5000).
    n_cap = cfg.get("e3_detect_n_prompts", 500)
    if n_cap:
        gen_rows = gen_rows[:n_cap]
    if smoke:
        gen_rows = gen_rows[:30]

    conds = list(cfg["a_models"])
    if only:
        conds = [c for c in conds if c in only]
        if not conds:
            import sys as _s
            _s.exit(f"bilinmeyen koşu(lar): {sorted(only)} "
                    f"(seçenekler: {cfg['a_models']})")

    # 20 log-aralıklı nokta eğrileri (koşu başına toplam küme üzerinden)
    ns = sorted({int(round(n)) for n in np.logspace(np.log10(25), np.log10(4000), 20)})
    parts = E3 / "detect_curves_parts"
    for cond in conds:
        path = f"results/E3/models/{cond}" if not cond.startswith("Qwen") \
            else cond
        gen = gen_cond_outputs(path, gen_rows, cfg, smoke)
        # koşu bazında: her kaydı koşuyla puanla (küresel kural d koşu içi)
        res, per = detect_dataset(gen, main_key)
        # eğri: koşu toplam skor dizisinden kümülatif
        scores_all = list(res.scores)
        rows = [{"model": cond, "n": n, "neglog10p": np_neglog10p(scores_all[:n])}
                for n in ns if n <= len(scores_all)]
        # koşu parçası: GPU başına koşumlarda birleşirme için
        dump_csv(rows, parts / f"{cond}.csv",
                 fieldnames=["model", "n", "neglog10p"])

    # ---- birleştirme: TÜM parçaları oku (paralel koşumlardan) ----
    merged = []
    if parts.is_dir():
        for pf in sorted(parts.glob("*.csv")):
            merged.extend(read_csv(pf))
    if merged:
        dump_csv(merged, E3 / "detect_curves.csv",
                 fieldnames=["model", "n", "neglog10p"])
        update_manifest(**{"e3.detect_curves": len(merged),
                           "e3.detect_n_prompts": len(gen_rows)})


def gen_cond_outputs(model_path, prompts, cfg, smoke):
    """Bir E3 koşusunun çıktılarını üret (ya da önbellekten oku).

    Önbellek adı prompt sayısını içerir — ölçek sınırı değişirse yeniden
    üretilir.
    """
    cache = RESULTS / "E3" / "gen_outputs"
    cache.mkdir(parents=True, exist_ok=True)
    name = Path(model_path).name
    # "_nowm": önbellek sürümü — eski (yanlışlıkla watermark'lı üretilmiş)
    # önbellek dosyaları ASLA yeniden kullanılmaz.
    out = cache / f"{name}_n{len(prompts)}_nowm.jsonl"
    if out.exists():
        return list(read_jsonl(out))
    from gen.generate import generate_batch
    recs = [{"id": r["id"], "prompt": r["prompt"]} for r in prompts]
    # RADYOAKTİVİTE TESTİ: öğrenci watermark'ı ÖĞRENDİ mi? Öğrenci (A₀ dahil HER
    # koşul) watermark örnekleyicisi OLMADAN, standart örneklemeyle üretir; tespit
    # yalnızca öğrenilmiş sinyali ölçer. (watermark=True olsaydı sinyal
    # örnekleyiciden gelirdi ve E3 anlamsızlaşırdı.)
    batch = generate_batch(
        model_path, recs, watermark=False, key=None, key_id=None,
        temperature=cfg.get("T", 0.8), top_p=cfg.get("top_p", 0.95),
        max_new_tokens=cfg.get("max_new_tokens", 512),
        seed=cfg.get("seed", 20260814), tp_size=cfg.get("tp_size", 1))
    from wm.util import dump_jsonl
    dump_jsonl(batch, out)
    return batch


def np_neglog10p(scores):
    from wm.scoring import log10_p
    s = float(np.sum(scores))
    return log10_p(s, len(scores))


# ---------------- E4 ----------------

def run_e4(cfg, smoke):
    main_key, wrong_keys = _keys()
    k = cfg.get("k", 2)
    d_gen = list(read_jsonl(DATA / "D_gen.jsonl"))
    agent = list(read_jsonl(DATA / "agent_greedy.jsonl"))
    a0 = list(read_jsonl(DATA / "a0_greedy.jsonl"))
    b_sample = list(read_jsonl(DATA / "agent_sampled.jsonl"))
    if smoke:
        d_gen, agent, a0, b_sample = (d_gen[:20], agent[:20], a0[:20],
                                      b_sample[:20])

    import torch
    from transformers import AutoModelForCausalLM
    preds_dir = E4 / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Halka 1: D_gen metin modu (temiz ajan çıktısında tespit)
    res1, per1 = detect_dataset(d_gen, main_key)
    rows.append({"ring": 1, "what": "D_gen metin modu (ajana sızan veri)",
                 "key": "main", "neglog10p": res1.neglog10p, "N": res1.N,
                 "median_per_sample": float(np.median([p[1] for p in per1]))})

    # Halka 2: A-gen-100 okuma modu (eğitilmiş model gerçek metni okuyor)
    def reading(model_path, records, tag):
        npz = preds_dir / f"{tag}.npz"
        if npz.exists():
            windows, preds, samples, kk = load_predictions(npz)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16).cuda().eval()
            compute_predictions(model, records, k=k, out_path=npz)
            del model
            torch.cuda.empty_cache()
            windows, preds, samples, kk = load_predictions(npz)
        res = rescore_predictions(windows, preds, main_key, kk,
                                  prompt_overlap_windows=collect_prompt_windows(records, kk))
        row = {"ring": None, "what": tag, "key": "main",
               "neglog10p": res["neglog10p"], "N": res["N"]}
        wrongs = []
        for wi, wk in enumerate(wrong_keys):
            r = rescore_predictions(windows, preds, wk, kk,
                                    prompt_overlap_windows=collect_prompt_windows(records, kk))
            wrongs.append(r["neglog10p"])
        row["wrong_key_median"] = float(np.median(wrongs)) if wrongs else None
        return row

    rows.append(reading(cfg["agent_checkpoint"], d_gen, "ring2_A-gen-100_okuma"))

    # Halka 3: ajan çıktılarının doğrudan metin modu tespiti
    res3, per3 = detect_dataset(agent, main_key)
    rows.append({"ring": 3, "what": "ajan çıktısı metin modu",
                 "key": "main", "neglog10p": res3.neglog10p, "N": res3.N,
                 "median_per_sample": float(np.median([p[1] for p in per3]))})

    # Halka 4: B-greedy okuma modu (öğrenci model — radioactivity)
    rows.append(reading(f"results/E4/models/B-greedy", a0, "ring4_B-greedy_okuma"))
    if b_sample:
        rows.append(reading(f"results/E4/models/B-sample", b_sample,
                            "ring4b_B-sample_okuma"))

    for i, r in enumerate(rows):
        if r["ring"] is None:
            r["ring"] = 2 if "ring2" in r["what"] else 4
        rows[i] = r

    dump_csv(rows, E4 / "chain.csv",
             fieldnames=["ring", "what", "key", "neglog10p", "N",
                         "median_per_sample", "wrong_key_median"])
    update_manifest(**{"e4.chain": rows})
    print(f"E4 zinciri: {len(rows)} halka tamam")


def main():
    argv = parse_args("tespit orkestrasyonu")
    cfg = load_config(argv.config)
    log_cmd(sys.argv)
    t0 = time.time()
    # argparse (wm.util.parse_args) hem "--stage E0" hem "--stage=E0" biçimini
    # okur. (Eski elle ayrıştırma yalnız "=" biçimini tanıyordu; boşluklu
    # biçimde stage BOŞ kalıp TÜM aşamalar koşuyordu.)
    ALL = ["E0", "E2", "E3", "E4"]
    stages = ([s.strip().upper() for s in argv.stage.split(",") if s.strip()]
              if argv.stage else ALL)
    bad = [s for s in stages if s not in ALL]
    if bad:
        sys.exit(f"bilinmeyen stage: {bad} (seçenekler: {','.join(ALL)})")
    print(f"tespit aşamaları: {','.join(stages)}")
    if "E0" in stages:
        run_e0(cfg, argv.smoke)
    if "E2" in stages:
        run_e2(cfg, argv.smoke)
    only = None
    for i, a in enumerate(sys.argv):
        if a == "--only":
            only = set(sys.argv[i + 1].split(","))
    if "E3" in stages:
        run_e3(cfg, argv.smoke, only=only)
    if "E4" in stages:
        run_e4(cfg, argv.smoke)
    record_duration("detect_all", time.time() - t0)
    print("tespit orkestrasyonu tamam")


if __name__ == "__main__":
    main()