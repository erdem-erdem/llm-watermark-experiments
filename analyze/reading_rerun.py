"""Okuma modu yeniden koşumu — E3 (tüm A modelleri) ve E4 (düzeltilmiş halkalar).

NEDEN: ilk koşuda E3 eğrileri öğrencilerin SERBEST üretimlerinden (metin modu)
çıkarılmıştı; Bölüm 5 ise "okuma modu, watermark'lı verinin kendisiyle" der.
E4 halka 4 de yanlış veri üzerinde okunmuştu (B-greedy → a0 çıktıları). Burada:

  E3  Her A modeli, EĞİTİMDE GÖRDÜĞÜ watermark'lı cevaplar üzerinde (eğitimle
      aynı şablonlu prompt; cevaba bitiş tokenı eklenmez), okuma modunda:
        A-gen-ρ   → gördüğü D_gen cevapları      (test "seen")
        A-post-ρ  → gördüğü D_post cevapları     (test "seen")
        A0        → tüm D_gen ("gen") ve tüm D_post ("post")
        A-rew     → tüm D_post ("post")
      Ana anahtar + 5 yanlış anahtar; örnekler rastgele sırayla eklenerek 20
      log aralıklı noktada eğri; 50 parça p-değerleri (E0 histogramı için).
  E4  Halka 2: A-gen-100 / D_gen (E3 "seen" sonucu).  Halka 3: aracının
      watermark'sız çıktılarının doğrudan metin tespiti (greedy, örnekleme ve
      A₀ kontrolü).  Halka 4: B-greedy, B-sample, B-control — D_gen üzerinde okuma.

Puanlama spesifikasyona TAM uygun: (c) cevap başına, (d) test boyunca küresel;
E1 post-hoc cevaplarında parça başları atlanır (chunk_bounds).

KULLANIM (GPU başına bir süreç; modeller GPU'lara bölünür):
  python -m analyze.reading_rerun --stage model --only A0,A-gen-10,B-greedy
  python -m analyze.reading_rerun --stage text            # halka 3 (CPU)
  python -m analyze.reading_rerun --stage merge           # tüm parçaları birleştir
  veya hepsi:  bash scripts/rerun_reading.sh 4

Çıktılar:
  results/E3/reading/<model>__<test>.json      (+ .npz tahmin önbelleği preds/)
  results/E3/reading_curves.csv                (model,test,n,neglog10p,wrong_median)
  results/E3/reading_summary.csv               (model,test,N,neglog10p,wrong0..4,...)
  results/E0/pvals_parts_all.csv               (model,key,part,N,neglog10p,p)
  results/E4/chain.csv, fig_E4_1_chain.csv     (düzeltilmiş 4 halka)
"""

from __future__ import annotations

import json
import random
import sys
import time

import numpy as np

from wm.detect import detect_dataset
from wm.reading_mode import (collect_prompt_windows, compute_predictions,
                             load_predictions, neglog10p_of, score_pairs,
                             select_pairs)
from wm.scoring import log10_p
from wm.util import (RESULTS, ROOT, dump_csv, dump_json, load_config, log_cmd,
                     parse_args, read_csv, read_jsonl, record_duration,
                     update_manifest)

DATA = ROOT / "data"
E0, E3, E4 = (RESULTS / d for d in ("E0", "E3", "E4"))
READ = E3 / "reading"
N_PARTS = 50
CURVE_POINTS = 20
A_MODELS = ["A0", "A-gen-10", "A-gen-50", "A-gen-100",
            "A-post-10", "A-post-50", "A-post-100", "A-rew"]
B_MODELS = ["B-greedy", "B-sample", "B-control"]


def _keys():
    sel = json.loads((E0 / "selected_keys.json").read_text(encoding="utf-8"))
    return int(sel["main_key"]), [int(k) for k in sel["wrong_keys"]]


def model_dir(name):
    return RESULTS / ("E4" if name.startswith("B-") else "E3") / "models" / name


def tests_for(name):
    """[(test_etiketi, kaynak_dosya, 'seen'|'all')]"""
    if name == "A0":
        return [("gen", "D_gen.jsonl", "all"), ("post", "D_post.jsonl", "all")]
    if name.startswith("A-gen-"):
        return [("seen", "D_gen.jsonl", "seen")]
    if name.startswith("A-post-"):
        return [("seen", "D_post.jsonl", "seen")]
    if name == "A-rew":
        return [("post", "D_post.jsonl", "all")]
    if name.startswith("B-"):
        return [("gen", "D_gen.jsonl", "all")]
    sys.exit(f"bilinmeyen model: {name}")


def build_records(name, source_file, mode):
    """Eğitimdeki GÖRÜNÜMLE aynı kayıtlar: özgün soru + kaynak (watermark'lı) cevap."""
    from train.chat_format import load_template_tokenizer, to_chat_records
    source = {r["id"]: r for r in read_jsonl(DATA / source_file)}
    if mode == "seen":
        # tam olarak eğitim karışımı → watermark'lı payı (wm=True) seç
        from train.sft import build_training_set
        cfg = load_config(ROOT / "train" / "configs" / f"{name}.yaml")
        rows = [r for r in build_training_set(cfg) if r.get("wm")]
    else:
        rows = [{"id": r["id"], "prompt": r["prompt"], "topic": r.get("topic"),
                 "prompt_ids": r["prompt_ids"], "answer_ids": source[r["id"]]["answer_ids"]}
                for r in read_jsonl(DATA / "train.jsonl") if r["id"] in source]
    for r in rows:
        r["chunk_bounds"] = source[r["id"]].get("chunk_bounds")
    return to_chat_records(rows, load_template_tokenizer(), add_end=False)


def _skip_fn(records, k):
    heads = []
    for r in records:
        b = r.get("chunk_bounds")
        heads.append([int(s) for s, _e in b] if b else [])

    def skip(sidx, t):
        return any(s <= t < s + k for s in heads[sidx])
    return skip


def run_model(name, cfg, smoke=False):
    import torch
    from transformers import AutoModelForCausalLM
    main_key, wrong_keys = _keys()
    k = cfg.get("k", 2)
    READ.mkdir(parents=True, exist_ok=True)
    for test, src, mode in tests_for(name):
        out = READ / f"{name}__{test}.json"
        if out.exists() and not smoke:
            print(f"[{name}/{test}] var, atlanıyor")
            continue
        recs = build_records(name, src, mode)
        if smoke:
            recs = recs[:20]
        print(f"[{name}/{test}] {len(recs)} kayıt")
        npz = READ / "preds" / f"{name}__{test}.npz"
        if not npz.exists() or smoke:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir(name)), dtype=torch.bfloat16).cuda().eval()
            compute_predictions(model, recs, k=k, out_path=npz)
            del model
            torch.cuda.empty_cache()
        windows, preds, samples, kk = load_predictions(npz)
        overlap = collect_prompt_windows(recs, kk)
        skip = _skip_fn(recs, kk)
        ids = sorted({int(i) for i, _ in samples})
        rng = random.Random(cfg.get("seed", 20260814))
        order = ids[:]
        rng.shuffle(order)                       # örnekler rastgele sırayla eklenir
        W, P, owner = select_pairs(windows, preds, samples, kk, include=order,
                                   prompt_overlap_windows=overlap, skip_fn=skip)
        N = len(P)
        ns = sorted({int(round(x)) for x in np.logspace(np.log10(25), np.log10(max(N, 26)),
                                                        CURVE_POINTS)})
        ns = [n for n in ns if n <= N]

        def curve(sc):
            cs = np.cumsum(sc)
            return [log10_p(float(cs[n - 1]), n) for n in ns]

        sc_main = score_pairs(main_key, W, P)
        sc_wrong = [score_pairs(wk, W, P) for wk in wrong_keys]
        wcurves = np.asarray([curve(s) for s in sc_wrong])

        # 50 parça p-değeri (E0 histogramları): örnek sırası bölünür; her parça AYRI test
        parts_rows = []
        parts = [list(map(int, p_)) for p_ in np.array_split(np.asarray(order), N_PARTS)]
        for pi, part in enumerate(parts):
            Wp, Pp, _ = select_pairs(windows, preds, samples, kk, include=part,
                                     prompt_overlap_windows=overlap, skip_fn=skip)
            for label, key in [("main", main_key)] + [(f"wrong{i}", wk)
                                                       for i, wk in enumerate(wrong_keys)]:
                sc = score_pairs(key, Wp, Pp)
                nl = neglog10p_of(sc)
                parts_rows.append({"model": name, "test": test, "key": label,
                                   "part": pi, "N": len(sc), "neglog10p": nl,
                                   "p": float(10.0 ** (-nl)) if len(sc) else ""})
        dump_csv(parts_rows, READ / f"{name}__{test}_parts.csv",
                 ["model", "test", "key", "part", "N", "neglog10p", "p"])
        res = {"model": name, "test": test, "source": src, "mode": mode,
               "n_records": len(recs), "N": int(N),
               "neglog10p": neglog10p_of(sc_main),
               "wrong_neglog10p": [neglog10p_of(s) for s in sc_wrong],
               "curve": {"n": ns, "main": curve(sc_main),
                         "wrong_median": [float(x) for x in np.median(wcurves, axis=0)]},
               "mean_score_per_token": float(np.mean(sc_main)) if N else None}
        dump_json(res, out)
        print(f"  N={N}  -log10p(ana)={res['neglog10p']:.2f}  "
              f"yanlış={[round(x, 2) for x in res['wrong_neglog10p']]}")


def run_text(cfg, smoke=False):
    """Halka 3: aracının watermark'sız çıktıları — doğrudan metin tespiti (CPU)."""
    main_key, wrong_keys = _keys()
    out = []
    for tag, f in (("agent_greedy", "agent_greedy.jsonl"),
                   ("agent_sampled", "agent_sampled.jsonl"),
                   ("a0_greedy", "a0_greedy.jsonl")):
        p = DATA / f
        if not p.exists():
            continue
        recs = list(read_jsonl(p))[: (20 if smoke else None)]
        res, per = detect_dataset(recs, main_key)
        wr = [detect_dataset(recs, wk)[0].neglog10p for wk in wrong_keys]
        out.append({"tag": tag, "n_records": len(recs), "N": res.N,
                    "neglog10p": res.neglog10p,
                    "wrong_neglog10p": wr, "wrong_median": float(np.median(wr)),
                    "median_per_sample": float(np.median([x[1] for x in per
                                                          if np.isfinite(x[1])]))})
        print(f"[halka3/{tag}] N={res.N} -log10p={res.neglog10p:.2f} "
              f"yanlış medyan={np.median(wr):.2f}")
    dump_json(out, E4 / "ring3_text.json")


def merge(cfg):
    main_key, wrong_keys = _keys()
    # --- E3 özet + eğriler + yanlış anahtar parçaları
    curves, summ, parts_all = [], [], []
    for p in sorted(READ.glob("*__*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        row = {"model": r["model"], "test": r["test"], "n_records": r["n_records"],
               "N": r["N"], "neglog10p": r["neglog10p"],
               "mean_score_per_token": r["mean_score_per_token"]}
        for i, v in enumerate(r["wrong_neglog10p"]):
            row[f"wrong{i}"] = v
        row["wrong_mean_neglog10p"] = float(np.mean(r["wrong_neglog10p"]))
        row["wrong_max_neglog10p"] = float(np.max(r["wrong_neglog10p"]))
        summ.append(row)
        for n, v, w in zip(r["curve"]["n"], r["curve"]["main"], r["curve"]["wrong_median"]):
            curves.append({"model": r["model"], "test": r["test"], "n": n,
                           "neglog10p": v, "wrong_median": w})
        pp = READ / f"{r['model']}__{r['test']}_parts.csv"
        if pp.exists():
            parts_all += read_csv(pp)
    if summ:
        dump_csv(summ, E3 / "reading_summary.csv",
                 ["model", "test", "n_records", "N", "neglog10p", "mean_score_per_token",
                  "wrong0", "wrong1", "wrong2", "wrong3", "wrong4",
                  "wrong_mean_neglog10p", "wrong_max_neglog10p"])
        dump_csv(curves, E3 / "reading_curves.csv",
                 ["model", "test", "n", "neglog10p", "wrong_median"])
        dump_csv(parts_all, E0 / "pvals_parts_all.csv",
                 ["model", "test", "key", "part", "N", "neglog10p", "p"])

    # --- E4 zinciri (düzeltilmiş)
    by = {(r["model"], r["test"]): r for r in summ}
    rows = []
    old = {}
    cp = E4 / "chain.csv"
    if cp.exists():
        for r in read_csv(cp):
            old[(r["ring"], r["what"])] = r
    r1 = next((r for (ring, w), r in old.items() if ring == "1"), None)
    if r1:
        rows.append({"ring": 1, "what": "D_gen_metin_modu", "cond": "D_gen",
                     "key": "main", "neglog10p": r1["neglog10p"], "N": r1["N"],
                     "median_per_sample": r1.get("median_per_sample", ""),
                     "wrong_key_median": ""})

    def add_reading(ring, what, cond, model, test):
        r = by.get((model, test))
        if r:
            rows.append({"ring": ring, "what": what, "cond": cond, "key": "main",
                         "neglog10p": r["neglog10p"], "N": r["N"],
                         "median_per_sample": "",
                         "wrong_key_median": float(np.median(
                             [r[f"wrong{i}"] for i in range(5)]))})
    add_reading(2, "A-gen-100_okuma", "A-gen-100", "A-gen-100", "seen")
    t3 = E4 / "ring3_text.json"
    if t3.exists():
        for r in json.loads(t3.read_text(encoding="utf-8")):
            rows.append({"ring": 3, "what": f"{r['tag']}_metin_modu", "cond": r["tag"],
                         "key": "main", "neglog10p": r["neglog10p"], "N": r["N"],
                         "median_per_sample": r["median_per_sample"],
                         "wrong_key_median": r["wrong_median"]})
    add_reading(4, "B-greedy_okuma_Dgen", "B-greedy", "B-greedy", "gen")
    add_reading(4, "B-sample_okuma_Dgen", "B-sample", "B-sample", "gen")
    add_reading(4, "B-control_okuma_Dgen", "B-control", "B-control", "gen")
    # karşılaştırma: A₀'ın aynı veri üzerindeki sonucu
    add_reading(0, "A0_okuma_Dgen", "A0", "A0", "gen")
    fields = ["ring", "what", "cond", "key", "neglog10p", "N", "median_per_sample",
              "wrong_key_median"]
    if rows:
        if cp.exists() and not (E4 / "chain_v1_hatali.csv").exists():
            (E4 / "chain_v1_hatali.csv").write_bytes(cp.read_bytes())   # eskiyi sakla
        dump_csv(rows, cp, fields)
        dump_csv(rows, E4 / "fig_E4_1_chain.csv", fields)
        dump_csv(rows, E4 / "T_E4.csv", fields)
    update_manifest(**{"reading_rerun": {"models": sorted({r["model"] for r in summ}),
                                         "ring_rows": len(rows)}})
    print(f"birleştirildi: {len(summ)} model-test, {len(curves)} eğri noktası, "
          f"{len(rows)} E4 satırı")


def main():
    argv = parse_args("okuma modu yeniden koşumu")
    cfg = load_config(argv.config or "configs/detect.yaml")
    log_cmd(sys.argv)
    t0 = time.time()
    stage = (argv.stage or "model").lower()
    if stage == "model":
        names = (argv.only.split(",") if argv.only else A_MODELS + B_MODELS)
        bad = [n for n in names if n not in A_MODELS + B_MODELS]
        if bad:
            sys.exit(f"bilinmeyen model(ler): {bad}")
        for n in names:
            run_model(n, cfg, argv.smoke)
    elif stage == "text":
        run_text(cfg, argv.smoke)
    elif stage == "merge":
        merge(cfg)
    else:
        sys.exit("--stage model|text|merge")
    record_duration(f"reading_rerun_{stage}", time.time() - t0)


if __name__ == "__main__":
    main()
