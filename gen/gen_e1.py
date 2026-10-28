"""E1 — Post-hoc watermark: tarama ve tam veri (Bölüm 5-E1).

  python -m gen.gen_e1 --config configs/e1_sweep.yaml --stage sweep|full [--smoke]

sweep: 300 D_orig cevap × {0.8B,2B,4B,9B} × T∈{0.7,1.0,1.2}, watermark açık.
       Ölçümler: metin tespiti, anlam benzerliği (e5), uzunluk oranı, entropi.
full : yalnızca E1/selected.json onaydan SONRA; 5.000 yeniden yazım → D_post,
       aynı ayarla watermark'sız → D_rew. Elenenler işaretlenir, oranı raporlanır.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from gen.generate import (DEFAULT_TEACHER, REWRITE_INSTRUCTION,
                          close_llm, generate_batch, load_prompt_records,
                          make_llm)
from wm.detect import detect_text
from wm.util import (RESULTS, ROOT, dump_csv, dump_json, dump_jsonl, log_cmd,
                     load_config, parse_args, read_csv, read_jsonl, record_duration,
                     update_manifest)

DATA_DIR = ROOT / "data"
E1 = RESULTS / "E1"

SWEEP_MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B",
                "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]
SWEEP_T = [0.7, 1.0, 1.2]


def rewrite_instruction(number_safe: bool = False) -> str:
    instr = REWRITE_INSTRUCTION
    if number_safe:
        instr += " Sayıları ve matematiksel ifadeleri değiştirme."
    return instr


def detect_record(rec, key, k=2):
    r = detect_text(rec["prompt_ids"], rec["answer_ids"], key, k=k)
    return r.neglog10p, r.N


def similarity(source_text, rewritten, model, device="cuda"):
    """Çok dilli cümle gömmeleri arası kosinüs (kaynak ile çıktı)."""
    emb = model.encode([source_text, rewritten], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def length_ratio(src_text, out_text):
    return len(out_text) / max(len(src_text), 1)


def run_sweep(cfg, smoke=False, only_model=None, force=False):
    from transformers import AutoTokenizer

    key = _main_key()
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_model"])
    sweep = load_prompt_records(DATA_DIR / "sweep.jsonl",
                                limit=20 if smoke else None)
    # sweep kayıtları D_orig cevaplarını içerir (source_answer = answer)
    for r in sweep:
        r["source_answer"] = r["answer"]
    models = SWEEP_MODELS
    if only_model:
        models = [m for m in SWEEP_MODELS
                  if m == only_model or m.endswith(only_model)
                  or only_model in m]
        if not models:
            sys.exit(f"bilinmeyen model filtresi: {only_model} "
                     f"(seçenekler: {SWEEP_MODELS})")
    if only_model and not smoke and not force:
        part = E1 / "sweep_parts" / f"sweep_{models[0].replace('/', '_')}.csv"
        if part.exists():
            print(f"parça var, atlanıyor: {part} (--force ile yeniden üret)")
            return
    n_models = 1 if smoke else len(models)
    temps = [SWEEP_T[0]] if smoke else SWEEP_T
    rows = []
    for model in models[:n_models]:
        # Model BİR kez yüklenir, tüm T değerleri aynı örnekle üretilir.
        # Üretim bitince LLM kapatılır ve GPU belleği boşalır; benzerlik modeli
        # (e5) ANCAK ondan sonra yüklenir (küçük GPU'larda ikisi sığmaz).
        gen_by_T = {}
        llm = make_llm(model, tp_size=cfg.get("tp_size", 1), seed=cfg["seed"])
        try:
            for T in temps:
                gen_by_T[T] = generate_batch(
                    model, sweep, watermark=True, key=key, key_id="main",
                    temperature=T, top_p=cfg["top_p"],
                    max_new_tokens=cfg["max_new_tokens"],
                    seed=cfg["seed"], tp_size=cfg.get("tp_size", 1),
                    chunk_mode=True, tokenizer=tok,
                    extra_system=None, log_fn=print, llm=llm,
                    think_leak_tolerance=0.05)
        finally:
            close_llm(llm)
        sim_model = _load_sim_model(cfg["similarity_model"])
        for T, recs in gen_by_T.items():
            for r in recs:
                r["instruction"] = rewrite_instruction()
                neglog10p, N = detect_record(r, key)
                sim = similarity(r["source_answer"], r["answer"], sim_model)
                lr = length_ratio(r["source_answer"], r["answer"])
                rows.append({
                    "id": r["id"], "model": model, "T": T,
                    "N": N, "neglog10p": neglog10p,
                    "similarity": sim, "length_ratio": lr,
                    "filtered": (not (0.75 <= lr <= 1.25)) or r.get("think_leak", False),
                    "chunk_bounds": json.dumps(r["chunk_bounds"]),
                })
        del sim_model
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    fields = ["id", "model", "T", "N", "neglog10p", "similarity",
              "length_ratio", "filtered", "chunk_bounds"]
    if only_model:
        # GPU'ya bölünmüş koşum: model başına bir parça CSV; e1_report
        # tüm parçaları birleştirir.
        slug = models[0].replace("/", "_")
        dump_csv(rows, E1 / "sweep_parts" / f"sweep_{slug}.csv",
                 fieldnames=fields)
    else:
        dump_csv(rows, E1 / "sweep.csv", fieldnames=fields)
    # token başına entropi (model başına): üretim logprobs'ları ile pahalı;
    # E1 çıktılarının toplam skoru / N üzerinden pratik vekil yerine,
    # yeniden üretim yapmadan eğri için per-token skorlar saklanır
    print("sweep tamam — şimdi: python -m analyze.e1_report")


def _load_sim_model(name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cuda")


def run_full(cfg, smoke=False, force=False, shard=None):
    """Yalnızca yazar onayı sonrası (E1/selected.json zorunlu).

    shard=(i, n): kayıtların i::n dilimini üretir, results/E1/full_parts/ altına
    parça yazar (çoklu GPU). Birleştirme: --stage full_merge --nshards n.
    shard=None: tek süreçte tümü (eski davranış).
    """
    sel_path = E1 / "selected.json"
    if not sel_path.exists():
        sys.exit("Yazar onayı yok! E1/SWEEP_REPORT.md'yi inceleyip "
                 "E1/selected.json'a {\"model\":..., \"T\":..., \"approved\": true} "
                 "yazın (Bölüm 5-E1 adım 4).")
    if (not smoke and not force and (DATA_DIR / "D_post.jsonl").exists()
            and (DATA_DIR / "D_rew.jsonl").exists()):
        print("D_post.jsonl ve D_rew.jsonl var, atlanıyor (--force ile yeniden üret)")
        return
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    assert sel.get("approved"), "selected.json'da approved=true olmalı"
    key = _main_key()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_model"])
    train = load_prompt_records(DATA_DIR / "train.jsonl")
    for r in train:
        r["source_answer"] = r["answer"]
        r["instruction"] = rewrite_instruction()
    if smoke:
        train = train[:20]
    model, T = sel["model"], sel["T"]
    if shard:
        i, n = shard
        tag = f"{i}of{n}"
        paths = _part_paths(tag)
        if not force and _part_valid(paths, model, T):
            print(f"parça {tag} var, atlanıyor ({paths['post']})")
            return
        n_total = len(train)
        train = train[i::n]
        print(f"parça {tag}: {len(train)}/{n_total} kayıt")

    # Tek LLM örneği: D_post ve D_rew aynı yüklemeyle üretilir.
    # NOT: watermark'lı üretim greedy + anahtar belirlenimlidir; "başka tohumla
    # yeniden dene" aynı metni üretirdi (anlamsız ve her denemede model
    # yeniden yüklenirdi). Bu yüzden elenenler yeniden DENENMEZ: işaretlenir ve
    # oranı raporlanır (plan: Bölüm 5-E1.5 "elenenleri işaretle, oranı raporla").
    llm = make_llm(model, tp_size=cfg.get("tp_size", 1), seed=cfg["seed"])
    try:
        final_post = generate_batch(
            model, train, watermark=True, key=key, key_id="main",
            temperature=T, top_p=cfg["top_p"],
            max_new_tokens=cfg["max_new_tokens"], seed=cfg["seed"],
            tp_size=cfg.get("tp_size", 1), chunk_mode=True, tokenizer=tok,
            llm=llm)
        recs_rew = generate_batch(
            model, train, watermark=False, key=None, key_id=None,
            temperature=T, top_p=cfg["top_p"],
            max_new_tokens=cfg["max_new_tokens"], seed=cfg["seed"] + 7,
            tp_size=cfg.get("tp_size", 1), chunk_mode=True, tokenizer=tok,
            llm=llm)
    finally:
        close_llm(llm)
    for r in final_post:
        lr = length_ratio(r["source_answer"], r["answer"])
        r["filtered"] = not (0.75 <= lr <= 1.25)
    n_filtered = sum(r["filtered"] for r in final_post)
    print(f"D_post: {len(final_post)} satır, filtrelenen: {n_filtered} "
          f"(%{100 * n_filtered / len(final_post):.1f})")

    for r in recs_rew:
        lr = length_ratio(r["source_answer"], r["answer"])
        r["filtered"] = not (0.75 <= lr <= 1.25)

    if shard:
        dump_jsonl(final_post, paths["post"])
        dump_jsonl(recs_rew, paths["rew"])
        dump_json({"i": i, "n": n, "n_records": len(train), "n_total": n_total,
                   "model": model, "T": T}, paths["meta"])   # meta EN SON: tamamlanma işareti
        print(f"parça {tag} tamam: {len(final_post)} D_post, {len(recs_rew)} D_rew")
        return
    _write_full_outputs(final_post, recs_rew, sel, model, T, cfg)


def _part_paths(tag):
    d = E1 / "full_parts"
    return {"post": d / f"post_{tag}.jsonl", "rew": d / f"rew_{tag}.jsonl",
            "meta": d / f"meta_{tag}.json"}


def _part_valid(paths, model, T) -> bool:
    """Parça tamam mı ve AYNI seçimle (model, T) mi üretildi?"""
    if not all(p.exists() for p in paths.values()):
        return False
    m = json.loads(paths["meta"].read_text(encoding="utf-8"))
    return m.get("model") == model and float(m.get("T")) == float(T)


def merge_full(cfg, nshards, smoke=False, force=False):
    """Parçaları train.jsonl sırasıyla birleştir → D_post.jsonl, D_rew.jsonl."""
    if (not smoke and not force and (DATA_DIR / "D_post.jsonl").exists()
            and (DATA_DIR / "D_rew.jsonl").exists()):
        print("D_post.jsonl ve D_rew.jsonl var, birleştirme atlanıyor")
        return
    sel = json.loads((E1 / "selected.json").read_text(encoding="utf-8"))
    model, T = sel["model"], sel["T"]
    missing = [i for i in range(nshards)
               if not _part_valid(_part_paths(f"{i}of{nshards}"), model, T)]
    if missing:
        sys.exit(f"HATA: eksik/uyumsuz E1 full parçaları: {missing} (n={nshards}) — "
                 f"o parçaları yeniden koşun: --stage full --shard i/{nshards}")
    order = {r["id"]: k for k, r in enumerate(read_jsonl(DATA_DIR / "train.jsonl"))}
    post, rew = [], []
    for i in range(nshards):
        p = _part_paths(f"{i}of{nshards}")
        post.extend(read_jsonl(p["post"]))
        rew.extend(read_jsonl(p["rew"]))
    post.sort(key=lambda r: order[r["id"]])
    rew.sort(key=lambda r: order[r["id"]])
    for name, recs in (("D_post", post), ("D_rew", rew)):
        ids = [r["id"] for r in recs]
        if len(ids) != len(set(ids)):
            sys.exit(f"HATA: {name} birleşiminde yinelenen id var")
        expect = 20 if smoke else len(order)
        if len(ids) != expect:
            sys.exit(f"HATA: {name} birleşimi {len(ids)} kayıt, beklenen {expect}")
    print(f"birleştirildi: {len(post)} D_post, {len(rew)} D_rew ({nshards} parça)")
    _write_full_outputs(post, rew, sel, model, T, cfg)


def _write_full_outputs(final_post, recs_rew, sel, model, T, cfg):
    n_filtered = sum(bool(r["filtered"]) for r in final_post)
    retried = 0
    out = {"model": model, "T": T, "seed": cfg["seed"],
           "n": len(final_post), "n_filtered": n_filtered,
           "n_retried": retried,
           "filter_rate": n_filtered / max(len(final_post), 1)}
    dump_json(out, E1 / "full_stats.json")
    dump_jsonl(final_post, DATA_DIR / "D_post.jsonl")
    dump_jsonl(recs_rew, DATA_DIR / "D_rew.jsonl")
    update_manifest(**{"e1.selected": sel, "e1.n_filtered": n_filtered})
    print("E1 tam veri tamam: D_post.jsonl, D_rew.jsonl")


def _main_key():
    import json
    p = RESULTS / "E0" / "selected_keys.json"
    if not p.exists():
        sys.exit("Anahtar seçimi yapılmamış (data/prep_keys.py çalıştırın).")
    return json.loads(p.read_text(encoding="utf-8"))["main_key"]


def main():
    argv = parse_args("E1 post-hoc watermark")
    ap_extra = sys.argv[1:]
    stage = "sweep"
    only_model = None
    for i, a in enumerate(ap_extra):
        if a == "--stage":
            stage = ap_extra[i + 1]
        elif a == "--model":
            only_model = ap_extra[i + 1]
    cfg = load_config(argv.config)
    log_cmd(sys.argv)
    import time
    t0 = time.time()
    if stage == "sweep":
        run_sweep(cfg, smoke=argv.smoke, only_model=only_model,
                  force=argv.force)
    elif stage == "full":
        shard = None
        if argv.shard:
            i, n = (int(x) for x in argv.shard.split("/"))
            assert 0 <= i < n, f"--shard i/n: 0 <= i < n olmalı ({argv.shard})"
            shard = (i, n)
        run_full(cfg, smoke=argv.smoke, force=argv.force, shard=shard)
    elif stage == "full_merge":
        if not argv.nshards:
            sys.exit("--stage full_merge için --nshards N gerekli")
        merge_full(cfg, argv.nshards, smoke=argv.smoke, force=argv.force)
    else:
        sys.exit(f"bilinmeyen stage: {stage}")
    record_duration(f"e1_{stage}", time.time() - t0)


if __name__ == "__main__":
    main()