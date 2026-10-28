"""InstrucTurca veri hazırlığı — SPEC.md Bölüm 4.2.

Akışla (streaming) okuma; 4 GB dosya belleğe alınmaz. Filtreler, tohumlu
aday havuzu (20k), konu etiketleme (Qwen3.5-9B, watermark'sız), tabakalı
seçim (5.000 train + 500 eval_loss + 300 sweep), iç içe alt kümeler ve
saklanan token ID'leri (kural 2). DATA_CARD.md yazılır.

Çalıştırma:
  python -m data.prep_data --config configs/prep_data.yaml [--smoke]

YAML alanları: dataset, teacher_model, n_pool, n_train, n_eval_loss,
n_sweep, seed, filters{...}, quotas{...}
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from wm.util import (RESULTS, ROOT, dump_csv, dump_jsonl, log_cmd, load_config,
                     parse_args, read_jsonl, record_duration, update_manifest)

DATA_DIR = ROOT / "data"

TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜâîûÂÎÛ")
CONTEXT_PATTERNS = [
    r"yukarıdaki metne göre", r"aşağıdaki metne göre", r"yukarıdaki metin",
    r"aşağıdaki metin", r"yukarıdaki paragraf", r"önceki metin",
    r"yukarıdaki e-posta", r"yukarıdaki yorum",
]

REWRITE_P = re.compile(r"\[kod\]|```", re.IGNORECASE)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def turkish_ratio(text: str) -> float:
    """Türkçe'ye özgü karakter + yaygın Türkçe kelime oranı kaba ölçü."""
    words = re.findall(r"[A-Za-zÇçĞğİıÖöŞşÜü]+", text)
    if not words:
        return 0.0
    tk = sum(1 for w in words if any(c in TURKISH_CHARS for c in w.lower()))
    common = {"ve", "bir", "bu", "için", "ile", "olarak", "daha", "çok",
              "ama", "gibi", "kadar", "sonra", "önce", "ancak", "değil",
              "var", "yok", "olan", "ise", "her", "en"}
    cw = sum(1 for w in words if w.lower() in common)
    return (tk + cw) / len(words)


def is_near_duplicate(a: str, b: str) -> bool:
    na = re.sub(r"\s+", " ", a.lower()).strip()
    nb = re.sub(r"\s+", " ", b.lower()).strip()
    return na == nb or (len(na) > 80 and na[:80] == nb[:80])


def context_dependent(prompt: str, extra_input: str) -> bool:
    """Dış bağlam gerektiren promptlar (girdisi olmayan)."""
    if extra_input.strip():
        return False  # girdi alanı varsa bağlam örnek içinde
    return any(re.search(p, prompt, re.IGNORECASE) for p in CONTEXT_PATTERNS)


def mostly_code(answer: str) -> bool:
    """Cevabın büyük kısmı kod bloğu mu?"""
    fenced = len(REWRITE_P.findall(answer))
    lines = [l for l in answer.splitlines() if l.strip()]
    return fenced >= 2 or (lines and sum(
        1 for l in lines if l.startswith(("    ", "\t", "def ", "class ", "#"))
    ) > 0.6 * len(lines))


FIELD_CANDIDATES = {
    "instr": ("instruction", "talimat", "prompt", "question", "soru",
              "komut", "komutu"),
    "extra": ("input", "girdi", "giris", "context"),
    "ans": ("output", "cevap", "response", "answer", "yanit", "yanıt",
            "çıktı", "cikti"),
}


def resolve_fields(ex):
    """Alan adlarını esnek çöz (küçük harfe indirip aday adları dener)."""
    low = {}
    for k, v in ex.items():
        try:
            low[str(k).lower().strip()] = v
        except Exception:
            continue

    def pick(names):
        for n in names:
            v = low.get(n)
            if isinstance(v, str) and v.strip():
                return v
            # bazı varyantlar listeye sarılı olabilir
            if isinstance(v, list) and v and isinstance(v[0], str):
                return v[0]
        return ""

    instr = pick(FIELD_CANDIDATES["instr"])
    extra = pick(FIELD_CANDIDATES["extra"])
    ans = pick(FIELD_CANDIDATES["ans"])
    # InstrucTurca gibi bazı varyantlar instruction sütunu içermez; talimat
    # metni 'Input' sütunundadır → onu talimat olarak kullan.
    if not instr and extra:
        instr, extra = extra, ""
    return instr, extra, ans


def stream_pool(cfg, tokenizer):
    """InstrucTurca'yı akışla oku, filtreleri uygula, aday havuz döndür."""
    from datasets import load_dataset

    f = cfg["filters"]
    counts = Counter()
    pool = []
    seen_norm = set()
    code_kept = 0
    ds = load_dataset(cfg["dataset"], split="train", streaming=True)
    first_keys = None
    for i, ex in enumerate(ds):
        counts["total"] += 1
        if first_keys is None:
            first_keys = sorted(str(k) for k in ex.keys())
            # tanılama: gerçek alan adlarını diske yaz (her koşumda güncellenir)
            peek = {"keys": first_keys,
                    "sample": {k: (str(v)[:120] if isinstance(v, str)
                                  else str(v)[:60]) for k, v in ex.items()}}
            (ROOT / "data" / "field_peek.json").write_text(
                json.dumps(peek, ensure_ascii=False, indent=2),
                encoding="utf-8")
        # 1) alan doğrulama: talimat / girdi / cevap adlarını esnek çöz
        instr, extra, ans = resolve_fields(ex)
        if not instr or not ans:
            counts["missing_fields"] += 1
            continue
        prompt = (instr + "\n\n" + extra) if extra.strip() else instr
        # 2) filtreler
        n_prompt = count_tokens(tokenizer, prompt)
        n_ans = count_tokens(tokenizer, ans)
        if not (f["prompt_min"] <= n_prompt <= f["prompt_max"]):
            counts["f_prompt_len"] += 1
            continue
        if not (f["answer_min"] <= n_ans <= f["answer_max"]):
            counts["f_answer_len"] += 1
            continue
        if turkish_ratio(prompt) < f["min_turkish_ratio"] or \
           turkish_ratio(ans) < f["min_turkish_ratio"]:
            counts["f_turkish"] += 1
            continue
        if context_dependent(instr, extra):
            counts["f_context"] += 1
            continue
        if mostly_code(ans):
            counts["f_code_quota_hit"] += 1
            if code_kept >= cfg["quotas"]["max_code"]:
                continue
            code_kept += 1
        norm = re.sub(r"\s+", " ", (prompt + ans).lower()).strip()[:120]
        if norm in seen_norm:
            counts["f_duplicate"] += 1
            continue
        seen_norm.add(norm)
        counts["kept"] += 1
        pool.append({
            "src_id": i, "prompt": prompt, "answer": ans,
            "n_prompt_tokens": n_prompt, "n_answer_tokens": n_ans,
            "mostly_code": mostly_code(ans),
        })
        if len(pool) >= cfg["n_pool"]:
            break
        if cfg.get("max_stream_rows") and i >= cfg["max_stream_rows"]:
            break
    if not pool and first_keys is not None:
        sys.exit(
            f"HATA: havuz boş — {counts['total']} satır, {counts['missing_fields']}"
            f" 'missing_fields'.\n"
            f"Gerçek alan adları: {first_keys}\n"
            f"Ayrıntı: data/field_peek.json; data/prep_data.py içindeki "
            f"FIELD_CANDIDATES listesine bu adları ekleyip yeniden koşturun.")
    return pool, counts


TOPIC_INSTRUCTION = (
    "Aşağıdaki talimatı üç gruptan birine ayır. Yalnızca grup adını yaz: "
    "yaratici, olgusal, matematik_kod.\n\nTalimat:\n{prompt}")


def label_topics(pool, teacher_model, tokenizer, tp_size=1):
    """Havuz promptlarını 9B ile üç gruba etiketle (watermark'sız, greedy)."""
    from vllm import LLM, SamplingParams
    from gen.generate import close_llm, llm_memory_kwargs

    llm = LLM(model=teacher_model, tensor_parallel_size=tp_size, seed=7,
              dtype="bfloat16", **llm_memory_kwargs(0.9))
    sp = SamplingParams(temperature=0, max_tokens=8)
    prompts = []
    for r in pool:
        msgs = [{"role": "user",
                 "content": TOPIC_INSTRUCTION.format(prompt=r["prompt"][:2000])}]
        try:
            t = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False,
                tokenize=False)
        except TypeError:
            t = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
        prompts.append(t)
    outs = llm.generate(prompts, sp, use_tqdm=True)
    close_llm(llm)
    topics = []
    valid = {"yaratici", "olgusal", "matematik_kod"}
    for o in outs:
        txt = o.outputs[0].text.strip().lower()
        match = next((v for v in valid if v in txt), "olgusal")
        topics.append(match)
    return topics


def main():
    argv = parse_args("InstrucTurca veri hazırlığı")
    cfg = load_config(argv.config)
    log_cmd(sys.argv)
    t0 = __import__("time").time()

    # İDEMPOTENT: veri bir kez hazırlanır. Yeniden hazırlama bölmeleri DEĞİŞTİRİR
    # (9B etiketleme GPU'da bire bir belirlenimli değil) ve bunlara bağlı her şeyi
    # (anahtar kalibrasyonu, D_gen, eğitimler) geçersiz kılar.
    needed = [DATA_DIR / f for f in ("train.jsonl", "eval_loss.jsonl",
                                     "sweep.jsonl", "nested_subsets.json")]
    if not argv.smoke and not argv.force and all(p_.exists() for p_ in needed):
        print("veri hazırlığı zaten yapılmış — atlanıyor "
              "(yeniden üretmek bağımlı HER ŞEYİ geçersiz kılar; zorlamak için --force)")
        return

    if argv.smoke:
        cfg = dict(cfg)
        cfg["n_pool"] = min(cfg["n_pool"], 400)
        cfg["n_train"] = min(cfg["n_train"], 150)
        cfg["n_eval_loss"] = min(cfg["n_eval_loss"], 30)
        cfg["n_sweep"] = min(cfg["n_sweep"], 20)
        cfg.setdefault("max_stream_rows", 20000)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_model"])

    # ARŞİV: süzülmüş + etiketlenmiş havuz data/pool_labeled.jsonl'e yazılır. Yeniden
    # kurulum (--force) onu kullanır → ağ YOK, 9B etiketleme YOK, bölmeler bire bir
    # aynı. Kaynağı baştan çekmek (yeniden akış + yeniden etiketleme): --refetch.
    pool_path = DATA_DIR / "pool_labeled.jsonl"
    meta_path = DATA_DIR / "pool_meta.json"
    if pool_path.exists() and meta_path.exists() and not argv.smoke             and not argv.refetch:
        pool = list(read_jsonl(pool_path))
        counts = Counter(json.loads(meta_path.read_text(encoding="utf-8"))["counts"])
        print(f"arşivlenmiş etiketli havuz kullanılıyor: {len(pool)} örnek "
              f"(ağ ve etiketleme yok; baştan çekmek için --refetch)")
    else:
        pool, counts = stream_pool(cfg, tokenizer)
        print(f"havuz: {len(pool)} örnek; filtre sayımları: {dict(counts)}")
        if argv.smoke:
            topics = ["yaratici", "olgusal", "matematik_kod"]
            rng = random.Random(cfg["seed"])
            topics = [rng.choice(topics) for _ in pool]
        else:
            topics = label_topics(pool, cfg["teacher_model"], tokenizer,
                                  tp_size=cfg.get("tp_size", 1))
        for r, t in zip(pool, topics):
            r["topic"] = t
        if not argv.smoke:
            dump_jsonl(pool, pool_path)
            meta_path.write_text(json.dumps(
                {"counts": dict(counts), "dataset": cfg["dataset"],
                 "seed": cfg["seed"], "n_pool": len(pool)},
                ensure_ascii=False, indent=1), encoding="utf-8")

    # konu kontrol listesi (100 örnek)
    rng = random.Random(cfg["seed"] + 1)
    check_idx = rng.sample(range(len(pool)), min(100, len(pool)))
    dump_csv(
        [{"src_id": pool[i]["src_id"], "topic": pool[i]["topic"],
          "prompt_excerpt": pool[i]["prompt"][:200]} for i in check_idx],
        DATA_DIR / "label_check.csv")

    # tabakalı seçim
    rng = random.Random(cfg["seed"])
    by_topic = {}
    for i, r in enumerate(pool):
        by_topic.setdefault(r["topic"], []).append(i)
    target = {"yaratici": 0.40, "olgusal": 0.40, "matematik_kod": 0.20}

    def stratified(n_total, exclude: set):
        sel = []
        avail = {t: [i for i in idx if i not in exclude] for t, idx in by_topic.items()}
        for t, frac in target.items():
            n = min(int(n_total * frac), len(avail.get(t, [])))
            sel.extend(rng.sample(avail[t], n))
        # eksik kalanı diğerlerinden tamamla
        chosen = set(sel)
        rest = [i for idx in avail.values() for i in idx if i not in chosen]
        rng.shuffle(rest)
        sel.extend(rest[: n_total - len(sel)])
        return sel[:n_total]

    eval_idx = stratified(cfg["n_eval_loss"], set())
    train_idx = stratified(cfg["n_train"], set(eval_idx))
    sweep_idx = stratified(cfg["n_sweep"], set())

    def write_subset(idxs, name):
        recs = []
        for j, i in enumerate(idxs):
            r = pool[i]
            prompt_ids = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(r["answer"], add_special_tokens=False)["input_ids"]
            recs.append({
                "id": f"{name}-{j:05d}", "src_id": r["src_id"],
                "split": name, "topic": r["topic"],
                "prompt": r["prompt"], "prompt_ids": prompt_ids,
                "answer": r["answer"], "answer_ids": answer_ids,
                "n_prompt_tokens": len(prompt_ids),
                "n_answer_tokens": len(answer_ids),
            })
        return recs

    train_recs = write_subset(train_idx, "train")
    eval_recs = write_subset(eval_idx, "eval_loss")
    sweep_recs = write_subset(sweep_idx, "sweep")

    # iç içe alt kümeler S10 ⊂ S50 ⊂ S100
    rng2 = random.Random(cfg["seed"] + 2)
    by_topic_train = {}
    for r in train_recs:
        by_topic_train.setdefault(r["topic"], []).append(r["id"])
    nested = {}
    for frac, name in [(0.10, "S10"), (0.50, "S50")]:
        ids = []
        for t, lst in by_topic_train.items():
            k = max(1, int(len(lst) * frac))
            ids.extend(rng2.sample(lst, k))
        nested[name] = sorted(ids)
    with open(DATA_DIR / "nested_subsets.json", "w", encoding="utf-8") as f:
        json.dump({"S10": nested["S10"], "S50": nested["S50"],
                   "S100": [r["id"] for r in train_recs]}, f, indent=1)

    dump_jsonl(train_recs, DATA_DIR / "train.jsonl")
    dump_jsonl(eval_recs, DATA_DIR / "eval_loss.jsonl")
    dump_jsonl(sweep_recs, DATA_DIR / "sweep.jsonl")

    # DATA_CARD
    topic_dist = Counter(r["topic"] for r in train_recs)
    lens = sorted(r["n_answer_tokens"] for r in train_recs)
    total_tok = sum(r["n_answer_tokens"] + r["n_prompt_tokens"] for r in train_recs)
    card = f"""# DATA_CARD — InstrucTurca seçkisi

- Kaynak: `{cfg['dataset']}` (CC BY-SA 4.0; atıf zorunlu: Turkish NLP Suite,
  *InstrucTurca: Diverse Instructional Content Dataset for Turkish*, Hugging Face)
- Kaynak revizyon: akışla okundu; tohum {cfg['seed']}. Kod: `data/prep_data.py`
- Filtre sayımları: {json.dumps(dict(counts), ensure_ascii=False, indent=1)}
- Konu dağılımı (train): {dict(topic_dist)}
- Cevap uzunluğu (token): ort {sum(lens)/len(lens):.1f}, medyan {lens[len(lens)//2]}
- Toplam token (train, prompt+cevap): {total_tok:,}
- Bölmeler: train={len(train_recs)}, eval_loss={len(eval_recs)}, sweep={len(sweep_recs)}
- İç içe: S10={len(nested['S10'])}, S50={len(nested['S50'])}, S100={len(train_recs)}

Örnekler:
```json
{json.dumps(train_recs[0], ensure_ascii=False)[:600]}
```
```json
{json.dumps(train_recs[1], ensure_ascii=False)[:600]}
```
"""
    (DATA_DIR / "DATA_CARD.md").write_text(card, encoding="utf-8")

    update_manifest(
        **{"data.dataset": cfg["dataset"], "data.seed": cfg["seed"],
           "data.n_train": len(train_recs),
           "data.topic_dist": dict(topic_dist),
           "data.total_tokens": total_tok,
           "data.filter_counts": dict(counts)})
    record_duration("data_prep", __import__("time").time() - t0)
    print("veri hazırlığı tamam")


if __name__ == "__main__":
    main()