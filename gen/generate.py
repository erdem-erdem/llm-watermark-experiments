"""Ortak üretim motoru — vLLM offline batch, özel Gumbel-max logits processor.

Watermark'lı üretim: SamplingParams.logits_processors ile bizim işlemcimiz,
temperature=0 (greedy) — T ve top-p işlemcinin içinde uygulanır.
Watermark'sız üretim: vLLM standart örnekleyici, aynı T/top-p, tohumlu.

Kural 2: token ID'ler saklanır, yeniden tokenize edilmez.
Kural 6: thinking kapalı; çıktıda düşünme bloğu görülürse dur (wm.util.check_no_thinking).
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

from wm.util import (ROOT, check_no_thinking, dump_jsonl, hf_commit_hash,
                     log_note, read_jsonl, update_manifest)

# Kural 6 güvenlik ağı: thinking token'ları üretimde zaten YASAK (wm_ban). Yine de
# sızan olursa: oran bu eşiğin altındaysa koşu SÜRER (kayıt think_leak=True ile
# işaretlenir ve manifest'e sayılır); üstündeyse şablon bozuk demektir → DUR.
# Ortam değişkeniyle değiştirilebilir: WM_THINK_LEAK_TOLERANCE=0.10
THINK_LEAK_TOLERANCE = float(os.environ.get("WM_THINK_LEAK_TOLERANCE", "0.05"))

DEFAULT_TEACHER = "Qwen/Qwen3.5-9B"
DEFAULT_STUDENT = "Qwen/Qwen3.5-2B-Base"


def build_prompts(tokenizer, prompt_texts: list[str], system_text: str | None = None):
    """Sohbet şablonunu uygula (thinking kapalı). Dönüş: (şablonlu_metin, prompt_ids).

    Şablon post-trained modelin şablonudur (Bölüm 4.1); Base eğitim ve tespit
    aynı şablonu kullanır.
    """
    out = []
    for text in prompt_texts:
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": text})
        templ = render_chat(tokenizer, messages)
        # Metni kendimiz tokenize ederiz: apply_chat_template(tokenize=True)
        # dönüş biçimi transformers sürümleri arasında değişiyor (düz liste /
        # BatchEncoding / iç içe liste); metin→id her sürümde aynıdır.
        ids = tokenizer(templ, add_special_tokens=False)["input_ids"]
        out.append((templ, [int(t) for t in ids]))
    return out


def render_chat(tokenizer, messages) -> str:
    """Sohbet şablonunu METİN olarak uygula (thinking kapalı)."""
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
            tokenize=False)
    except TypeError:
        # şablon enable_thinking kabul etmiyorsa
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)


def chunk_source_text(tokenizer, text: str, max_tokens: int = 400):
    """E1: kaynak > max_tokens ise paragraf sınırlarından böl (Bölüm 5-E1 adım 2).

    Dönüş: parça listesi — chunk_bounds üretimden sonra gerçek token
    sınırlarına göre hesaplanır (her parçanın cevap içindeki aralığı).
    """
    if len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= max_tokens:
        return [text]
    paragraphs = [p for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paragraphs:
        cand = (cur + "\n" + p) if cur else p
        if len(tokenizer(cand, add_special_tokens=False)["input_ids"]) > max_tokens and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks if len(chunks) > 1 else [text]


def generate_batch(
    model_name: str,
    prompt_records: list[dict],
    *,
    watermark: bool,
    key: int | None,
    key_id: str = "main",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    seed: int = 1234,
    k: int = 2,
    tp_size: int = 1,
    gpu_mem_util: float = 0.92,
    chunk_mode: bool = False,
    tokenizer=None,
    extra_system: str | None = None,
    log_fn=print,
    sampling_params_extra: dict | None = None,
    llm=None,
    think_leak_tolerance: float = THINK_LEAK_TOLERANCE,
):
    """Tek watermark koşulu için toplu üretim.

    prompt_records: [{"id":..., "src_id":..., "topic":..., "prompt":...} | ...]
    chunk_mode: E1 parçalı yeniden yazım (context=cumulative).
    llm: hazır bir vLLM örneği verilirse yeniden kullanılır (model yükleme
    ~dakikalar sürer; aynı modelle birden çok koşul için make_llm kullanın).
    Dönüş: JSONL kayıtları listesi.
    """
    from vllm import SamplingParams

    from wm.vllm_proc import watermark_extra_args

    own_llm = llm is None
    if own_llm:
        llm = make_llm(model_name, tp_size=tp_size, gpu_mem_util=gpu_mem_util,
                       seed=seed)
    try:
        return _generate(llm, SamplingParams, watermark_extra_args, model_name,
                         prompt_records, watermark, key, key_id, temperature,
                         top_p, max_new_tokens, seed, k, chunk_mode, tokenizer,
                         extra_system, log_fn, sampling_params_extra,
                         think_leak_tolerance)
    finally:
        if own_llm:
            close_llm(llm)


def make_llm(model_name: str, *, tp_size: int = 1, gpu_mem_util: float = 0.92,
             seed: int = 1234):
    """vLLM örneği + watermark LogitsProcessor sınıfı kaydı (V1 yolu).

    EngineCore ayrı süreçtir ve işlemci sınıfını ADIYLA içe aktarır; bu yüzden
    proje kökü PYTHONPATH'e eklenir (alt süreç ortamı devralır).
    """
    from vllm import LLM

    from wm.vllm_proc import FQCN

    # Yeni vLLM sürümleri iki model-runner arasında seçim yapar; V2'nin
    # LogitsProcessor tabanı FARKLIDIR (wm/vllm_proc.py V1 arayüzüne yazıldı).
    # V1'e sabitle (alt süreç ortamı devralır). Elle ezmek için: dışarıdan ayarla.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    parts = [str(ROOT)] + [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(parts))
    return LLM(model=model_name, tensor_parallel_size=tp_size,
               **llm_memory_kwargs(gpu_mem_util), seed=seed,
               dtype="bfloat16", logits_processors=[FQCN])


def llm_memory_kwargs(gpu_mem_util: float = 0.92) -> dict:
    """Bellek ayarları. Ortam değişkenleri (hepsi isteğe bağlı) küçük GPU içindir:
      WM_GPU_MEM_UTIL  vLLM'in alacağı VRAM oranı (varsayılan 0.92; 6GB için 0.80)
      WM_MAX_MODEL_LEN en uzun istek+üretim tokenı (varsayılan 8192; modelin
                       262k bağlamı için KV ayırmaya çalışmasın)
      WM_EAGER=1       CUDA graph/derleme kapalı (daha az bellek, hızlı başlangıç)
    """
    return {
        "gpu_memory_utilization": float(os.environ.get("WM_GPU_MEM_UTIL", gpu_mem_util)),
        "max_model_len": int(os.environ.get("WM_MAX_MODEL_LEN", 8192)),
        "enforce_eager": os.environ.get("WM_EAGER", "0") == "1",
    }


def close_llm(llm) -> None:
    """GPU belleğini GERİ ver: aynı süreçte ikinci bir LLM açılabilsin."""
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:  # noqa: BLE001 — kapatma en iyi çabadır
        pass
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _generate(llm, SamplingParams, watermark_extra_args, model_name,
              prompt_records, watermark, key, key_id, temperature, top_p,
              max_new_tokens, seed, k, chunk_mode, tokenizer, extra_system,
              log_fn, sampling_params_extra, think_leak_tolerance=THINK_LEAK_TOLERANCE):
    tok = tokenizer or llm.get_tokenizer()
    leaks: set = set()   # <think> sızdıran kayıt id'leri (kural 6)
    extra = dict(sampling_params_extra or {})
    # Kural 6: thinking token'ları ÜRETİMDE yasak (model T yüksekken ara sıra
    # kendiliğinden </think> üretiyor; şablon sağlam). Hem wm'li hem wm'siz
    # isteklere aynı yasak → koşullar karşılaştırılabilir kalır.
    ban = think_token_ids(tok)

    def make_sp():
        """İstek parametreleri. Watermark'lı: greedy + extra_args (T/top-p
        işlemcinin içinde). Watermark'sız: vLLM'in standart örnekleyicisi."""
        if watermark:
            return SamplingParams(
                temperature=0, max_tokens=max_new_tokens,
                extra_args={**watermark_extra_args(key, temperature, top_p, k),
                            "wm_ban": ban},
                **extra)
        return SamplingParams(temperature=temperature, top_p=top_p,
                              max_tokens=max_new_tokens, seed=seed,
                              extra_args={"wm_ban": ban}, **extra)

    records = []
    t0 = time.time()

    if chunk_mode:
        # E1 parçalı yeniden yazım: uzun kaynaklar parçalara bölünür, her
        # parçaya önceki parçanın yeniden yazımı bağlam olarak verilir.
        # Parça SIRASI içinde bağımlılık var, kayıtlar arasında YOK: her tur
        # (parça indeksi) tüm kayıtlar için tek toplu çağrıda üretilir.
        chunk_lists = [chunk_source_text(tok, r["source_answer"])
                       for r in prompt_records]
        st = [{"bounds": [], "ids": [], "text": "", "prev": "",
               "templ": None, "pids": None} for _ in prompt_records]
        n_rounds = max(len(c) for c in chunk_lists)
        for ci in range(n_rounds):
            idxs = [i for i, c in enumerate(chunk_lists) if len(c) > ci]
            texts = []
            for i in idxs:
                rec = prompt_records[i]
                instr = rec.get("instruction", REWRITE_INSTRUCTION)
                chunk = chunk_lists[i][ci]
                if ci > 0:
                    user_msg = (f"{instr}\n\nÖnceki parçanın yeniden yazılmış hâli:\n"
                                f"{st[i]['prev']}\n\nŞimdi şu parçayı yeniden yaz:\n{chunk}")
                else:
                    user_msg = f"{instr}\n\n{chunk}"
                ctx_templ = render_chat(tok, [{"role": "user", "content": user_msg}])
                if ci == 0:
                    # saklanan prompt, modele GERÇEKTEN verilen ilk istemdir
                    st[i]["templ"] = ctx_templ
                    st[i]["pids"] = [int(t) for t in tok(
                        ctx_templ, add_special_tokens=False)["input_ids"]]
                texts.append(ctx_templ)
            outs = llm.generate(texts, make_sp(), use_tqdm=False)
            for i, res in zip(idxs, outs):
                ids = [int(t) for t in res.outputs[0].token_ids]
                txt = res.outputs[0].text
                if has_thinking(txt):
                    leaks.add(prompt_records[i]["id"])
                s_ = st[i]
                s_["bounds"].append((len(s_["ids"]), len(s_["ids"]) + len(ids)))
                s_["ids"].extend(ids)
                s_["text"] += txt
                s_["prev"] = txt
            log_fn(f"  parça turu {ci + 1}/{n_rounds}: {len(idxs)} istek")
        for rec, s_ in zip(prompt_records, st):
            records.append(_make_record(
                rec, s_["templ"], s_["pids"], s_["text"], s_["ids"],
                watermark, key_id, key, temperature, top_p, model_name,
                skipped=[], chunk_bounds=s_["bounds"]))
    else:
        templated = build_prompts(tok, [r["prompt"] for r in prompt_records],
                                  extra_system)
        # Watermark'lıysa AdapterLogitsProcessor her istek için ayrı örnek
        # üretir (istek başına pencere durumu); tek SamplingParams yeterli.
        outs = llm.generate([t for t, _ in templated], make_sp(), use_tqdm=True)
        for rec, (templ, pids), res in zip(prompt_records, templated, outs):
            txt = res.outputs[0].text
            if has_thinking(txt):
                leaks.add(rec["id"])
            ids = [int(t) for t in res.outputs[0].token_ids]
            records.append(_make_record(
                rec, templ, pids, txt, ids, watermark, key_id, key,
                temperature, top_p, model_name, skipped=[]))

    enforce_think_leaks(records, leaks, think_leak_tolerance, log_fn)
    dt = time.time() - t0
    n_tok = sum(len(r["answer_ids"]) for r in records)
    log_fn(f"  üretim: {len(records)} örnek, {n_tok} token, {dt:.0f}s "
           f"({n_tok / max(dt, 1e-9):.0f} tok/s)")
    update_manifest(**{f"speed.{model_name.split('/')[-1]}_tok_per_s":
                       round(n_tok / max(dt, 1e-9), 1)})
    return records


def think_token_ids(tok) -> list[int]:
    """<think>/</think> tek-token kimlikleri (tokenizer'da yoksa boş)."""
    ids = set()
    for s in ("<think>", "</think>"):
        enc = tok(s, add_special_tokens=False)["input_ids"]
        if len(enc) == 1:
            ids.add(int(enc[0]))
    return sorted(ids)


def has_thinking(text: str) -> bool:
    return "<think>" in text or "</think>" in text


def enforce_think_leaks(records, leaks, tolerance, log_fn=print):
    """Kural 6. Şablon bozuksa (thinking kapanmamışsa) ÇOĞU çıktı sızdırır →
    oran toleransı aşarsa DUR. Tolerans içindeki tek tük sızıntılar koşuyu
    DURDURMAZ: kayıt `think_leak=True` ile işaretlenir, sayısı manifest'e
    (think_leaks.*) yazılır; sessizce gizlenmez."""
    rate = len(leaks) / max(len(records), 1)
    if leaks and rate > tolerance:
        raise RuntimeError(
            f"Thinking modu açık kalmış! {len(leaks)}/{len(records)} çıktı "
            f"(%{100 * rate:.1f} > tolerans %{100 * tolerance:.1f}); örnek id'ler: "
            f"{sorted(map(str, leaks))[:5]} — kural 6: dur ve şablonu düzelt.")
    for r in records:
        r["think_leak"] = r["id"] in leaks
    if leaks:
        log_fn(f"  UYARI: {len(leaks)}/{len(records)} çıktıda <think> sızıntısı "
               f"(%{100 * rate:.2f}, tolerans %{100 * tolerance:.0f} içinde) — koşu "
               f"sürüyor; işaretlendi: think_leak=True")
    if records:
        tag = str(records[0].get("gen_model", "?")).split("@")[0].split("/")[-1]
        update_manifest(**{f"think_leaks.{tag}_{'wm' if records[0].get('wm') else 'nowm'}":
                           {"n_leak": len(leaks), "n": len(records),
                            "rate": round(rate, 5)}})


def _make_record(rec, templ, pids, answer, answer_ids, watermark, key_id, key,
                 temperature, top_p, model_name, skipped, chunk_bounds=None):
    return {
        "id": rec["id"],
        "src_id": rec.get("src_id", rec["id"]),
        "split": rec.get("split", "train"),
        "topic": rec.get("topic", ""),
        "prompt": rec["prompt"],
        "prompt_ids": list(pids),
        "prompt_templated": templ,
        "answer": answer,
        "answer_ids": [int(t) for t in answer_ids],
        "wm": bool(watermark),
        "key_id": key_id,
        "key": int(key) if watermark else None,
        "k": 2,
        "gen_model": f"{model_name}@{hf_commit_hash(model_name) or '?'}",
        "T": temperature,
        "top_p": top_p,
        "skipped_positions": skipped,
        "chunk_bounds": chunk_bounds,
        **({"source_answer": rec["source_answer"]} if rec.get("source_answer") else {}),
    }


REWRITE_INSTRUCTION = (
    "Aşağıdaki metni anlamını, içerdiği bilgileri ve biçimini koruyarak "
    "yeniden yaz. Bilgi ekleme ya da çıkarma. Yalnızca yeniden yazılmış "
    "metni ver.")


def save_records(records, path: Path):
    dump_jsonl(records, path)
    print(f"  yazıldı: {path} ({len(records)} satır)")


def load_prompt_records(path: Path, limit=None):
    recs = list(read_jsonl(path))
    return recs[:limit] if limit else recs