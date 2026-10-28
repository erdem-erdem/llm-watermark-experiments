"""Eğitim/değerlendirme girdi biçimi — TEK kaynak (Bölüm 4.1).

Öğrenci (Base) modeller, post-trained Qwen3.5 sohbet şablonuyla eğitilir ve
değerlendirilir; üretim/tespit tarafı (gen.generate.build_prompts) ile AYNI
prompt kimlikleri:

  prompt_ids = şablon(kullanıcı: prompt, asistan başlangıcı, thinking kapalı)
  answer_ids = cevap + <|im_end|>      (model DURMAYI öğrenir)

Kullananlar: train/sft.py (eğitim), eval/eval_loss.py (kayıp), analyze/detect_all.py
(E0 okuma modu — D_orig kayıtları). Üretilmiş kayıtlar (D_gen, ajan, A₀ çıktıları)
zaten bu şablonla üretildiği için dönüştürülmez.

SFT_FORMAT değişirse eski eğitimler otomatik yeniden yapılır (sft.py işareti).
"""

from __future__ import annotations

SFT_FORMAT = "chat-v1"
TEMPLATE_MODEL = "Qwen/Qwen3.5-9B"     # post-trained şablon (tokenizer'lar özdeş)
END_TOKEN = "<|im_end|>"


def load_template_tokenizer(name: str | None = None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name or TEMPLATE_MODEL)
    if not getattr(tok, "chat_template", None):
        raise RuntimeError(f"{name or TEMPLATE_MODEL}: sohbet şablonu yok")
    return tok


def end_ids(tok) -> list[int]:
    """Asistan turunu bitiren token (Qwen: <|im_end|>); yoksa eos."""
    vocab = tok.get_vocab()
    if END_TOKEN in vocab:
        return [int(vocab[END_TOKEN])]
    if tok.eos_token_id is None:
        raise RuntimeError("şablon tokenizer'ında bitiş token'ı yok")
    return [int(tok.eos_token_id)]


def to_chat_records(records, tok, add_end: bool = True) -> list[dict]:
    """Kayıtların prompt_ids'ini şablonlu hâle getir (prompt METNİNDEN), isteğe
    bağlı cevaba bitiş token'ı ekle. Girdi kayıtları değiştirilmez."""
    from gen.generate import build_prompts

    end = end_ids(tok) if add_end else []
    cache: dict[str, list[int]] = {}
    out = []
    for r in records:
        text = r["prompt"]
        if text not in cache:
            cache[text] = build_prompts(tok, [text])[0][1]
        out.append({**r, "prompt_ids": cache[text],
                    "answer_ids": list(r["answer_ids"]) + end})
    return out
