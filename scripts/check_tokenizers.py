"""Tokenizer karşılaştırma — Bölüm 4.1 (§4.1 uyum denetimi).

4 SWEEP modeli + Base: sözlük boyutu ve Türkçe örnek tokenizasyonu.
Uyumsuzluk varsa çıkış kodu 1 (duman koşusu burada durur).

  python scripts/check_tokenizers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-2B-Base",
]
SAMPLE = ("Suyun belirleyici özellikleri nelerdir? Veri bilimi, güvenilir "
          "metin üretimi ve 42 sayısının önemi.")

from transformers import AutoTokenizer  # noqa: E402


def main():
    ref = None
    ok = True
    for m in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(m)
        except Exception as e:
            print(f"  {m}: YÜKLENEMEDİ ({e})")
            ok = False
            continue
        vocab = len(tok)
        ids = tok(SAMPLE, add_special_tokens=False)["input_ids"]
        toks = tok.convert_ids_to_tokens(ids)
        if ref is None:
            ref = (ids, toks)
        else:
            same = ids == ref[0]
            print(f"  {m}: vocab={vocab}, token sayısı={len(ids)}, "
                  f"Base ile aynı={'EVET' if same else 'HAYIR'}")
            if not same:
                ok = False
                print(f"    örnek: {toks[:20]}")
        if ref is not None and m == MODELS[0]:
            print(f"  {m}: vocab={vocab}, token sayısı={len(ids)} (referans)")
            print(f"    örnek: {toks[:20]}")
    print("SONUÇ:", "UYUMLU" if ok else "UYUMSUZ — §4.1 kuralına göre durun")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()