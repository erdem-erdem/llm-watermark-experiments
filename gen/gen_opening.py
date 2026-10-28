"""Mini deneme 1 — Blog açılışının (söz + ilk paragraf) watermark'lı yeniden yazımı.

data/opening_draft.txt (depoda yok; kendi metninizi koyun): ilk paragraf SÖZ, boş satırdan sonrası İLK PARAGRAF.
5 talimat varyantı × {söz, paragraf}: Qwen3.5-9B, ana anahtar, post-hoc yeniden
yazım talimatı. Söz ve paragraf AYRI isteklerle üretilir, böylece her biri kendi
N ve −log10 p değerini verir ("söz tek başına kanıt değil; paragraf biraz daha
güçlü; iz uzunlukla birikir").

ÖNEMLİ: T = cfg["opening_T"] (varsayılan 1.0), top-p 0.95. temperature=0 KULLANILMAZ:
sıcaklık 0'da dağılım tek-sıcak olur, watermark seçimi değiştiremez ve iz hiç oluşmaz.

Çıktılar (results/mini/):
  opening_candidates.md   her aday: söz/paragraf metni, N, −log10 p (okunur özet)
  opening_tokens.csv      variant,segment,pos,piece,scored,score  (token renkleri için)
  opening_curve.csv       variant,segment,n,neglog10p             (birikim eğrisi)

  python -m gen.gen_opening --config configs/mini.yaml [--smoke]
"""

from __future__ import annotations

import json
import sys
import time

from gen.mini_util import cum_curve, scored_tokens, summarize, token_pieces
from wm.util import (RESULTS, ROOT, dump_csv, log_cmd, load_config, parse_args,
                     record_duration, update_manifest)

MINI = RESULTS / "mini"

VARIANTS = [
    "Aşağıdaki metni anlamını, içerdiği bilgileri ve biçimini koruyarak yeniden yaz. "
    "Bilgi ekleme ya da çıkarma. Yalnızca yeniden yazılmış metni ver.",
    "Aşağıdaki metni aynı uzunlukta yeniden yaz. Bilgi ekleme ya da çıkarma. "
    "Yalnızca yeniden yazılmış metni ver.",
    "Aşağıdaki metni daha akıcı bir dille yeniden yaz; anlamını koru. "
    "Yalnızca yeniden yazılmış metni ver.",
    "Aşağıdaki metni yeniden yaz; cümle yapısını değiştir, içeriği koru. "
    "Yalnızca yeniden yazılmış metni ver.",
    "Aşağıdaki metni aynı üslupla, kendi sözcüklerinle yeniden yaz. "
    "Yalnızca yeniden yazılmış metni ver.",
]


def load_opening():
    p = ROOT / "data" / "opening_draft.txt"
    parts = [x.strip() for x in p.read_text(encoding="utf-8").split("\n\n") if x.strip()]
    if len(parts) < 2:
        sys.exit("data/opening_draft.txt: söz ve paragraf boş satırla ayrılmalı")
    return parts[0], parts[1]


def main():
    argv = parse_args("açılış mini denemesi")
    cfg = load_config(argv.config)
    log_cmd(sys.argv)
    t0 = time.time()
    sel = json.loads((RESULTS / "E0" / "selected_keys.json").read_text(encoding="utf-8"))
    main_key = int(sel["main_key"])
    k = cfg.get("k", 2)
    quote, para = load_opening()
    segs = {"soz": quote, "paragraf": para}
    variants = VARIANTS[:2] if argv.smoke else VARIANTS
    MINI.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    from gen.generate import generate_batch
    recs = [{"id": f"opening-v{i}-{s}", "prompt": f"{v}\n\n{text}"}
            for i, v in enumerate(variants) for s, text in segs.items()]
    batch = generate_batch(cfg["model"], recs, watermark=True, key=main_key,
                           key_id="main", temperature=cfg.get("opening_T", 1.0),
                           top_p=cfg.get("top_p", 0.95),
                           max_new_tokens=cfg.get("max_new_tokens", 400),
                           seed=cfg.get("seed", 20260814), k=k,
                           tp_size=cfg.get("tp_size", 1))

    tok_rows, curve_rows, md = [], [], [
        "# Açılış adayları (post-hoc yeniden yazım, ana anahtar)\n",
        f"Kaynak söz ({len(quote)} karakter):\n> {quote}\n",
        f"Kaynak paragraf ({len(para)} karakter):\n> {para}\n",
        f"Üretim: Qwen3.5-9B, T={cfg.get('opening_T', 1.0)}, top-p {cfg.get('top_p', 0.95)}, k={k}\n"]
    by_id = {r["id"]: r for r in batch}
    for i, v in enumerate(variants):
        md.append(f"\n## Aday {i}\n_Talimat:_ {v}\n")
        for s in segs:
            r = by_id[f"opening-v{i}-{s}"]
            ids = list(r["answer_ids"])
            st = scored_tokens(ids, len(r["prompt_ids"]), main_key, k)
            sc = [x for _, x in st]
            res = summarize(sc)
            pieces = token_pieces(tok, ids)
            scored_pos = {pos: x for pos, x in st}
            for pos, piece in enumerate(pieces):
                tok_rows.append({"variant": f"v{i}", "segment": s, "pos": pos,
                                 "piece": piece, "scored": int(pos in scored_pos),
                                 "score": scored_pos.get(pos, "")})
            for n, nl in cum_curve(sc):
                curve_rows.append({"variant": f"v{i}", "segment": s, "n": n,
                                   "neglog10p": nl})
            md.append(f"**{s}** — N = {res['N']}, −log₁₀(p) = {res['neglog10p']:.2f}\n\n"
                      f"> {r['answer'].strip()}\n")
            print(f"  v{i}/{s}: N={res['N']}  -log10p={res['neglog10p']:.2f}")
    dump_csv(tok_rows, MINI / "opening_tokens.csv",
             ["variant", "segment", "pos", "piece", "scored", "score"])
    dump_csv(curve_rows, MINI / "opening_curve.csv",
             ["variant", "segment", "n", "neglog10p"])
    (MINI / "opening_candidates.md").write_text("\n".join(md), encoding="utf-8", newline="")
    update_manifest(**{"mini.opening": {"n_variants": len(variants),
                                        "T": cfg.get("opening_T", 1.0)}})
    record_duration("gen_opening", time.time() - t0)
    print("açılış mini denemesi tamam")


if __name__ == "__main__":
    main()
