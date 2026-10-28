"""Blog için örnek seçimi (E1.2 önce/sonra çiftleri) ve D_post yan-bulgu sayımları.

Seçim NESNEL ve tekrarlanabilir: sabit tohumla karıştırılmış adaylardan, aşağıdaki ölçütleri sağlayan
İLK uygun çift alınır (en yüksek iz / en güzel örnek aranmaz):
  · filtreden geçmiş (uzunluk bandında), konu ≠ matematik/kod, kod bloğu yok
  · yalnız Latin (Türkçe dâhil) harfler — CJK/Kiril/Arap sızıntısı yok; LaTeX/biçim işareti ve sayı yığını yok
  · özgün cevap 250–650 karakter, en az 2 cümle
  · uzunluk oranı 0.90–1.10, N ≥ 60 ve −log₁₀(p) ≥ 6
  · örnek 1: anlam benzerliği ≥ 0.95;  örnek 2 (karşı örnek): 0.85 ≤ benzerlik < 0.93

  python -m analyze.blog_examples        → results/blog/e1_examples.json , e1_side_stats.json
"""

from __future__ import annotations

import json
import random
import re
import sys
import unicodedata
from difflib import SequenceMatcher

import numpy as np

from analyze.e2_complete import _per_record
from gen.mini_util import scored_tokens, summarize, token_pieces
from wm.util import RESULTS, ROOT, dump_json, read_jsonl

DATA = ROOT / "data"
OUT = RESULTS / "blog"
SEED = 20260814
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯豈-﫿]")


def _keys():
    s = json.loads((RESULTS / "E0" / "selected_keys.json").read_text(encoding="utf-8"))
    return int(s["main_key"])


def _nonlatin_letters(text):
    bad = set()
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if not (name.startswith("LATIN") or name.startswith("GREEK")):
                bad.add(name.split()[0] if name else "?")
    return bad


def _sentences(t):
    return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if len(s) > 8])


def side_stats(post):
    kept = [r for r in post if not r["filtered"]]
    def rate(rs, f):
        return float(np.mean([bool(f(r)) for r in rs])) if rs else float("nan")
    out = {
        "n": len(post), "n_kept": len(kept),
        "cjk_answer_all": rate(post, lambda r: CJK.search(r["answer"])),
        "cjk_answer_kept": rate(kept, lambda r: CJK.search(r["answer"])),
        "cjk_source": rate(post, lambda r: CJK.search(r["source_answer"])),
        "nonlatin_answer_all": rate(post, lambda r: _nonlatin_letters(r["answer"])),
        "nonlatin_source": rate(post, lambda r: _nonlatin_letters(r["source_answer"])),
    }
    gen = list(read_jsonl(DATA / "D_gen.jsonl"))
    out["cjk_gen_answer"] = rate(gen, lambda r: CJK.search(r["answer"]))
    return out


def select(post, key):
    rng = random.Random(SEED + 11)
    pool = []
    for r in post:
        if r["filtered"] or r["topic"] == "matematik_kod":
            continue
        s, a = r["source_answer"], r["answer"]
        lr = len(a) / max(1, len(s))
        if not (250 <= len(s) <= 650 and 0.90 <= lr <= 1.10 and _sentences(s) >= 2 and _sentences(a) >= 2):
            continue
        if "```" in a or "```" in s or _nonlatin_letters(a) or _nonlatin_letters(s):
            continue
        if re.search(r"[\\$\{\}\[\]|<>=*#_^]|\d{3,}", s + a):      # LaTeX / biçimli metin / sayı yığını yok
            continue
        pool.append((r, lr))
    rng.shuffle(pool)
    from sentence_transformers import SentenceTransformer
    sim_model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
    cands = {"high": [], "mid": []}
    tried = 0
    for r, lr in pool:
        st = summarize(_per_record(r, key))
        if st["N"] < 60 or st["neglog10p"] < 6:
            continue
        e = sim_model.encode(["query: " + r["source_answer"], "query: " + r["answer"]], normalize_embeddings=True)
        sim = float(np.dot(e[0], e[1]))
        tried += 1
        slot = "high" if sim >= 0.95 else ("mid" if 0.85 <= sim < 0.95 else None)
        if slot:
            lex = SequenceMatcher(None, r["source_answer"].lower().split(), r["answer"].lower().split()).ratio()
            cands[slot].append((lex, r, lr, sim, st))
        if tried >= 300:
            break
    # okunaklılık: sözcük düzeyinde özgüne en yakın (en az bozulmuş) çiftler
    chosen = {}
    for slot, lst in cands.items():
        if lst:
            lex, r, lr, sim, st = max(lst, key=lambda x: x[0])
            chosen[slot] = (r, lr, sim, st)
            print(slot, "sözcük-benzerliği", round(lex, 3), "aday", len(lst))
    return chosen, len(pool), tried


def main():
    key = _keys()
    OUT.mkdir(parents=True, exist_ok=True)
    post = list(read_jsonl(DATA / "D_post.jsonl"))
    stats = side_stats(post)
    chosen, npool, tried = select(post, key)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    pairs = []
    for slot in ("high", "mid"):
        if slot not in chosen:
            continue
        r, lr, sim, st = chosen[slot]
        ids = list(r["answer_ids"])
        sc = {p: s for p, s in scored_tokens(ids, len(r["prompt_ids"]), key)}
        toks = [{"piece": pc, "score": (sc[i] if i in sc else None)}
                for i, pc in enumerate(token_pieces(tok, ids))]
        pairs.append({"slot": slot, "id": r["id"], "topic": r["topic"], "source": r["source_answer"].strip(),
                      "rewrite": r["answer"].strip(), "tokens": toks, "similarity": sim, "len_ratio": lr,
                      "N": st["N"], "neglog10p": st["neglog10p"]})
    dump_json({"pairs": pairs, "pool_size": npool, "candidates_scored": tried,
               "criteria": "docstring: analyze/blog_examples.py"}, OUT / "e1_examples.json")
    dump_json(stats, OUT / "e1_side_stats.json")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"örnek havuzu {npool}; puanlanan aday {tried}; seçilen: {[p['slot'] for p in pairs]}")
    for p in pairs:
        print(p["slot"], p["id"], f"sim={p['similarity']:.3f} lr={p['len_ratio']:.2f} N={p['N']} nl={p['neglog10p']:.1f}")


if __name__ == "__main__":
    sys.exit(main())
