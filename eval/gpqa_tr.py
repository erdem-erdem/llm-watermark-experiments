"""GPQA Extended Türkçe downstream doğruluk — Bölüm 4.3.

0-shot, dört seçenek sabit tohumla bir kez karıştırılıp A–D sıralanır (bütün
modellerde aynı karıştırma). Puanlama: " A"/" B"/" C"/" D" tokenlarının
log-olasılıkları arasında en yükseği. 1.000 bootstrap %95 GA.
**Lisans notu:** GPQA soruları hiçbir çıktı dosyasına yazılmaz; yalnızca
sayısal sonuçlar raporlanır.

  python -m eval.gpqa_tr --config configs/eval.yaml [--smoke]
"""

from __future__ import annotations

import json
import random
import sys
import time

import numpy as np
import torch

from wm.util import (RESULTS, dump_json, log_cmd, load_config, parse_args,
                     record_duration, update_manifest)

LETTERS = [" A", " B", " C", " D"]


def load_gpqa(cfg):
    from datasets import load_dataset
    ds = load_dataset(cfg["gpqa_dataset"], split=cfg.get("gpqa_split", "train"))
    return list(ds)


def shuffle_options(cfg, rows):
    """Seçenekleri sabit tohumla bir kez karıştır (tüm modeller aynı)."""
    rng = random.Random(cfg.get("gpqa_shuffle_seed", 20260814))
    shuffled = []
    order_list = []
    for r in rows:
        opts = [r["Correct Answer"], r["Incorrect Answer 1"],
                r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        idx = list(range(4))
        rng.shuffle(idx)
        shuffled.append([opts[i] for i in idx])
        order_list.append(idx)  # doğru cevabın yeni konumu
    return shuffled, [idx.index(0) for idx in order_list]


def format_prompt(question, options):
    lines = [question]
    for letter, opt in zip("ABCD", options):
        lines.append(f"{letter}. {opt}")
    lines.append("Cevap:")
    return "\n".join(lines)


@torch.no_grad()
def score_choice(model, tokenizer, prompt, device="cuda"):
    """ A/B/C/D logitlerini karşılaştır (tek ileri geçiş)."""
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"].to(device)
    logits = model(input_ids=ids).logits[0, -1]
    # seçenek harfleri tek token olmayabilir; " A" dizisinin ilk tokenını kullan
    cand_ids = []
    for letter in LETTERS:
        t = tokenizer(letter, add_special_tokens=False)["input_ids"]
        cand_ids.append(t[0])
    vals = logits[torch.tensor(cand_ids, device=device)]
    return int(torch.argmax(vals).item())


def bootstrap_ci(correct: list[int], n_boot: int = 1000, seed: int = 0,
                 rng=None):
    arr = np.asarray(correct, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng(seed)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def eval_model(model_path, rows, options, gold, tokenizer=None, device="cuda",
               log_fn=print):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = tokenizer or AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to(device).eval()
    correct = []
    for q, opts, g in zip(rows, options, gold):
        prompt = format_prompt(q, opts)
        pred = score_choice(model, tok, prompt, device)
        correct.append(int(pred == g))
    del model
    torch.cuda.empty_cache()
    lo, hi = bootstrap_ci(correct, rng=np.random.default_rng(20260814))
    acc = float(np.mean(correct))
    log_fn(f"  {model_path}: doğruluk {acc:.3f} (%95 GA [{lo:.3f}, {hi:.3f}])")
    return {"model": model_path, "accuracy": acc, "ci_lo": lo, "ci_hi": hi,
            "n": len(correct)}


def main():
    argv = parse_args("GPQA-TR downstream değerlendirme")
    cfg = load_config(argv.config)
    log_cmd(sys.argv)
    t0 = time.time()
    parts_dir = RESULTS / "eval_parts" / "gpqa"

    rows = load_gpqa(cfg)
    questions = [r.get("Question") or r.get("question") for r in rows]
    options, gold = shuffle_options(cfg, rows)
    if argv.smoke:
        rows, questions, options, gold = (rows[:20], questions[:20],
                                          options[:20], gold[:20])
    print(f"GPQA-TR: {len(rows)} soru (soru metinleri çıktılara yazılmaz)")

    only = None
    for i, a in enumerate(sys.argv):
        if a == "--only":
            only = set(sys.argv[i + 1].split(","))
    models = cfg["models"]
    if only:
        models = [m for m in models if m["label"] in only]
    results = []
    for m in models:
        p = m["path"]
        if not __import__("pathlib").Path(p).exists():
            print(f"  atlandı (yok): {p}")
            continue
        label = m.get("label", p)
        r = eval_model(p, questions, options, gold)
        r["label"] = label
        results.append(r)
        # parça dosyası: GPU'ya bölünmüş koşumlarda model başına
        dump_json({"label": label, **{k: v for k, v in r.items()
                                      if k != "label"}},
                  parts_dir / f"{label}.json")

    # parçaları birleştir (paralel koşumlardan kalanlar dahil)
    if parts_dir.is_dir():
        for pf in sorted(parts_dir.glob("*.json")):
            pr = json.loads(pf.read_text(encoding="utf-8"))
            if pr["label"] not in {r["label"] for r in results}:
                results.append(pr)

    accs = {r["label"]: r["accuracy"] for r in results}
    base_acc = accs.get("base")
    note = None
    if base_acc is not None and abs(base_acc - 0.25) <= 0.03:
        note = ("Taban model %25 ± 3 içinde — bu ölçüt ayırt edici değil; "
                "SUMMARY'de açıkça belirtilecek.")
        print(note)
    out = {"results": results, "n_questions": len(rows), "note": note,
           "shuffle_seed": cfg.get("gpqa_shuffle_seed", 20260814)}
    dump_json(out, RESULTS / "E3" / "gpqa_tr.json")
    dump_json(out, RESULTS / "E4" / "gpqa_tr.json")
    update_manifest(**{"eval.gpqa": {r["label"]: r["accuracy"] for r in results}})
    record_duration("gpqa_eval", time.time() - t0)
    print("GPQA-TR tamam")


if __name__ == "__main__":
    main()