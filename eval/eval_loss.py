"""Ayrılmış eval_loss ölçümü — Bölüm 4.3 (E3-E4 ikinci downstream ölçütü).

500 InstrucTurca örneği (watermark'sız özgün cevaplar) üzerinde cevap
tokenları başına ortalama kayıp ve perplexity; bütün A ve B modelleri ve
taban model; örnek düzeyinde bootstrap %95 GA.

  python -m eval.eval_loss --config configs/eval.yaml [--smoke]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from wm.util import (ROOT, RESULTS, dump_json, log_cmd, load_config,
                     parse_args, read_jsonl, record_duration, update_manifest)

DATA_DIR = ROOT / "data"


@torch.no_grad()
def per_sample_loss(model, rec, device="cuda", max_len=2048):
    """Cevap tokenları üzerinde ortalama kayıp (labels=-100 prompt'ta)."""
    ids = (list(rec["prompt_ids"]) + list(rec["answer_ids"]))[:max_len]
    labels = ([-100] * len(rec["prompt_ids"]) + list(rec["answer_ids"]))[:max_len]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    labels_t = torch.tensor([labels], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, labels=labels_t)
    # out.loss ortalama; örnek düzeyi için n_token ile çarpıp geri böleriz
    n_ans = sum(1 for l in labels if l != -100)
    return float(out.loss), n_ans


def eval_model(model_path, recs, device="cuda", log_fn=print):
    """recs: chat_format.to_chat_records ile ŞABLONLU (eğitimle aynı biçim)."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to(device).eval()
    losses = []
    n_tok = 0
    for rec in recs:
        l, n = per_sample_loss(model, rec, device)
        losses.append(l)
        n_tok += n
    del model
    torch.cuda.empty_cache()
    mean_loss = float(np.mean(losses))
    ppl = float(np.exp(mean_loss))
    rng = np.random.default_rng(20260814)
    boots = [rng.choice(np.asarray(losses), size=len(losses), replace=True).mean()
             for _ in range(1000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    log_fn(f"  {model_path}: loss {mean_loss:.4f}, ppl {ppl:.3f} "
           f"(GA [{lo:.4f}, {hi:.4f}])")
    return {"model": model_path, "mean_loss": mean_loss, "ppl": ppl,
            "ci_lo": float(lo), "ci_hi": float(hi), "n": len(recs),
            "total_answer_tokens": n_tok}


def main():
    argv = parse_args("eval_loss ölçümü")
    cfg = load_config(argv.config)
    log_cmd(sys.argv)
    t0 = time.time()
    parts_dir = RESULTS / "eval_parts" / "eval_loss"
    recs = list(read_jsonl(DATA_DIR / "eval_loss.jsonl"))
    if argv.smoke:
        recs = recs[:20]
    # eğitimle AYNI biçim: şablonlu prompt + cevap + <|im_end|>
    from train.chat_format import load_template_tokenizer, to_chat_records
    recs = to_chat_records(recs, load_template_tokenizer(), add_end=True)
    print(f"eval_loss: {len(recs)} örnek")

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
        if not Path(p).exists():
            print(f"  atlandı (yok): {p}")
            continue
        r = eval_model(p, recs)
        r["label"] = m.get("label", p)
        results.append(r)
        dump_json({"label": r["label"],
                   **{k: v for k, v in r.items() if k != "label"}},
                  parts_dir / f"{r['label']}.json")

    # parçaları birleştir (paralel koşumlardan kalanlar dahil)
    if parts_dir.is_dir():
        import json as _json
        for pf in sorted(parts_dir.glob("*.json")):
            pr = _json.loads(pf.read_text(encoding="utf-8"))
            if pr["label"] not in {r["label"] for r in results}:
                results.append(pr)

    out = {"results": results, "n_samples": len(recs)}
    dump_json(out, RESULTS / "E3" / "eval_loss.json")
    dump_json(out, RESULTS / "E4" / "eval_loss.json")
    update_manifest(**{"eval.eval_loss": {r["label"]: r["mean_loss"]
                                          for r in results}})
    record_duration("eval_loss", time.time() - t0)
    print("eval_loss tamam")


if __name__ == "__main__":
    main()