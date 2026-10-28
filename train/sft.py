"""SFT eğitimi — SPEC.md Bölüm 5, KARAR-3/4.

HF transformers Trainer; tam fine-tuning, bf16, AdamW lr 2e-5 cosine %3
warmup, 3 epoch, efektif batch 32 (grad accum), max len 2048, packing kapalı,
kayıp YALNIZCA cevap tokenlarında, girdiler saklanan prompt_ids/answer_ids'ten
kurulur (kural 2 — yeniden tokenizasyon yok).

  python -m train.sft --config train/configs/A-gen-100.yaml [--smoke] [--force]

Koşu karışımı (Bölüm 5-E3): ρ alt kümesindeki promptlar watermark'lı cevabı,
kalanlar D_orig cevabı ile eğitilir; toplam veri hacmi sabittir.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from wm.util import (RESULTS, ROOT, log_cmd, log_note, parse_args, read_jsonl,
                     record_duration, update_manifest)

DATA_DIR = ROOT / "data"
MODELS_DIR = RESULTS / "E3" / "models"
MIX_VERSION = "subset-v2"      # ρ = alt küme oranı (S10/S50/S100); v1: ρ iki kez uygulanıyordu


def load_by_id(path: Path) -> dict:
    return {r["id"]: r for r in read_jsonl(path)}


def build_training_set(cfg) -> list[dict]:
    """E3 karışımı: ρ oranında watermark'lı cevap, kalan D_orig.

    Veri sırası tohumla sabit ve her koşuda aynı (Bölüm 5 genel eğitim).
    """
    train = load_by_id(DATA_DIR / "train.jsonl")
    nested = json.loads((DATA_DIR / "nested_subsets.json").read_text(encoding="utf-8"))
    source_path = DATA_DIR / cfg["source_file"]        # D_gen/D_post/D_rew
    source = load_by_id(source_path)

    rho = float(cfg["rho"])
    if rho >= 0.999 and cfg.get("subset", "S100") == "S100":
        # A-rew: tüm veri watermark'sız yeniden yazımla (D_rew) eğitilir;
        # _row'un use_wm=None yolu kaynak yoksa D_orig'e düşer.
        return [_row(tr, source, rid) for rid, tr in train.items()]

    # S10 ⊂ S50 ⊂ S100 alt kümesi ZATEN ρ'yu tanımlar (S10 = tüm prompt'ların %10'u).
    # (Eski sürüm ρ'yu alt kümeye BİR KEZ DAHA uyguluyordu: A-*-10 → %1, A-*-50 → %25.)
    subset_ids = sorted(nested[cfg["subset"]])
    share = len(subset_ids) / max(1, len(train))
    if abs(share - rho) > 0.01:
        raise ValueError(f"{cfg['name']}: rho={rho} ama {cfg['subset']} tüm verinin "
                         f"%{100 * share:.0f}'i — yapılandırma tutarsız")
    wm_ids = set(subset_ids)

    rows = []
    for rid, tr in train.items():
        rows.append(_row(tr, source, rid, use_wm=rid in wm_ids))
    return rows


def _row(tr, source, rid, use_wm=None):
    """Tek eğitim satırı: prompt_ids + (watermark'lı ya da özgün) answer_ids."""
    if use_wm is None:                                   # A-rew: tüm veri D_rew
        src = source.get(rid)
        ans = src["answer_ids"] if src else tr["answer_ids"]
        wm = bool(src)
    elif use_wm:
        ans = source[rid]["answer_ids"]
        wm = True
    else:
        ans = tr["answer_ids"]
        wm = False
    # prompt METNİ taşınır: şablonlu prompt_ids main()'de chat_format ile kurulur
    return {"id": rid, "prompt": tr["prompt"], "prompt_ids": tr["prompt_ids"],
            "answer_ids": ans, "wm": wm, "topic": tr["topic"]}


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows, max_len=2048):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids = (list(r["prompt_ids"]) + list(r["answer_ids"]))[: self.max_len]
        labels = ([-100] * len(r["prompt_ids"]) + list(r["answer_ids"]))[: self.max_len]
        if len(ids) > self.max_len:
            labels = ([-100] * len(r["prompt_ids"]) +
                      list(r["answer_ids"]))[: self.max_len]
        return {"input_ids": ids, "labels": labels}


class Collator:
    """Doldurma: padding ile input_ids/labels/attention_mask. Packing yok."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids = list(f["input_ids"]); lab = list(f["labels"])
            pad = n - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main():
    argv = parse_args("SFT eğitimi")
    from wm.util import load_config
    cfg = load_config(argv.config)
    log_cmd(sys.argv)
    t0 = time.time()
    if argv.smoke:
        cfg = dict(cfg)
        cfg["epochs"] = 1
        smoke_rows = cfg.get("smoke_n", 50)

    run_name = cfg["name"]
    out_dir = Path(cfg.get("out_dir", MODELS_DIR / run_name))
    # Tamamlanma işareti: yalnızca model KAYDEDİLDİKTEN sonra yazılır. Yarıda
    # kalmış (çökmüş) bir koşunun klasörü "bitti" sayılmaz, yeniden eğitilir.
    # İşaret eğitim BİÇİMİNİ de taşır: biçim değişirse (ör. şablonsuz → şablonlu)
    # eski model "bitti" sayılmaz, otomatik yeniden eğitilir.
    from train.chat_format import (SFT_FORMAT, TEMPLATE_MODEL, end_ids,
                                   load_template_tokenizer, to_chat_records)
    done = out_dir / "_TRAINING_DONE"
    if done.exists() and not argv.force:
        try:
            fmt = json.loads(done.read_text(encoding="utf-8")).get("format")
        except Exception:  # noqa: BLE001 — eski düz metin işaret
            fmt = None
        try:
            mix = json.loads(done.read_text(encoding="utf-8")).get("mix")
        except Exception:  # noqa: BLE001
            mix = None
        # veri karışımı sürümü: alt-küme koşuları (S10/S50) eski ρ-ρ hatasıyla
        # eğitildiyse (mix yok) yeniden eğitilir; S100 koşuları etkilenmedi
        mix_ok = mix == MIX_VERSION or cfg.get("subset", "S100") == "S100"
        if fmt == SFT_FORMAT and mix_ok:
            print(f"eğitim tamam, atlanıyor: {out_dir} (--force ile zorla)")
            return
        if fmt == SFT_FORMAT and not mix_ok:
            print(f"eski veri karışımı ({mix}) ≠ {MIX_VERSION}: yeniden eğitiliyor: {out_dir}")
        print(f"eski eğitim biçimi ({fmt}) ≠ {SFT_FORMAT}: yeniden eğitiliyor: {out_dir}")
    if out_dir.exists():
        print(f"yarım kalmış çıktı bulundu, yeniden eğitiliyor: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_training_set(cfg)
    if argv.smoke:
        rows = rows[: smoke_rows]

    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    # Post-trained şablon tokenizer'ı (Bölüm 4.1): prompt şablonlu, cevap +
    # <|im_end|> — üretim/tespit tarafıyla aynı biçim. Model klasörüne de BU
    # tokenizer kaydedilir (şablon + bitiş token'ı öğrencinin üretiminde de geçerli).
    tokenizer = load_template_tokenizer(cfg.get("template_model", TEMPLATE_MODEL))
    rows = to_chat_records(rows, tokenizer, add_end=True)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    # flash-attn paketi yoksa (kurulumu derleme gerektirir) PyTorch SDPA'ya düş:
    # H100'de SDPA zaten flash çekirdeğini kullanır; sonuç matematiksel aynı.
    import importlib.util
    attn = ("flash_attention_2"
            if cfg.get("flash_attn", True) and importlib.util.find_spec("flash_attn")
            else "sdpa")
    print(f"attn_implementation: {attn}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], dtype=torch.bfloat16, attn_implementation=attn)
    model.config.use_cache = False

    train_ds = SFTDataset(rows, max_len=cfg["max_len"])
    coll = Collator(pad_id)

    # WM_TRAIN_LOWMEM=1 (küçük GPU duman testi): Adafactor (neredeyse durumsuz
    # optimizer), batch 1, gradient checkpointing. Üretim koşusu etkilenmez.
    lowmem = os.environ.get("WM_TRAIN_LOWMEM", "0") == "1"
    per_device = 1 if lowmem else cfg.get("per_device_batch", 4)
    if lowmem:
        model.gradient_checkpointing_enable()
        print("WM_TRAIN_LOWMEM: adafactor + batch 1 + gradient checkpointing")
    accum = max(1, cfg["effective_batch"] // per_device)
    # %3 ısınma, ADIM SAYISI olarak hesaplanır: transformers v5 `warmup_ratio`'yu
    # kaldırdı (v4'te vardı); tam sayı warmup_steps iki sürümde de aynı anlamda.
    # Optimizer adımı = ceil(ceil(n / batch) / accum) × epoch (tek GPU).
    import math
    total_steps = math.ceil(math.ceil(len(rows) / per_device) / accum) * cfg["epochs"]
    warmup_steps = max(1, math.ceil(0.03 * total_steps))
    print(f"optimizer adımı: {total_steps}, ısınma: {warmup_steps} (%3)")
    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        bf16=True,
        optim="adafactor" if lowmem else "adamw_torch",
        logging_steps=10,
        save_strategy="no",
        seed=cfg.get("train_seed", 20260814),
        data_seed=cfg.get("train_seed", 20260814),
        report_to=[],
        remove_unused_columns=False,
        # veri zaten tokenize edilmiş listeler: işçi süreç gereksiz; 0 ile
        # Python 3.14'ün yeni varsayılan süreç başlatma yöntemi (forkserver)
        # kaynaklı pickle sorunları da olmaz
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=coll)
    trainer.train()

    # üretimde <|im_end|>'de dursun (vLLM generation_config'deki eos'u da okur)
    model.generation_config.eos_token_id = end_ids(tokenizer)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    done.write_text(json.dumps({"format": SFT_FORMAT, "mix": MIX_VERSION,
                                "time": time.strftime("%Y-%m-%d %H:%M:%S")}),
                    encoding="utf-8")

    # kayıp eğrisi CSV (sağlık kontrolü: E3/loss/)
    history = trainer.state.log_history
    loss_rows = [{"step": h.get("step"), "loss": h.get("loss")}
                 for h in history if "loss" in h]
    import csv
    loss_dir = RESULTS / "E3" / "loss"
    loss_dir.mkdir(parents=True, exist_ok=True)
    with open(loss_dir / f"{run_name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss"])
        w.writeheader()
        w.writerows(loss_rows)

    update_manifest(**{f"trainings.{run_name}": {
        "config": argv.config, "n_rows": len(rows), "epochs": cfg["epochs"],
        "lr": cfg["lr"], "source_file": cfg["source_file"],
        "rho": cfg["rho"], "subset": cfg.get("subset"),
        "final_loss": loss_rows[-1]["loss"] if loss_rows else None}})
    record_duration(f"train_{run_name}", time.time() - t0)
    print(f"eğitim tamam: {out_dir}")


if __name__ == "__main__":
    main()