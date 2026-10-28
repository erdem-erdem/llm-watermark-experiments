"""train/sft.py uçtan uca CPU ön-testi — GERÇEK transformers TrainingArguments/Trainer
ile minik rastgele modeli eğitir, kaydeder; eğitim BİÇİMİNİ (şablonlu prompt +
<|im_end|>), işareti ve atlama/yeniden eğitim mantığını denetler (~saniyeler).
Kütüphane sürümü uyumsuzluklarını (ör. v5 warmup_ratio, eski accelerate) GPU saatleri
harcanmadan yakalar.  python -m train.selftest
"""
import json, shutil, sys, tempfile
from pathlib import Path

import torch
from transformers import (AutoTokenizer, LlamaConfig, LlamaForCausalLM,
                          PreTrainedTokenizerFast)
from tokenizers import Tokenizer, models, pre_tokenizers

tmp = Path(tempfile.mkdtemp())
V0 = 64                  # kelime dağarcığı; +2 özel token (<|im_start|>, <|im_end|>)
V = V0 + 8               # model sözlüğü (özel token'lar sığsın)
TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }} {{ m['content'] }}"
            "<|im_end|> {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant "
            "{% endif %}")
mdir = tmp / "tiny"
cfg_m = LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256)
LlamaForCausalLM(cfg_m).save_pretrained(mdir)
vocab = {f"t{i}": i for i in range(V0)}
tk = Tokenizer(models.WordLevel(vocab, unk_token="t0")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
tok0 = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="t0", pad_token="t1", eos_token="t2",
                               additional_special_tokens=["<|im_start|>", "<|im_end|>"])
tok0.chat_template = TEMPLATE
tok0.save_pretrained(mdir)
IM_END = tok0.convert_tokens_to_ids("<|im_end|>")

import train.sft as sft
import wm.util as u
from train.chat_format import SFT_FORMAT

sft.DATA_DIR = tmp / "data"; sft.RESULTS = tmp / "results"; u.RESULTS = tmp / "results"
sft.DATA_DIR.mkdir(); (tmp / "results").mkdir()
g = torch.Generator().manual_seed(0)
rows = []
for k in range(40):
    p = torch.randint(3, V0, (6,), generator=g).tolist()
    rows.append({"id": f"train-{k:05d}", "topic": "olgusal",
                 "prompt": " ".join(f"t{i}" for i in p), "prompt_ids": p,
                 "answer_ids": torch.randint(3, V0, (10,), generator=g).tolist()})


def dump(name, rs):
    with open(sft.DATA_DIR / name, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")


dump("train.jsonl", rows)
dump("D_gen.jsonl", [{**r, "answer_ids": r["answer_ids"][::-1]} for r in rows])
ids = [r["id"] for r in rows]
(sft.DATA_DIR / "nested_subsets.json").write_text(json.dumps({"S10": ids[:4], "S50": ids[:20], "S100": ids}))
out_dir = tmp / "results" / "E3" / "models" / "T-run"
cfg_path = tmp / "T-run.yaml"
cfg_path.write_text(f"""name: T-run
base_model: "{mdir.as_posix()}"
template_model: "{mdir.as_posix()}"
source_file: "D_gen.jsonl"
subset: S50
rho: 0.5
epochs: 2
lr: 2.0e-5
effective_batch: 8
per_device_batch: 4
max_len: 128
train_seed: 1
data_seed: 1
out_dir: "{out_dir.as_posix()}"
""", encoding="utf-8")

# eğitime giren satırları yakala (biçim denetimi)
seen = []
_Orig = sft.SFTDataset


class _Spy(_Orig):
    def __init__(self, rows, max_len=2048):
        seen.extend(rows)
        super().__init__(rows, max_len)


sft.SFTDataset = _Spy


def train_once():
    sys.argv = ["sft", "--config", str(cfg_path)]
    seen.clear()
    sft.main()
    return len(seen) > 0          # True = gerçekten eğitti


try:
    assert train_once(), "eğitim yapılmadı"
    # 1) biçim: şablonlu prompt (rol başlığıyla) + cevap sonunda <|im_end|>
    r0 = seen[0]
    start = tok0.convert_tokens_to_ids("<|im_start|>")
    assert r0["prompt_ids"][0] == start, "prompt şablonla başlamıyor"
    assert r0["prompt_ids"][-1] != IM_END and start in r0["prompt_ids"][1:], \
        "asistan başlığı yok"
    assert all(r["answer_ids"][-1] == IM_END for r in seen), "cevap <|im_end|> ile bitmiyor"
    print("ok  biçim: şablonlu prompt + cevap sonunda <|im_end|> (tüm satırlar)")
    # 2) kayıt: model, ŞABLONLU tokenizer, eos = <|im_end|>, işaret biçimi
    assert (out_dir / "config.json").exists(), "model kaydedilmedi"
    LlamaForCausalLM.from_pretrained(out_dir)
    assert AutoTokenizer.from_pretrained(out_dir).chat_template, "şablon kaydedilmedi"
    gc = json.loads((out_dir / "generation_config.json").read_text())
    assert gc["eos_token_id"] in (IM_END, [IM_END]), gc["eos_token_id"]
    assert json.loads((out_dir / "_TRAINING_DONE").read_text())["format"] == SFT_FORMAT
    loss = (tmp / "results" / "E3" / "loss" / "T-run.csv").read_text().splitlines()
    print(f"ok  kayıt: model + şablonlu tokenizer + eos=<|im_end|> + işaret "
          f"({SFT_FORMAT}); kayıp satırı: {len(loss) - 1}")
    # 3) aynı biçim → atla; eski biçim işareti → yeniden eğit
    assert not train_once(), "aynı biçimde yeniden eğitildi (atlamalıydı)"
    (out_dir / "_TRAINING_DONE").write_text("2026-10-12 10:00:00")   # eski düz işaret
    assert train_once(), "eski biçimli model yeniden eğitilmedi"
    print("ok  aynı biçim atlandı; eski biçimli model otomatik yeniden eğitildi")
    print("SFT TESTİ GEÇTİ")
finally:
    sft.SFTDataset = _Orig
    shutil.rmtree(tmp, ignore_errors=True)
