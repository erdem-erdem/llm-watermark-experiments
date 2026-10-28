#!/usr/bin/env bash
# Küçük GPU (≈6 GB VRAM) duman testi — TÜM kod yollarını, küçük modelle sınar.
#
#   bash scripts/run_smoke_6gb.sh
#
# Ne değişir (yalnızca bu betikte, üretim koşusuna dokunmaz):
#   * 9B öğretmen, 2B-Base öğrenci ve A₀/ajan kontrol noktaları → Qwen3.5-0.8B
#     (tokenizer 9B'ninkiyle AYNI kalır; id'ler gerçek koşuyla uyumludur).
#   * vLLM: VRAM oranı 0.80, bağlam 4096, CUDA graph kapalı (WM_EAGER=1).
#   * Eğitim: Adafactor + batch 1 + gradient checkpointing (WM_TRAIN_LOWMEM=1).
# SINANMAYANLAR (bellek/veri yüzünden): 9B/2B boyut davranışı, 4 GPU paralelliği,
# AIME/GPQA (gated veri + 4096 token üretim), E4 B-eğitimleri. Bunlar H100'de.
#
# GEREKSİNİM: Ampere veya yeni GPU (compute capability >= 8.0; bf16 için).
# GTX 16xx / RTX 20xx / T4 bf16 desteklemez → bu kod yolları çalışmaz.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

# --- yalnızca ilk GPU, küçük-GPU ayarları (dışarıdan değiştirilebilir) -------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WM_GPU_MEM_UTIL="${WM_GPU_MEM_UTIL:-0.80}"   # masaüstü VRAM kullanıyorsa 0.70
export WM_MAX_MODEL_LEN="${WM_MAX_MODEL_LEN:-4096}"
export WM_EAGER="${WM_EAGER:-1}"
export WM_TRAIN_LOWMEM=1
SMALL="Qwen/Qwen3.5-0.8B"
SMOKE_CKPT="results/E3/models/A-gen-10"
export WM_MODEL_OVERRIDES="{\"teacher_model\":\"$SMALL\",\"base_model\":\"$SMALL\",\"student_model\":\"$SMALL\",\"model\":\"$SMALL\",\"a0_model\":\"$SMOKE_CKPT\",\"agent_checkpoint\":\"$SMOKE_CKPT\",\"a0_checkpoint\":\"$SMOKE_CKPT\"}"

echo "== 0. GPU denetimi =="
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "GPU görünmüyor"
p = torch.cuda.get_device_properties(0)
cc = (p.major, p.minor)
print(f"{p.name}: {p.total_memory/2**30:.1f} GiB, compute capability {cc[0]}.{cc[1]}")
assert cc >= (8, 0), "bf16 için Ampere+ gerekli (compute capability >= 8.0)"
x = torch.ones(4, device="cuda", dtype=torch.bfloat16) * 2
assert float(x.sum()) == 8.0
print("bf16 CUDA işlemi OK")
EOF

echo "== 1. Birim testleri (CPU) =="
python -m wm.tests.test_prf
python -m wm.tests.test_sampler_detect
python -m wm.tests.test_vllm_adapter
python -m wm.tests.test_think_ban

echo "== 2. Veri hazırlama (küçük; etiketleme 0.8B ile) =="
python -m data.prep_data --config configs/prep_data.yaml --smoke
python -m eval.overlap_check --smoke

echo "== 3. Anahtar seçimi =="
python -m data.prep_keys --config configs/keys.yaml --smoke

echo "== 4. Tokenizer karşılaştırma =="
python scripts/check_tokenizers.py

echo "== 5. E1 tarama (20 prompt, 0.8B, T=0.7; watermark LogitsProcessor sınanır) =="
python -m gen.gen_e1 --config configs/e1_sweep.yaml --stage sweep --smoke
python -m analyze.e1_report --smoke

echo "== 6. E2 üretim (D_gen / D_clean / çeşitlilik) =="
python -m gen.gen_e2_e4 --config configs/e2.yaml --stage e2 --smoke

echo "== 7. E3 tek eğitim (0.8B, 1 epoch, 50 satır, düşük bellek) =="
python -m train.sft --config train/configs/A-gen-10.yaml --smoke --force

echo "== 8. E4 üretimi (ajan + A₀ kontrol, eğitilen küçük model) =="
python -m gen.gen_e2_e4 --config configs/e2.yaml --task agent --smoke
python -m gen.gen_e2_e4 --config configs/e2.yaml --task a0 --smoke

echo "== 9. Tespit: E0 (okuma modu) + E2 + E3(A-gen-10) =="
python -m analyze.detect_all --config configs/detect.yaml --smoke --stage E0,E2
python -m analyze.detect_all --config configs/detect.yaml --smoke --stage E3 --only A-gen-10

echo "== 10. eval_loss (A-gen-10) + figürler =="
python -m eval.eval_loss --config configs/eval.yaml --smoke --only A-gen-10 || echo "ATLANDI: eval_loss (isteğe bağlı)"
python -m analyze.figures || echo "ATLANDI: bazı figürler eksik veriyle üretilemedi (smoke'ta beklenir)"

echo
echo "6GB DUMAN KOŞUSU TAMAM. Kontrol: results/E1/sweep.csv, data/D_gen.jsonl,"
echo "results/E3/models/A-gen-10, results/E0/*.csv, results/E3/detect_curves.csv"
