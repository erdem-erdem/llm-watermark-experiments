#!/usr/bin/env bash
# H100 ortam hazırlığı — RUN_MANUAL.md Bölüm 1
# Kullanım: bash scripts/setup_env.sh  (veya VENV'e aktifken)
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install --upgrade pip
python -m pip install \
  "torch>=2.4" \
  "vllm>=0.6.4" \
  "transformers>=4.45" \
  "datasets>=2.20" \
  "accelerate>=1.1.0" \
  "sentence-transformers>=3.0" \
  "scipy>=1.13" \
  "matplotlib>=3.9" \
  "numpy>=1.26" \
  "pyyaml>=6.0" \
  "sentencepiece" \
  "tiktoken"

# Qwen3.5 lineer-dikkat çekirdekleri (yoksa yavaş + bellek-aç PyTorch yedek yolu).
# --no-deps: torch/vllm/transformers sürümlerine DOKUNMAZ.
python -m pip install --no-deps "flash-linear-attention" einops

# GPU kontrolü
python - <<'EOF'
import torch
n = torch.cuda.device_count()
print(f"CUDA cihaz sayısı: {n}")
assert n >= 1, "GPU görünmüyor — nvidia-smi ve sürücüleri kontrol edin"
for i in range(n):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

echo "ortam hazır"