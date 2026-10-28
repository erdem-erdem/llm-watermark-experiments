# syntax=docker/dockerfile:1.7
# llm-watermark-experiments — tek imaj: vLLM çıkarım + HF Trainer eğitim + analiz.
# Conda YOK. Sistem python'u (Ubuntu 22.04 = python 3.10, proje ortamıyla aynı)
# + /opt/venv. Konteyner içindeki tek dış bağımlılık: ana makinenin NVIDIA sürücüsü.
#
# Derleme (proje kökünden):
#   docker build -f docker/Dockerfile -t llm-watermark-experiments:latest .
# Ayrıntı: DOCKER_GUIDE.md

# nvcc'li "devel" imajı: vLLM/flashinfer JIT ve triton derlemeleri için gerekir.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/hf \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    TRITON_CACHE_DIR=/tmp/.triton \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential ninja-build git curl ca-certificates libnuma1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv /opt/venv \
    && pip install -U pip wheel setuptools

# --- Python bağımlılıkları -------------------------------------------------
# Varsayılan: çalışan makinede üretilmiş KİLİT dosyası (docker/requirements.lock.txt).
# Kilit yoksa gevşek liste: --build-arg REQ_FILE=requirements.txt
ARG REQ_FILE=requirements.lock.txt
# Kilit torch'u "+cu128" gibi yerel etiketle içeriyorsa:
#   --build-arg EXTRA_INDEX=https://download.pytorch.org/whl/cu128
ARG EXTRA_INDEX=
COPY docker/${REQ_FILE} /tmp/requirements.txt
RUN pip install ${EXTRA_INDEX:+--extra-index-url ${EXTRA_INDEX}} -r /tmp/requirements.txt

# Qwen3.5'in lineer-dikkat katmanları için hızlı çekirdekler (yoksa HF yavaş
# torch yoluna düşer — doğru ama yavaş). --no-deps: torch/vllm sürümlerine dokunma.
RUN pip install --no-deps flash-linear-attention einops \
    || echo "UYARI: flash-linear-attention kurulamadı; eğitim yavaş yoldan çalışır"

# İsteğe bağlı (derleme 10-20 dk): --build-arg INSTALL_CAUSAL_CONV1D=1
ARG INSTALL_CAUSAL_CONV1D=0
RUN if [ "$INSTALL_CAUSAL_CONV1D" = "1" ]; then \
      pip install --no-build-isolation causal-conv1d; fi

COPY docker/selfcheck.py /opt/selfcheck.py

# Kod ve çıktılar imaja GÖMÜLMEZ: proje klasörü /workspace'e bağlanır
# (data/ klasörü hem kod hem çıktı barındırır; bind-mount en sağlam yoldur).
WORKDIR /workspace
CMD ["bash"]
