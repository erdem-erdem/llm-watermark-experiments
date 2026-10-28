#!/usr/bin/env bash
# Tam deney koşusu — E1 kapısına kadar.
# GPU sayısı: ilk argüman ya da GPUS ortam değişkeni (varsayılan 4).
#   bash scripts/run_all.sh 4      |  GPUS=2 bash scripts/run_all.sh
#   GPU_IDS=0,2 bash scripts/run_all.sh   # bitişik olmayan GPU seçimi
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
N_GPUS="${1:-${GPUS:-4}}"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((N_GPUS-1)))}"
export N_GPUS GPU_IDS
echo "GPU sayısı: $N_GPUS  (list: $GPU_IDS)"

echo "== 0. Testler =="
python -m wm.tests.test_prf
python -m wm.tests.test_sampler_detect
python -m wm.tests.test_vllm_adapter
python -m wm.tests.test_think_ban
python -m wm.tests.test_manifest   # paralel işlerin manifest'e eşzamanlı yazımı
python -m train.selftest      # eğitim yolunu CPU'da saniyeler içinde dener

echo "== 0b. HF önbelleği (bir kerelik ön-indirme; tamamsa ağsız doğrulanır) =="
python scripts/predownload.py

echo "== 1. Veri (yalnızca ilk seferde; sonra atlanır) =="
python -m data.prep_data --config configs/prep_data.yaml
python -m eval.overlap_check

# Veri hazırlığı (InstrucTurca akışı) ağ gerektirebilir; bundan SONRASI için, önbellek
# tamamsa, Hub'a hiç istek atılmaz (429 yok). Kapat: HF_OFFLINE_MODE=off
source scripts/hf_offline.sh

echo "== 2. Anahtarlar (E0 kalibrasyon verisi) =="
python -m data.prep_keys --config configs/keys.yaml

echo "== 3. E1 tarama (4 model x 3T x 300 prompt; model başına bir GPU) =="
# min(N_GPUS, 4) paralel iş; kalan modeller kuyruğa girer
python - "$N_GPUS" <<'EOF'
import json, sys
n = int(sys.argv[1])
models = ["0.8B", "2B", "4B", "9B"]
jobs = [{"cmd": f"python -m gen.gen_e1 --config configs/e1_sweep.yaml "
                f"--stage sweep --model {m}"} for m in models]
json.dump(jobs, open("scripts/_jobs_e1.json", "w"))
print(f"{len(jobs)} iş, {min(n, len(jobs))} GPU'da paralel")
EOF
python scripts/gpu_runner.py scripts/_jobs_e1.json --gpus "$GPU_IDS"
rm -f scripts/_jobs_e1.json
python -m analyze.e1_report

echo "== E1 SEÇİMİ (otomatik) =="
cat results/E1/GATE.md
# Varsayılan: elle inceleme YOK — seçim e1_report tarafından yazıldı, akış sürer.
# Eski davranış (elle onay için dur): E1_GATE=manual bash scripts/run_all.sh
if [ "${E1_GATE:-auto}" = "manual" ]; then
  echo "E1_GATE=manual: durdu. results/E1/selected.json'u düzenleyin, sonra:"
  echo "  bash scripts/run_e1_full.sh $N_GPUS"
  exit 0
fi
bash scripts/run_e1_full.sh "$N_GPUS"