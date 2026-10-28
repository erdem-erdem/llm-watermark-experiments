#!/usr/bin/env bash
# Tam duman (smoke) koşusu — her aşamanın minik versiyonu.
# Beklenen süre: ~30-60 dk (model indirmeleri dahil).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

echo "== 1. PRF/tüp birim testleri =="
python -m wm.tests.test_prf
python -m wm.tests.test_sampler_detect

echo "== 2. Veri hazırlama (küçük) =="
python -m data.prep_data --config configs/prep_data.yaml --smoke
python -m eval.overlap_check --smoke || true

echo "== 3. Anahtar seçimi (küçük) =="
python -m data.prep_keys --config configs/keys.yaml --smoke

echo "== 4. Tokenizer karşılaştırma (4 model + Base) =="
python scripts/check_tokenizers.py

echo "== 5. E1 tarama (küçük) =="
python -m gen.gen_e1 --config configs/e1_sweep.yaml --stage sweep --smoke
python -m analyze.e1_report --smoke

echo "== 6. E2 üretim (küçük) =="
python -m gen.gen_e2_e4 --config configs/e2.yaml --stage e2 --smoke || true

echo "== 7. E3 tek eğitim (küçük) =="
python -m train.sft --config train/configs/A-gen-10.yaml --smoke || true

echo "== 8. Tespit (küçük) =="
python -m analyze.detect_all --config configs/detect.yaml --smoke --stage E0,E2 || true

echo "== 9. Bir figür =="
python -m analyze.make_figures || true

echo "DUMAN KOŞUSU TAMAM — çıktılar results/ altında. Tam koşuya geçmeden önce"
echo "her aşamanın çıktısını kontrol edin (bkz. RUN_MANUAL.md Bölüm 2)."