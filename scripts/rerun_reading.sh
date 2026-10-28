#!/usr/bin/env bash
# Düzeltme koşumu (uzak makinede, TÜM ana koşu bittikten sonra).
#
# Neden:
#  1) A-*-10 ve A-*-50 eğitimlerinde ρ İKİ KEZ uygulanmıştı (alt küme zaten ρ'dur):
#     A-gen-10/A-post-10 gerçekte %1, A-gen-50/A-post-50 %25 watermark'lıydı.
#     Düzeltildi (train/sft.py, MIX_VERSION="subset-v2"); bu 4 model yeniden eğitilir.
#  2) E3 eğrileri serbest üretimden çıkarılmıştı; spesifikasyon okuma modu der.
#  3) E4 halka 4 yanlış veri üzerinde okunmuştu; halka 3 kontrolleri eksikti.
#
# Kullanım:   bash scripts/rerun_reading.sh 4        (GPUS=2 ... / GPU_IDS=0,2 ...)
#             KEEP_GOING=1 RETRIES=1 bash scripts/rerun_reading.sh 2   (hata olsa da sürdür / yeniden dene)
# Süre (4×H100, tahmini): yeniden eğitim ~12 dk, değerlendirme ~5 dk, okuma modu ~15-25 dk,
#   serbest üretim ~10 dk, olimpiyat ~15 dk, mini ~15 dk  (toplam ~1.5 saat)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
N_GPUS="${1:-${GPUS:-4}}"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((N_GPUS-1)))}"
export N_GPUS GPU_IDS
# Runner seçenekleri: KEEP_GOING=1 → başarısız iş pipeline'ı durdurmaz; RETRIES=n → n kez yeniden dene
RUNNER_OPTS="--retries ${RETRIES:-0}"
[ "${KEEP_GOING:-0}" = "1" ] && RUNNER_OPTS="$RUNNER_OPTS --keep-going"
echo "GPU sayısı: $N_GPUS  (list: $GPU_IDS)"
source scripts/hf_offline.sh

AFFECTED="A-gen-10 A-gen-50 A-post-10 A-post-50"

echo "== 0. Testler =="
python -m wm.tests.test_prf
python -m wm.tests.test_reading_scorer
python -m wm.tests.test_boxed_parser
python -m train.selftest

echo "== 1. Eski (ρ-hatalı) çıktıları kenara al =="
STALE="results/_stale_rho_bug"
mkdir -p "$STALE"
for m in $AFFECTED; do
  for f in "results/eval_parts/gpqa/$m.json" "results/eval_parts/eval_loss/$m.json" \
           "results/E3/detect_curves_parts/$m.csv" "results/E3/loss/$m.csv" \
           results/E3/gen_outputs/${m}_n*_nowm.jsonl; do
    [ -e "$f" ] && mv "$f" "$STALE/" || true
  done
done
ls "$STALE" || true

echo "== 2. Yeniden eğitim: $AFFECTED (her biri bir GPU) =="
python - <<'PY'
import json
names = "A-gen-10 A-gen-50 A-post-10 A-post-50".split()
json.dump([{"cmd": f"python -m train.sft --config train/configs/{x}.yaml"} for x in names],
          open("scripts/_jobs_retrain.json", "w"))
PY
python scripts/gpu_runner.py scripts/_jobs_retrain.json --gpus "$GPU_IDS" $RUNNER_OPTS
rm -f scripts/_jobs_retrain.json

echo "== 3. Yeniden eğitilen 4 model için GPQA-TR + eval_loss =="
python - <<'PY'
import json
names = "A-gen-10 A-gen-50 A-post-10 A-post-50".split()
jobs = [{"cmd": f"python -m eval.{s} --config configs/eval.yaml --only {n}"}
        for s in ("gpqa_tr", "eval_loss") for n in names]
json.dump(jobs, open("scripts/_jobs_reeval.json", "w"))
PY
python scripts/gpu_runner.py scripts/_jobs_reeval.json --gpus "$GPU_IDS" $RUNNER_OPTS
rm -f scripts/_jobs_reeval.json

echo "== 4. Okuma modu (11 model: 8 A + 3 B; modeller GPU'lara bölünür) + halka 3 (CPU) =="
python -m analyze.reading_rerun --stage text &
TEXT_PID=$!
python - "$N_GPUS" <<'PY'
import json, sys
n = int(sys.argv[1])
names = ["A0", "A-gen-10", "A-gen-50", "A-gen-100", "A-post-10", "A-post-50",
         "A-post-100", "A-rew", "B-greedy", "B-sample", "B-control"]
jobs = [{"cmd": "python -m analyze.reading_rerun --stage model --only " + ",".join(names[i::n])}
        for i in range(min(n, len(names)))]
json.dump(jobs, open("scripts/_jobs_reading.json", "w"))
print(f"{len(names)} model → {len(jobs)} GPU")
PY
python scripts/gpu_runner.py scripts/_jobs_reading.json --gpus "$GPU_IDS" $RUNNER_OPTS
rm -f scripts/_jobs_reading.json
wait $TEXT_PID

echo "== 5. Birleştir: E3/reading_*.csv, E0/pvals_parts_all.csv, E4/chain.csv =="
python -m analyze.reading_rerun --stage merge

echo "== 6. (Tamamlayıcı) Serbest-üretim modu: yeniden eğitilen 4 modelin çıktıları =="
python - "$N_GPUS" <<'PY'
import json, sys
names = "A-gen-10 A-gen-50 A-post-10 A-post-50".split()
jobs = [{"cmd": "python -m analyze.detect_all --config configs/detect.yaml --stage E3 --only " + n}
        for n in names]
json.dump(jobs, open("scripts/_jobs_gendet.json", "w"))
PY
python scripts/gpu_runner.py scripts/_jobs_gendet.json --gpus "$GPU_IDS" $RUNNER_OPTS
rm -f scripts/_jobs_gendet.json

echo "== 6b. Olimpiyat-TR (kalite kontrolü) yeniden: \\boxed{} talimatı + ayrıştırıcı düzeltmesi =="
# Eski koşuda %40 'boxed yok' vardı: iç içe parantezli kutular (\frac, \text) yanlış sayılıyordu
# ve talimat problemin ÖNÜNDEYDİ. Yeni talimat sonda + örnek biçim; üretilen cevapların
# kuyruğu ve kesilme bilgisi CSV'ye yazılır. 8 koşulun HEPSİ aynı talimatla yenilenir.
mkdir -p "$STALE/aime_parts"
mv results/E2/aime_parts/*.json "$STALE/aime_parts/" 2>/dev/null || true
mv results/E2/aime_per_problem.csv results/E2/aime_summary.json "$STALE/" 2>/dev/null || true
python - "$N_GPUS" <<'PY'
import json, sys
n = int(sys.argv[1])
conds = ["main", "wrong1", "wrong2", "wrong3", "seed0", "seed1", "seed2", "seed3"]
jobs = [{"cmd": "python -m eval.aime_tr --config configs/eval.yaml --only "
                + ",".join(conds[i::n])} for i in range(min(n, len(conds)))]
json.dump(jobs, open("scripts/_jobs_aime.json", "w"))
PY
python scripts/gpu_runner.py scripts/_jobs_aime.json --gpus "$GPU_IDS" $RUNNER_OPTS
rm -f scripts/_jobs_aime.json

echo "== 6c. Mini denemeler (GPU): açılış (söz+paragraf) ve benchmark işaretleme =="
# temperature=0 hatası düzeltildi (T=1.0): eski koşuda tek-sıcak dağılım izi sıfırlıyordu.
python - <<'PY'
import json
jobs = [{"cmd": "python -m gen.gen_opening --config configs/mini.yaml"},
        {"cmd": "python -m gen.gen_mini --config configs/mini.yaml"}]
json.dump(jobs, open("scripts/_jobs_mini.json", "w"))
PY
python scripts/gpu_runner.py scripts/_jobs_mini.json --gpus "$GPU_IDS" $RUNNER_OPTS
rm -f scripts/_jobs_mini.json

echo "== 7. CPU tamamlamaları (E0/E1/E2 + mini denemeler veriden) =="
python -m analyze.e0_complete
python -m analyze.e1_complete
python -m analyze.e2_complete
python -m analyze.mini_from_data

echo "DÜZELTME KOŞUMU BİTTİ"
echo "Aktarılacak: results/ (özellikle E0/ E1/ E2/ E3/ E4/, eval_parts/, run_manifest.json, logs/)"
