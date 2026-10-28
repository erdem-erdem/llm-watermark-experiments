#!/usr/bin/env bash
# E1 kapısı sonrası tam koşum.
# GPU sayısı: ilk argüman ya da GPUS (varsayılan 4).
# Ağır aşamalar (üretim/örgütim/değerlendirme) verilen GPU sayısına bölünür.
#   bash scripts/run_e1_full.sh 4   |  GPUS=2 bash scripts/run_e1_full.sh
#   GPU_IDS=0,2 bash scripts/run_e1_full.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
N_GPUS="${1:-${GPUS:-4}}"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((N_GPUS-1)))}"
export N_GPUS GPU_IDS
echo "GPU sayısı: $N_GPUS  (list: $GPU_IDS)"
source scripts/hf_offline.sh

if [ ! -f results/E1/selected.json ]; then
  echo "HATA: results/E1/selected.json yok — önce E1 kapısını onaylayın."
  exit 1
fi
python - <<'EOF'
import json, sys
sel = json.load(open("results/E1/selected.json", encoding="utf-8"))
if not sel.get("approved"):
    sys.exit("HATA: selected.json'da approved=true olmalı")
print(f"E1 onaylı: {sel}")
EOF

echo "== 4. E2 üretimi (dgen / dclean / çeşitlilik; GPU'ya bölünür) =="
python - "$N_GPUS" <<'EOF'
import json, sys
n = int(sys.argv[1])
tasks = ["dgen", "dclean", "div"]
jobs = [{"cmd": f"python -m gen.gen_e2_e4 --config configs/e2.yaml --task {t}"}
        for t in tasks]
json.dump(jobs, open("scripts/_jobs_e2.json", "w"))
print(f"{len(jobs)} iş → {min(n, len(jobs))} GPU")
EOF
python scripts/gpu_runner.py scripts/_jobs_e2.json --gpus "$GPU_IDS"
rm -f scripts/_jobs_e2.json

echo "== 5. E1 full (D_post, D_rew; kayıtlar $N_GPUS GPU'ya bölünür) =="
python - "$N_GPUS" <<'EOF'
import json, sys
n = int(sys.argv[1])
jobs = [{"cmd": "python -m gen.gen_e1 --config configs/e1_sweep.yaml "
                f"--stage full --shard {i}/{n}"} for i in range(n)]
json.dump(jobs, open("scripts/_jobs_e1full.json", "w"))
print(f"  train kayıtları {n} parçaya bölündü → {n} GPU (parçalar: results/E1/full_parts/)")
EOF
python scripts/gpu_runner.py scripts/_jobs_e1full.json --gpus "$GPU_IDS"
rm -f scripts/_jobs_e1full.json
# eksik parça varsa burada HATA ile durur; yeniden koşum yalnız eksik parçaları üretir
python -m gen.gen_e1 --config configs/e1_sweep.yaml --stage full_merge --nshards "$N_GPUS"

echo "== 6a. Eğitim ön-testi (minik model, saniyeler) =="
python -m train.selftest

echo "== 6. E3 eğitimleri ($N_GPUS GPU, dalga başına $N_GPUS iş) =="
python - "$N_GPUS" <<'EOF'
import json, sys
n = int(sys.argv[1])
names = ["A0", "A-gen-100", "A-post-100", "A-rew",
         "A-gen-10", "A-gen-50", "A-post-10", "A-post-50"]
for w, chunk in enumerate([names[i:i+n] for i in range(0, len(names), n)]):
    jobs = [{"cmd": f"python -m train.sft --config train/configs/{x}.yaml"}
            for x in chunk]
    json.dump(jobs, open(f"scripts/_jobs_e3_w{w}.json", "w"))
    print(f"  dalga {w+1}: {chunk}")
EOF
for jf in scripts/_jobs_e3_w*.json; do
  python scripts/gpu_runner.py "$jf" --gpus "$GPU_IDS"
done
rm -f scripts/_jobs_e3_w*.json

echo "== 7. E0 (A₀ kalibrasyon + yanlış anahtarlar) =="
python -m analyze.detect_all --config configs/detect.yaml --stage E0

echo "== 8. E2 tespit =="
python -m analyze.detect_all --config configs/detect.yaml --stage E2

echo "== 9. E3 çıktı üretimi + tespit eğrileri (8 koşul → GPU'lara bölünür) =="
python - "$N_GPUS" <<'EOF'
import json, sys
import yaml
n = int(sys.argv[1])
conds = yaml.safe_load(open("configs/detect.yaml",
                            encoding="utf-8"))["a_models"]
jobs = []
for i in range(min(n, len(conds))):
    chunk = conds[i::n]        # koşul listesini GPU'lara böl
    jobs.append({"cmd": "python -m analyze.detect_all "
                          "--config configs/detect.yaml --stage E3 "
                          f"--only {','.join(chunk)}"})
json.dump(jobs, open("scripts/_jobs_e3det.json", "w"))
print(f"  {len(conds)} koşul → {min(n, len(conds))} GPU "
      f"(parçalar detect_curves_parts/ altında birleşir)")
EOF
python scripts/gpu_runner.py scripts/_jobs_e3det.json --gpus "$GPU_IDS"
rm -f scripts/_jobs_e3det.json

echo "== 10. E4 ajan zinciri =="
python - "$N_GPUS" <<'EOF'
import json, sys
n = int(sys.argv[1])
jobs = [{"cmd": f"python -m gen.gen_e2_e4 --config configs/e2.yaml --task {t}"}
        for t in ("agent", "a0")]
json.dump(jobs, open("scripts/_jobs_e4.json", "w"))
print(f"{len(jobs)} iş → {min(n, len(jobs))} GPU")
EOF
python scripts/gpu_runner.py scripts/_jobs_e4.json --gpus "$GPU_IDS"
rm -f scripts/_jobs_e4.json

echo "== 10b. E4 öğrenci (B) eğitimleri: B-greedy, B-sample, B-control =="
# E4 tespiti results/E4/models/B-* modellerini okur; bunları eğiten başka adım YOK.
# Girdiler: data/agent_greedy.jsonl, agent_sampled.jsonl, a0_greedy.jsonl (adım 10).
python - <<'EOF'
import json
jobs = [{"cmd": f"python -m train.sft --config train/configs/{x}.yaml"}
        for x in ("B-greedy", "B-sample", "B-control")]
json.dump(jobs, open("scripts/_jobs_e4train.json", "w"))
print(f"  {len(jobs)} eğitim (her biri bir GPU'da paralel)")
EOF
python scripts/gpu_runner.py scripts/_jobs_e4train.json --gpus "$GPU_IDS"
rm -f scripts/_jobs_e4train.json

echo "== 10c. E4 tespit (4 halka) =="
python -m analyze.detect_all --config configs/detect.yaml --stage E4

echo "== 11. Değerlendirmeler =="
echo "  Olimpiyat (8 koşul → GPU'lara bölünür; parçalar aime_parts/ altında birleşir):"
python - "$N_GPUS" <<'EOF'
import json, sys
n = int(sys.argv[1])
conds = ["main", "wrong1", "wrong2", "wrong3",
         "seed0", "seed1", "seed2", "seed3"]
jobs = []
for i in range(min(n, len(conds))):
    chunk = conds[i::n]
    jobs.append({"cmd": "python -m eval.aime_tr "
                        f"--config configs/eval.yaml --only {','.join(chunk)}"})
json.dump(jobs, open("scripts/_jobs_aime.json", "w"))
print(f"  {len(conds)} koşul → {min(n, len(conds))} GPU")
EOF
python scripts/gpu_runner.py scripts/_jobs_aime.json --gpus "$GPU_IDS"
rm -f scripts/_jobs_aime.json
echo "  GPQA + eval_loss (model başına bir GPU'ya bölünür):"
python - "$N_GPUS" <<'EOF'
import json, sys
import yaml
n = int(sys.argv[1])
labels = [m["label"] for m in yaml.safe_load(open("configs/eval.yaml",
                                                  encoding="utf-8"))["models"]]
jobs = []
for stage in ("gpqa_tr", "eval_loss"):
    for i in range(min(n, len(labels))):
        chunk = labels[i::n]        # model listesini GPU'lara böl
        jobs.append({"cmd": f"python -m eval.{stage} "
                            f"--config configs/eval.yaml --only {','.join(chunk)}"})
json.dump(jobs, open("scripts/_jobs_eval.json", "w"))
print(f"  {len(labels)} model × 2 ölçüt → {min(n, len(labels))} GPU")
EOF
python scripts/gpu_runner.py scripts/_jobs_eval.json --gpus "$GPU_IDS"
rm -f scripts/_jobs_eval.json

echo "== 12. Mini denemeler (küçük, tek GPU) =="
python -m gen.gen_opening --config configs/mini.yaml
python -m gen.gen_mini --config configs/mini.yaml

echo "== 13. Figürler, tablolar, özet =="
python -m analyze.blog_examples
python -m analyze.blog_figures
python -m analyze.blog_figures_b
python -m analyze.tables
python -m analyze.summary

echo "TAM KOŞUM BİTTİ → results/SUMMARY.md, figures/"