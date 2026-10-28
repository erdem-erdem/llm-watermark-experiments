"""Basit GPU-kuyruk koşucu — iş listesi JSON'u alır, GPU'lara dağıtır.

İş formatı (jobs JSON):
  [{"cmd": "python -m train.sft --config train/configs/A0.yaml", "gpu": 0, "tp": 1}, ...]

- "gpu" verilmemişse sıradaki boş GPU'ya atanır.
- "tp": tensor parallel boyutu; tp > 1 ise iş o kadar GPU'yu tek seferde
  bloke eder (CUDA_VISIBLE_DEVICES=gpu:gpu+..) ve vLLM'e
  --tensor-parallel-size tp iletilmez — çağrılan betik kendi cfg'sinden alır.
- Her işin stdout/stderr'i results/logs/<slug>.log dosyasına yazılır.

Kullanım:
  python scripts/gpu_runner.py scripts/jobs_e3.json
  python scripts/gpu_runner.py scripts/jobs_e1.json --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"


def run_job(job, gpu, tp):
    n = int(job.get("tp", tp))
    gpus = ",".join(str(g) for g in range(gpu, gpu + n))
    env_gpu = {"CUDA_VISIBLE_DEVICES": gpus}
    import os
    env = {**os.environ, **env_gpu}
    cmd = job["cmd"]
    print(f"[runner] GPU {gpus}: {cmd}")
    # GPU kimliği + son argüman: aynı saniyede başlayan işler (ör. parçalar)
    # birbirinin günlüğünü EZMESİN
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", cmd.split()[-1])[-40:]
    log = LOGS / f"{time.strftime('%H%M%S')}_gpu{gpus.replace(',', '-')}_{slug}.log"
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, shell=True, cwd=ROOT, env=env,
                           stdout=f, stderr=subprocess.STDOUT)
    print(f"[runner] çıktı {p.returncode} → {log}")
    return p.returncode


def run_job_retry(job, gpu, tp):
    """Başarısızsa RETRIES kez daha dene (GPU belleğinin boşalması için bekleyerek)."""
    rc = run_job(job, gpu, tp)
    for k in range(RETRIES):
        if rc == 0:
            break
        print(f"[runner] yeniden deneme {k + 1}/{RETRIES} ({RETRY_WAIT:.0f} sn sonra): {job['cmd']}")
        time.sleep(RETRY_WAIT)
        rc = run_job(job, gpu, tp)
    return rc


RETRIES = 0
RETRY_WAIT = 30.0


def main():
    global RETRIES, RETRY_WAIT
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", help="iş listesi JSON")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--keep-going", action="store_true",
                    help="iş başarısız olsa da pipeline'ı durdurma (çıkış kodu 0); "
                         "başarısızlar yine de özetlenir")
    ap.add_argument("--retries", type=int, default=0,
                    help="başarısız işi bu kadar kez yeniden dene")
    ap.add_argument("--retry-wait", type=float, default=30.0,
                    help="denemeler arası bekleme (sn)")
    args = ap.parse_args()
    RETRIES, RETRY_WAIT = args.retries, args.retry_wait
    LOGS.mkdir(parents=True, exist_ok=True)

    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    free = [int(g) for g in args.gpus.split(",")]
    # basit model: işler GPU sırasına göre kuyruğa alınır, her GPU kendi
    # iş listesini sırayla çalıştırır.
    failed = []
    queues: dict[int, list] = {g: [] for g in free}
    i = 0
    for job in jobs:
        g = free[i % len(free)]
        n_tp = int(job.get("tp", 1))
        if n_tp > 1:
            # TP işleri ayrı: kuyruğa girmeden hemen koştur (bloke edici)
            if run_job_retry(job, g, n_tp) != 0:
                failed.append(job["cmd"])
        else:
            queues[g].append(job)
        i += 1

    # her GPU için sıralı koşturucu (paralel: GPU başına bir kanal)
    def worker(g, queue):
        for job in queue:
            rc = run_job_retry(job, g, 1)
            if rc != 0:
                print(f"[runner] HATA (gpu {g}): {job['cmd']}")
                failed.append(job["cmd"])
        print(f"[runner] GPU {g} kuyruğu bitti")

    import threading
    threads = [threading.Thread(target=worker, args=(g, queues[g])) for g in free]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("[runner] tüm kuyruklar tamam")
    if failed:
        # Varsayılan: başarısız iş varsa DUR: aksi halde sonraki aşamalar eksik/bozuk
        # çıktıyla koşar (ör. kaydedilmemiş model). Yeniden koşum biten işleri atlar.
        # --keep-going: özetle ama 0 ile çık, pipeline sürsün.
        verdict = "devam ediliyor (--keep-going)" if args.keep_going else "pipeline durduruluyor"
        print(f"[runner] {len(failed)} iş BAŞARISIZ — {verdict}:")
        for c in failed:
            print(f"   - {c}")
        print(f"[runner] günlükler: {LOGS}")
        sys.exit(0 if args.keep_going else 1)


if __name__ == "__main__":
    main()