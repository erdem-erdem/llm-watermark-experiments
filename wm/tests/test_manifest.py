"""run_manifest.json eşzamanlı yazma testi (CPU).

gpu_runner birden çok işi AYNI ANDA başlatır; her biri sonunda record_duration
→ update_manifest çağırır. Kilitsiz oku-değiştir-yaz bir işlemin dosyayı
KESTİĞİ anda öbürünün boş dosyayı okumasına ("JSONDecodeError: Expecting value:
line 1 column 1") ve güncelleme kayıplarına yol açıyordu.

  python -m wm.tests.test_manifest
"""

import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

N_PROC, N_EACH = 8, 40


def _worker(args):
    root, wid = args
    import wm.util as u
    u.RESULTS = Path(root)
    u.MANIFEST_PATH = Path(root) / "run_manifest.json"
    for i in range(N_EACH):
        u.update_manifest(**{f"w{wid}.k{i}": i})
        if i % 8 == 0:
            u.record_duration(f"w{wid}_{i}", 0.5)
    return True


def hammer():
    root = tempfile.mkdtemp()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(N_PROC) as pool:
        res = pool.map_async(_worker, [(root, w) for w in range(N_PROC)])
        res.get(timeout=300)           # işçi hatası burada yükselir
    data = json.loads((Path(root) / "run_manifest.json").read_text(encoding="utf-8"))
    missing = [(w, i) for w in range(N_PROC) for i in range(N_EACH)
               if data.get(f"w{w}", {}).get(f"k{i}") != i]
    assert not missing, f"{len(missing)}/{N_PROC * N_EACH} güncelleme KAYIP"
    n_dur = len(data.get("durations", {}))
    assert n_dur == N_PROC * (N_EACH // 8), f"süre kayıtları eksik: {n_dur}"
    leftovers = [p.name for p in Path(root).iterdir()
                 if p.name != "run_manifest.json" and not p.is_dir()]
    assert not leftovers, f"geçici dosya/kilit kaldı: {leftovers}"
    print(f"ok  {N_PROC} süreç × {N_EACH} yazım eşzamanlı: hiç kayıp yok, dosya geçerli, "
          f"artık dosya yok")


def corrupt_and_stale():
    import wm.util as u
    root = Path(tempfile.mkdtemp())
    u.RESULTS = root
    u.MANIFEST_PATH = root / "run_manifest.json"
    # 1) bozuk/boş dosya: yedeklenir, yeni manifest yazılır (koşu çökmez)
    u.MANIFEST_PATH.write_text("", encoding="utf-8")
    u.update_manifest(a=1)
    assert json.loads(u.MANIFEST_PATH.read_text(encoding="utf-8"))["a"] == 1
    assert list(root.glob("run_manifest.corrupt*")), "bozuk dosya yedeklenmedi"
    u.MANIFEST_PATH.write_text('{"x": 1', encoding="utf-8")      # yarım JSON
    u.update_manifest(b=2)
    assert json.loads(u.MANIFEST_PATH.read_text(encoding="utf-8"))["b"] == 2
    print("ok  boş/yarım manifest yedeklendi, koşu çökmeden devam etti")
    # 2) çöken sürecin bıraktığı ESKİ kilit: kısa sürede devralınır
    lock = u.MANIFEST_PATH.with_name(u.MANIFEST_PATH.name + ".lock")
    lock.write_text("x")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    t0 = time.time()
    u.update_manifest(c=3)
    assert time.time() - t0 < 10 and not lock.exists()
    print("ok  eski (çökmüş süreç) kilidi devralındı")


if __name__ == "__main__":
    hammer()
    corrupt_and_stale()
    print("BÜTÜN MANİFEST TESTLERİ GEÇTİ")
