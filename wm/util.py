"""Ortak yardımcılar: manifest güncelleme, LOG.md'ye komut yazma, tohumlar.

Her betik tek bir YAML yapılandırmasından çalışır; komutlar results/LOG.md'ye
eklenir (Bölüm 9). run_manifest.json'a tohumlar, anahtarlar, HF revizyonları,
kütüphane sürümleri ve gerçek süreler yazılır (kural 7).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOG_PATH = RESULTS / "LOG.md"
MANIFEST_PATH = RESULTS / "run_manifest.json"

SEED_GLOBAL = 20260814  # sabit proje tohumu (tarih anlamlı: AB YZ Yasası)


def ensure_results_dirs():
    for d in ["E0", "E1", "E2", "E3", "E4", "mini"]:
        (RESULTS / d).mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def log_cmd(argv: list[str], note: str = ""):
    """Çalıştırılan komutu LOG.md'ye ekle."""
    ensure_results_dirs()
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"- `{ts}` — `{' '.join(str(a) for a in argv)}`")
        if note:
            f.write(f" — {note}")
        f.write("\n")


def log_note(note: str):
    ensure_results_dirs()
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"- `{ts}` — {note}\n")


LOCK_STALE_S = 30.0       # bir güncelleme milisaniyeler sürer; bundan eski kilit = çökmüş süreç


@contextlib.contextmanager
def _manifest_lock(timeout: float = 120.0):
    """Süreçler arası kilit (O_EXCL; Linux/Windows/NFS'te çalışır).

    gpu_runner birden çok işi aynı anda koşturur ve hepsi manifest'e yazar.
    Kilitsiz oku-değiştir-yaz: biri dosyayı keserken öbürü BOŞ dosyayı okuyup
    JSONDecodeError ile çöküyor, ayrıca güncellemeler birbirini eziyordu.
    """
    lock = MANIFEST_PATH.with_name(MANIFEST_PATH.name + ".lock")
    t0 = time.time()
    while True:
        try:
            os.close(os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except (FileExistsError, PermissionError) as e:
            # Windows, silinmekte olan/açık dosya için PermissionError verir:
            # ikisi de "kilit dolu, tekrar dene" demektir.
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_S:
                    lock.unlink(missing_ok=True)       # çöken sürecin kilidi
                    continue
            except (FileNotFoundError, PermissionError):
                pass                                    # tam şimdi bırakıldı / silinmekte
            if time.time() - t0 > timeout:
                raise TimeoutError(
                    f"manifest kilidi {timeout:.0f}s alınamadı: {lock} ({e!r})")
            time.sleep(0.02)
    try:
        yield
    finally:
        for _ in range(100):                            # Windows: kısa süreli kilitli olabilir
            try:
                lock.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.02)


def _read_manifest() -> dict:
    """Manifest'i oku. Boş/bozuk dosya koşuyu ÇÖKERTMEZ: yedeklenir, yeniden başlanır."""
    if not MANIFEST_PATH.exists():
        return {}
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if text.strip() else None
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    bak = MANIFEST_PATH.with_name(
        f"run_manifest.corrupt.{int(time.time())}.{os.getpid()}.json")
    MANIFEST_PATH.replace(bak)
    print(f"UYARI: {MANIFEST_PATH.name} boş/bozuktu → {bak.name} olarak yedeklendi, "
          f"yeni manifest başlatıldı", file=sys.stderr)
    return {}


def update_manifest(**kwargs):
    """run_manifest.json'a alan ekle/güncelle (iç içe nokta-anahtar destekli).

    Süreçler arası güvenli: kilit altında oku-değiştir, geçici dosyaya yaz,
    os.replace ile ATOMİK değiştir (okuyan hiçbir zaman yarım dosya görmez).
    """
    ensure_results_dirs()
    with _manifest_lock():
        _update_manifest_locked(kwargs)


def _update_manifest_locked(kwargs):
    data = _read_manifest()
    for k, v in kwargs.items():
        if "." in k:
            cur = data
            parts = k.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = v
        else:
            data[k] = v
    data.setdefault("library_versions", {})
    data.setdefault("durations", {})
    tmp = MANIFEST_PATH.with_name(f".{MANIFEST_PATH.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MANIFEST_PATH)


def record_duration(step: str, seconds: float):
    update_manifest(**{f"durations.{step}_seconds": round(seconds, 1)})


def timed(step: str):
    """Dekoratör: fonksiyon süresini manifest'e yaz."""
    def deco(fn):
        def wrapper(*args, **kw):
            t0 = time.time()
            out = fn(*args, **kw)
            record_duration(step, time.time() - t0)
            return out
        return wrapper
    return deco


def torch_env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            info["gpu_names"] = [torch.cuda.get_device_name(i)
                                 for i in range(torch.cuda.device_count())]
    except Exception as e:  # torch yoksa
        info["torch_error"] = str(e)
    for mod in ["transformers", "vllm", "datasets", "sentence_transformers",
                "scipy", "numpy", "accelerate", "trl"]:
        try:
            m = __import__(mod)
            info.setdefault("library_versions", {})[mod] = getattr(m, "__version__", "?")
        except Exception:
            info.setdefault("library_versions", {})[mod] = None
    return info


def git_or_file_hash(path: Path) -> str | None:
    """HF dataset/model revizyonu için: dosya sha256 (yerel kopyalarda)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=None)
def hf_commit_hash(model_or_dataset: str) -> str | None:
    """HF hub revizyonu (commit hash). Ağ yoksa None.

    Önbellekli: her üretilen KAYIT için çağrılıyordu (koşu başına yüzlerce HF API
    isteği → IP başına 500/5dk sınırında 429). Yerel yol ya da çevrimdışı modda
    ağa hiç gidilmez.
    """
    if os.path.exists(model_or_dataset):
        return None
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        # çevrimdışı: revizyonu predownload.py'nin kaydından oku
        try:
            return json.loads((RESULTS / "hf_assets.json").read_text(
                encoding="utf-8"))[model_or_dataset]["revision"]
        except Exception:  # noqa: BLE001
            return None
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(model_or_dataset)
        return info.sha
    except Exception:
        return None


def parse_args(description: str) -> argparse.Namespace:
    """Ortak CLI: --config (YAML), --smoke, --force."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", type=str, default=None,
                    help="YAML yapılandırma dosyası (isteğe bağlı)")
    ap.add_argument("--smoke", action="store_true",
                    help="Küçük ölçekli doğrulama koşusu")
    ap.add_argument("--force", action="store_true",
                    help="Mevcut çıktıları yeniden üret")
    # Betiğe özgü seçiciler. Betikler değerlerini kendileri sys.argv'den okur
    # (gen_e1: --stage/--model, gen_e2_e4: --stage/--task, eval/detect: --only);
    # burada YALNIZCA argparse'ın bunları "tanınmayan argüman" diye reddetmemesi
    # için tanımlanırlar. Yazım hataları yine reddedilir.
    ap.add_argument("--stage", type=str, default=None,
                    help="aşama seçici (betiğe göre: sweep|full, e2|e4, E0,E2,E3,E4)")
    ap.add_argument("--model", type=str, default=None,
                    help="gen_e1: yalnız bu model boyutu (0.8B|2B|4B|9B)")
    ap.add_argument("--task", type=str, default=None,
                    help="gen_e2_e4: dgen|dclean|div|agent|a0")
    ap.add_argument("--shard", type=str, default=None,
                    help="gen_e1 --stage full: i/n — kayıtların i::n dilimi")
    ap.add_argument("--nshards", type=int, default=None,
                    help="gen_e1 --stage full_merge: parça sayısı")
    ap.add_argument("--refetch", action="store_true",
                    help="prep_data: arşivlenmiş havuzu yok say, kaynağı yeniden çek+etiketle")
    ap.add_argument("--only", type=str, default=None,
                    help="virgülle ayrılmış koşul/model etiketleri")
    return ap.parse_args()


def load_config(path: str) -> dict:
    """YAML yükle. WM_MODEL_OVERRIDES (JSON) ile, YAML'DA VAR OLAN anahtarlar
    değiştirilebilir — küçük GPU'da duman testi için model adlarını değiştirir
    (scripts/run_smoke_6gb.sh). Ayarlı değilse hiçbir etkisi yoktur."""
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), f"YAML sözlük olmalı: {path}"
    raw = os.environ.get("WM_MODEL_OVERRIDES")
    if raw:
        ov = json.loads(raw)
        hit = {k: v for k, v in ov.items() if k in cfg}
        cfg.update(hit)
        if hit:
            print(f"[WM_MODEL_OVERRIDES] {path}: {hit}")
    return cfg


def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_csv(rows, path: Path, fieldnames=None):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: Path) -> list[dict]:
    import csv
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_no_thinking(text: str, where: str):
    """Kural 6: çıktılarda <think> görürsen dur ve düzelt."""
    if "<think>" in text or "</think>" in text:
        raise RuntimeError(
            f"Thinking modu açık kalmış! ({where}) — kural 6: dur ve şablonu düzelt.")


def truncate_neglog10p(x: float) -> str:
    """CSV/md için sayısal yazım; ondalık ayırıcı nokta, kırpma yok (Bölüm 7)."""
    return repr(float(x))