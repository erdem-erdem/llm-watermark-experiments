"""Bir kerelik HF ön-indirme — bütün modeller + veri setleri önbelleğe alınır.

  python scripts/predownload.py [--force] [--no-datasets]

NEDEN: vLLM / transformers, yerel önbellek dolu olsa bile her başlatmada Hub'a
"bu repoda hangi dosyalar var?" diye sorar. Paralel işler + çok sayıda motor
başlatma bunu IP/hesap başına 5 dakikalık API sınırına (429) taşır. Önbellek
TAM olduğunda koşum betikleri HF_HUB_OFFLINE=1 ile çalışır → Hub'a sıfır istek.

NE YAPAR
  * Modelleri configs/*.yaml, train/configs/*.yaml, gen_e1.SWEEP_MODELS ve
    scripts/check_tokenizers.py'den TÜRETİR (elle liste yok; unutma riski yok).
  * Önce AĞSIZ doğrular (her dosya gerçekten diskte mi?); yalnızca eksikleri indirir.
    Yarım kalmış indirmeyi "var" saymaz.
  * 429'da Retry-After kadar bekleyip yeniden dener.
  * Veri setlerini (olimpiyat, GPQA-TR) pipeline'ın kullandığı AYNI çağrılarla ısıtır.
  * Her şey tamamsa results/.hf_offline_ok yazar (betikler çevrimdışı moda yalnızca
    bu dosya varsa geçer); eksik varsa dosyayı SİLER → betikler çevrimiçi kalır.
  * Revizyonları (commit hash) results/hf_assets.json ve manifest'e yazar
    (çevrimdışı modda Hub'dan sorulamaz; kayıtlar buradan okur).

Model eksikse çıkış kodu 1 (pipeline'ın devam etmesi anlamsız). Veri seti eksikse
(ör. GPQA'nın kapılı olması) UYARI verir, çıkış kodu 0; yalnızca çevrimdışı işaret
yazılmaz.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

RESULTS = ROOT / "results"
MARK = RESULTS / ".hf_offline_ok"
ASSETS = RESULTS / "hf_assets.json"

# YAML'da model adı taşıyan anahtarlar (veri seti anahtarları BİLEREK yok)
MODEL_KEYS = {"tokenizer_model", "teacher_model", "student_model", "similarity_model",
              "base_model", "model"}
HF_ID = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")
LOCAL_PREFIX = ("results/", "data/", "train/", "configs/", "scripts/")

# Ağır/yinelenen biçimler (ONNX, TF, Flax, OpenVINO, GGUF …) indirilmez
IGNORE = ["*.onnx", "onnx/*", "openvino/*", "*.msgpack", "*.h5", "*.ot", "*.tflite",
          "*.gguf", "original/*", "*.pt", "*.ckpt"]


def log(msg):
    print(f"[predownload] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Neyin gerektiğini bul
# ---------------------------------------------------------------------------

def _hf_id(v):
    return (isinstance(v, str) and HF_ID.match(v) and not v.startswith(LOCAL_PREFIX)
            and not os.path.exists(v))


def collect_models() -> list[str]:
    import yaml
    ids = set()
    for yml in sorted(list((ROOT / "configs").glob("*.yaml"))
                      + list((ROOT / "train" / "configs").glob("*.yaml"))):
        cfg = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        for k, v in cfg.items():
            if k in MODEL_KEYS and _hf_id(v):
                ids.add(v)
            if k == "models" and isinstance(v, list):
                for m in v:
                    if isinstance(m, dict) and _hf_id(m.get("path")):
                        ids.add(m["path"])
    try:
        from gen.gen_e1 import SWEEP_MODELS
        ids.update(SWEEP_MODELS)
    except Exception as e:  # noqa: BLE001
        log(f"UYARI: gen_e1.SWEEP_MODELS okunamadı ({e}); regex ile aranıyor")
        txt = (ROOT / "gen" / "gen_e1.py").read_text(encoding="utf-8")
        ids.update(re.findall(r'"(Qwen/[\w.\-]+)"', txt.split("SWEEP_T")[0]))
    ct = ROOT / "scripts" / "check_tokenizers.py"
    if ct.exists():
        block = ct.read_text(encoding="utf-8").split("MODELS = [", 1)[-1].split("]", 1)[0]
        ids.update(re.findall(r'"([\w.\-]+/[\w.\-]+)"', block))
    return sorted(ids)


# ---------------------------------------------------------------------------
# Modeller
# ---------------------------------------------------------------------------

def expected_files(api, repo) -> tuple[list[str], list[str]]:
    """(beklenen dosyalar, ignore kalıpları). safetensors varsa .bin indirilmez."""
    from huggingface_hub.utils import filter_repo_objects
    files = list(api.list_repo_files(repo))
    ign = list(IGNORE)
    if any(f.endswith(".safetensors") for f in files):
        ign += ["*.bin", "pytorch_model*", "tf_model*", "flax_model*"]
    keep = list(filter_repo_objects(items=files, ignore_patterns=ign))
    return keep, ign


def with_retry(fn, what, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                m = re.search(r"[Rr]etry after (\d+)", msg)
                wait = int(m.group(1)) + 5 if m else 60
                log(f"429 ({what}): {wait}s bekleniyor ({i + 1}/{tries})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"{what}: {tries} denemede 429 aşılamadı")


def cached_ok(repo, files) -> str | None:
    """AĞSIZ doğrulama: tüm beklenen dosyalar diskte mi? Evetse snapshot yolu."""
    from huggingface_hub import snapshot_download
    try:
        d = Path(snapshot_download(repo, local_files_only=True))
    except Exception:  # noqa: BLE001
        return None
    return str(d) if all((d / f).exists() for f in files) else None


def ensure_model(repo, known, force):
    """Dönüş: {'revision','files'} — tamamsa; aksi halde hata."""
    from huggingface_hub import HfApi, snapshot_download
    if not force and repo in known:
        d = cached_ok(repo, known[repo]["files"])
        if d:
            log(f"  var (ağsız doğrulandı): {repo}")
            return known[repo]
    api = HfApi()
    files, ign = with_retry(lambda: expected_files(api, repo), f"{repo} dosya listesi")
    log(f"  indiriliyor: {repo} ({len(files)} dosya)")
    path = with_retry(lambda: snapshot_download(repo, ignore_patterns=ign),
                      f"{repo} indirme")
    d = Path(path)
    missing = [f for f in files if not (d / f).exists()]
    if missing:
        raise RuntimeError(f"{repo}: indirme sonrası eksik dosyalar: {missing[:5]}")
    return {"revision": d.name, "files": files}


# ---------------------------------------------------------------------------
# Veri setleri (pipeline'ın kullandığı AYNI çağrılar → aynı önbellek anahtarı)
# ---------------------------------------------------------------------------

def _dataset_calls():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
    return [
        ("olimpiyat", lambda **kw: _load(cfg["aime_dataset"], None,
                                         split=cfg.get("aime_split", "ortaokul_matematik"),
                                         **kw)),
        ("GPQA-TR", lambda **kw: _load(cfg["gpqa_dataset"], None,
                                       split=cfg.get("gpqa_split", "train"), **kw)),
    ]


def _load(name, config, **kw):
    from datasets import load_dataset
    return load_dataset(name, config, **kw) if config else load_dataset(name, **kw)


def ensure_datasets() -> bool:
    from datasets import DownloadConfig
    ok = True
    for label, call in _dataset_calls():
        try:
            call(download_config=DownloadConfig(local_files_only=True))
            log(f"  var (ağsız doğrulandı): veri seti {label}")
            continue
        except Exception:  # noqa: BLE001
            pass
        try:
            ds = with_retry(call, f"veri seti {label}")
            log(f"  indirildi: veri seti {label} ({len(ds)} satır)")
        except Exception as e:  # noqa: BLE001
            ok = False
            log(f"UYARI: veri seti {label} alınamadı: {str(e)[:200]}\n"
                f"        (kapılı veri setiyse HF'de erişim koşullarını kabul edip "
                f"HF_TOKEN verin.) Çevrimdışı mod AÇILMAYACAK.")
    return ok


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="doğrulamayı atla, yeniden indir")
    ap.add_argument("--no-datasets", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    from huggingface_hub import get_token
    if not get_token():
        log("UYARI: HF_TOKEN / `huggingface-cli login` yok — anonim istek sınırı çok düşük "
            "(429 riski). `export HF_TOKEN=hf_...` önerilir.")

    known = {}
    if ASSETS.exists():
        try:
            known = json.loads(ASSETS.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            known = {}

    models = collect_models()
    log(f"{len(models)} model: {', '.join(models)}")
    result, failed = {}, []
    for repo in models:
        try:
            result[repo] = ensure_model(repo, known, args.force)
        except Exception as e:  # noqa: BLE001
            failed.append(repo)
            log(f"HATA: {repo}: {str(e)[:300]}")
    ASSETS.write_text(json.dumps({**known, **result}, indent=1), encoding="utf-8")

    data_ok = True if args.no_datasets else ensure_datasets()

    if failed:
        MARK.unlink(missing_ok=True)
        log(f"BAŞARISIZ modeller: {failed}")
        sys.exit(1)

    try:
        from wm.util import update_manifest
        update_manifest(**{"hf.revisions": {r: v["revision"] for r, v in result.items()}})
    except Exception as e:  # noqa: BLE001
        log(f"UYARI: manifest yazılamadı ({e})")

    if data_ok and not args.no_datasets:
        MARK.write_text(json.dumps({"models": models, "time": time.time()}), encoding="utf-8")
        log("TAMAM: önbellek eksiksiz → koşumlar HF_HUB_OFFLINE=1 ile çalışacak.")
    else:
        MARK.unlink(missing_ok=True)
        log("Modeller TAMAM; veri setleri eksik/atlandı → çevrimdışı mod AÇILMAYACAK.")


if __name__ == "__main__":
    main()
