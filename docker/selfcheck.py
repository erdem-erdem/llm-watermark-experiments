"""İmaj/ortam kendi kendini denetler — pahalı koşudan ÖNCE çalıştırın.

  docker run --rm --gpus all --ipc=host llm-watermark-experiments python /opt/selfcheck.py

Daha önce koşuyu ancak vLLM açılırken çökerten hataları (sqlite3 → libstdc++,
vLLM model-kayıt alt süreci, GPU görünmemesi) saniyeler içinde yakalar.
Çıkış kodu 0 = hepsi geçti.
"""

import importlib
import os
import subprocess
import sys

sys.path.insert(0, os.getcwd())  # /workspace: wm.* içe aktarılabilsin

FAIL = []


def check(name, fn):
    try:
        out = fn()
        print(f"ok    {name}" + (f"  [{out}]" if out else ""))
    except Exception as e:  # noqa: BLE001
        FAIL.append(name)
        print(f"FAIL  {name}  → {type(e).__name__}: {e}")


def _sqlite():
    import sqlite3
    return sqlite3.sqlite_version


def _gpus():
    import torch
    n = torch.cuda.device_count()
    assert n >= 1, "GPU görünmüyor (--gpus all? sürücü? nvidia-container-toolkit?)"
    names = {torch.cuda.get_device_name(i) for i in range(n)}
    # gerçek bir CUDA işlemi: sürücü/CUDA sürüm uyumsuzluğunu yakalar
    x = torch.ones(8, device="cuda") * 2
    assert float(x.sum()) == 16.0
    return f"{n} GPU {sorted(names)}, torch {torch.__version__}, cuda {torch.version.cuda}"


def _ver(mod):
    def f():
        return importlib.import_module(mod).__version__
    return f


def _vllm_registry():
    # vLLM'in model mimarisini incelerken açtığı alt sürecin AYNISI
    # (daha önce sqlite3/libstdc++ yüzünden burada patlıyordu).
    r = subprocess.run([sys.executable, "-m", "vllm.model_executor.models.registry"],
                       capture_output=True, text=True)
    assert r.returncode == 0, (r.stderr or r.stdout)[-600:]
    return "alt süreç temiz"


def _qwen35():
    from vllm.model_executor.models.registry import ModelRegistry
    assert "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs(), \
        "kurulu vLLM Qwen3.5'i tanımıyor — vllm sürümünü yükseltin"


def _optional(mod):
    def f():
        importlib.import_module(mod)
        return "var"
    return f


check("sqlite3", _sqlite)
check("torch + GPU + CUDA işlemi", _gpus)
for m in ("vllm", "transformers", "datasets", "sentence_transformers", "scipy",
          "matplotlib", "numpy", "yaml", "huggingface_hub"):
    check(f"import {m}", _ver(m) if m not in ("yaml",) else _optional(m))
check("vLLM model-kayıt alt süreci", _vllm_registry)
check("vLLM Qwen3_5 mimarisi kayıtlı", _qwen35)


def _wm_proc():
    from wm.vllm_proc import FQCN, WatermarkLogitsProcessor  # noqa: F401
    return FQCN


check("watermark LogitsProcessor (V1 API) içe aktarılıyor", _wm_proc)

for m in ("fla", "flash_attn", "causal_conv1d"):
    try:
        importlib.import_module(m)
        print(f"info  {m}: var")
    except Exception:  # noqa: BLE001
        print(f"info  {m}: yok (isteğe bağlı; yokluğu hata değil)")

print()
if FAIL:
    print("BAŞARISIZ:", ", ".join(FAIL))
    sys.exit(1)
print("TÜM KONTROLLER GEÇTİ")
