# DOCKER_GUIDE — Containerizing and running llm-watermark-experiments on another machine

**Goal:** run the whole pipeline (inference + training + analysis) on any NVIDIA-GPU Linux box with
**no conda, no venv juggling, no system-library conflicts**. The only thing taken from the host is
the NVIDIA driver.

> **Verification status (read this).** The Dockerfile, compose file, self-check and the
> `train/sft.py` fix were written on a Windows machine with no Docker and no GPU. I could not
> build the image or run it. They are written conservatively and the self-check (§5) exists to
> prove the image on the target machine in ~1 minute **before** you spend GPU hours. If the build
> fails, §10 maps symptoms to fixes.

---

## 1. What the project actually needs

There is no server, database, or web app. It is three kinds of workload:

| Workload | Where | Stack |
|---|---|---|
| **Inference** (all generation: E1 sweep/full, E2, E4, olympiad eval, minis, topic labeling) | `gen/`, `eval/aime_tr.py`, `data/prep_data.py` | vLLM `LLM` offline batch + our custom logits processor |
| **Training** (E3: 8 runs, E4: 3 runs, full fine-tune of Qwen3.5-2B-Base, bf16) | `train/sft.py` | HF `transformers` Trainer, PyTorch |
| **Scoring/analysis** (reading-mode forward passes, GPQA, eval-loss, detection, figures) | `eval/`, `analyze/`, `wm/` | PyTorch + transformers + scipy + matplotlib (+ sentence-transformers for E1 similarity) |

Python libs (from `scripts/setup_env.sh` and the imports in the code): `torch`, `vllm`,
`transformers`, `datasets`, `accelerate`, `sentence-transformers`, `scipy`, `matplotlib`, `numpy`,
`pyyaml`, `sentencepiece`, `tiktoken`, `huggingface_hub`.

**Network needs (first run only, unless you pre-populate the cache, §8):**

| What | Source | Note |
|---|---|---|
| Models: `Qwen/Qwen3.5-{0.8B,2B,4B,9B}`, `Qwen/Qwen3.5-2B-Base`, `intfloat/multilingual-e5-large` | Hugging Face | ≈ 45 GB total (approx.) |
| Datasets: `turkish-nlp-suite/InstrucTurca` (**streamed**, never fully stored), `ytu-ce-cosmos/tubitak-science-olympiad-tr`, `ytu-ce-cosmos/gpqa-extended_tr` | Hugging Face | GPQA-style data may be **gated** → needs `HF_TOKEN` |

**Disk (approx.):** image 20–25 GB + HF cache ≈ 45 GB + results (11 trained 2B checkpoints ≈ 4–5 GB
each, plus generations) → plan **≥ 150 GB** free.
**GPU:** designed for 4×H100-80GB; one GPU works (slower). 9B bf16 inference fits one 80 GB GPU.

---

## 2. Why it breaks today, and why Docker fixes it

Your `.venv` was created with `python -m venv` from the **conda `gom` env's Python**. A venv only
isolates `site-packages`; the interpreter, the stdlib (`sqlite3`) and the shared libraries that
stdlib links to (`libicui18n.so.78`, built with GCC 13) still come from conda. That ICU needs
`CXXABI_1.3.15`, but the host's `/lib/x86_64-linux-gnu/libstdc++.so.6` is older, so
`import sqlite3` fails — and vLLM imports `sqlite3` while inspecting the model architecture. This
is a host-environment mismatch, not a bug in the project code.

The container removes the whole class of problem: **its own** Ubuntu 22.04 userspace, **its own**
system Python 3.10 (same minor version as your current env), no conda anywhere, so `sqlite3`,
`libstdc++`, and everything else are a consistent set. Only the NVIDIA kernel driver is shared with
the host.

---

## 3. Target-machine prerequisites (one time)

1. Linux x86_64 with NVIDIA GPUs and a working driver. **Driver ≥ 570** for the default CUDA 12.8
   image (check: `nvidia-smi`). If your pinned PyTorch is built for CUDA 12.9/13.x, you need a
   correspondingly newer driver (see §4 to read the CUDA version).
2. Docker Engine ≥ 24 and the **NVIDIA Container Toolkit**:
   <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
   (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).
3. Prove GPU passthrough works *before anything else*:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
   ```
   You must see all your GPUs. If not, nothing below can work — fix this first.

> If your current machine is itself a container/Kubernetes pod (the `workspace-gom-…` prompt
> suggests so), you usually **cannot run Docker inside it**. Build the image elsewhere and use it
> as the pod image, or run these steps on the new machine.

---

## 4. Pin the environment (the step that guarantees "no dependency bug")

The only reliable way to avoid version drift (vLLM ↔ torch ↔ transformers) is to freeze exactly
what works. On the machine where **vLLM already imports fine and the pipeline got as far as
launching the model** (your current `.venv`), run:

```bash
cd llm-watermark-experiments
source .venv/bin/activate
python -c "import torch,vllm,transformers; print('torch',torch.__version__,'cuda',torch.version.cuda,'| vllm',vllm.__version__,'| transformers',transformers.__version__)"
pip freeze --exclude-editable | grep -v ' @ file:' > docker/requirements.lock.txt
```

- Look at the printed `cuda` value: it must be ≤ what your target driver supports (§3.1). If it is
  not `12.8`, rebuild with the matching base image, e.g.
  `--build-arg CUDA_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04`.
- If `docker/requirements.lock.txt` contains `torch==…+cu128` (a local version tag), add
  `--build-arg EXTRA_INDEX=https://download.pytorch.org/whl/cu128` when building.
- Commit the lock file with the code. It is the single source of truth for versions.

**No access to the old environment?** Build from the loose fallback list instead
(`--build-arg REQ_FILE=requirements.txt`). It installs `vllm` first (which selects its own matching
`torch`) and then the rest, so pip will *error out* on a real conflict rather than silently
mismatching. Then run `pip freeze` **inside the built image** and save it as the new lock:
```bash
docker run --rm llm-watermark-experiments pip freeze > docker/requirements.lock.txt
```

---

## 5. Build and prove the image

```bash
cd llm-watermark-experiments          # the folder that contains docker/ and docker-compose.yml

# Build (≈ 10–20 min; downloads torch/vllm wheels)
docker build -f docker/Dockerfile -t llm-watermark-experiments:latest .
#   fallback list:  --build-arg REQ_FILE=requirements.txt
#   torch +cu tags: --build-arg EXTRA_INDEX=https://download.pytorch.org/whl/cu128
#   other CUDA:     --build-arg CUDA_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04
#   optional kernel: --build-arg INSTALL_CAUSAL_CONV1D=1   (slow compile; faster Qwen3.5 training)

# Self-check: sqlite3, GPU math, imports, vLLM's model-registry subprocess, Qwen3.5 support
docker run --rm --gpus all --ipc=host llm-watermark-experiments:latest python /opt/selfcheck.py
```

Expected last line: `TÜM KONTROLLER GEÇTİ` (all checks passed). The two checks that
reproduce your old failure are **`sqlite3`** and **`vLLM model-kayıt alt süreci`** — both must be
`ok`. **Do not start a real run until this passes.**

---

## 6. Running

All commands run from the project root. The project folder is **bind-mounted** at `/workspace`
(code + `data/` + `results/` all live on the host and survive container deletion; this is
deliberate because `data/` contains both code and generated files). The Hugging Face cache is a
second mount so models are downloaded once.

Define once (adjust paths):
```bash
export HF_TOKEN=hf_xxx                     # needed for gated datasets; omit if not needed
export HF_CACHE_DIR=$HOME/hf_cache         # host folder for the HF cache
mkdir -p "$HF_CACHE_DIR"
alias wm='docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN -v "$PWD":/workspace -v "$HF_CACHE_DIR":/hf llm-watermark-experiments:latest'
```
(Or use compose: `docker compose run --rm wm <command>`; same effect.)

### 6.1 Smoke run (≈ 30–60 min incl. downloads)
```bash
wm bash scripts/run_smoke_all.sh
```

### 6.2 Full run — up to the E1 gate
```bash
wm bash scripts/run_all.sh 4
```
By default the E1 setting is picked automatically (rule documented in `RUN_MANUAL.md` §3) and the run continues through the whole pipeline. To stop at the gate and choose yourself, run `wm env E1_GATE=manual bash scripts/run_all.sh 4`; it then prints `results/E1/GATE.md`.

### 6.3 Approve the gate (only with `E1_GATE=manual`; on the **host**, the files are in your project folder)
Review `results/E1/SWEEP_REPORT.md` and `figures/en/main/fig_E1_1.png`, then write (values from the report):
```bash
cat > results/E1/selected.json <<'EOF'
{"model": "Qwen/Qwen3.5-9B", "T": 1.0, "approved": true}
EOF
```

### 6.4 Rest of the pipeline
```bash
wm bash scripts/run_e1_full.sh 4
```
Output: tables/CSVs under `results/`, figures (TR + EN) under `figures/`.

### 6.5 Choosing how many GPUs to use
Two independent knobs:

| Goal | How |
|---|---|
| Use only N GPUs | `wm bash scripts/run_all.sh N` (the scripts split heavy jobs across N) |
| Hide GPUs from the container entirely | `--gpus '"device=0,1"'` in the `docker run` line (inside the container they appear as GPU 0 and 1, so the scripts' default `0..N-1` list just works) |
| Non-contiguous inside the container | `-e GPU_IDS=0,2` |

Example, two specific GPUs: replace `--gpus all` with `--gpus '"device=2,3"'` and run
`... bash scripts/run_e1_full.sh 2`.

### 6.6 Run one stage / debug interactively
```bash
wm bash                                   # shell inside the container
wm python -m analyze.detect_all --config configs/detect.yaml --stage E3 --only A-gen-10
wm python -m eval.aime_tr --config configs/eval.yaml --only main
```
Run long jobs detached: replace `--rm` with `-d --name wmrun` and follow with `docker logs -f wmrun`.

### 6.7 File ownership
By default the container runs as root, so files it writes into your project folder are root-owned.
Either `sudo chown -R "$USER": .` afterwards, or add `--user "$(id -u):$(id -g)"` to the `docker run`
line (`HOME` and the Triton/HF caches are already pointed at writable locations).

---

## 7. What lives where

| Path (container) | Host location | Persistent? | Contents |
|---|---|---|---|
| `/workspace` | the project folder | ✅ | code, `data/*.jsonl` (generated), `results/` (everything the blog needs), logs |
| `/hf` | `$HF_CACHE_DIR` | ✅ | downloaded models/datasets |
| `/opt/venv`, system libs | inside the image | image layer | Python env — never touched at run time |

To move results: copy `results/` (and `data/*.jsonl` if you want to resume). Every stage is
idempotent: re-running skips finished outputs (`--force` to redo).

---

## 8. Air-gapped / offline target machine

1. On a connected machine: build the image, then
   `docker save llm-watermark-experiments:latest | gzip > wm-image.tar.gz`; copy it over and
   `gunzip -c wm-image.tar.gz | docker load`.
2. Warm the HF cache on the connected machine (run `scripts/run_smoke_all.sh` once, or
   `huggingface-cli download <repo>` for each model listed in §1) and copy the whole
   `$HF_CACHE_DIR` folder to the target.
3. Run with `-e HF_HUB_OFFLINE=1`.

Caveat: `data/prep_data.py` **streams** InstrucTurca, which needs network. Either run the data
preparation step (`run_all.sh` step 1) on a connected machine and copy `data/*.jsonl`, or give that
one step network access. All later stages work offline once the cache is warm (the revision-hash
lookup in `wm/util.py` simply returns `None` without network).

---

## 9. Changes made to the project for this

| File | Change |
|---|---|
| `docker/Dockerfile`, `docker/requirements.txt`, `docker/selfcheck.py`, `.dockerignore`, `docker-compose.yml` | new |
| `train/sft.py` | **Real latent bug fixed.** Training hard-requested `attn_implementation="flash_attention_2"`, but no step installs `flash-attn`, so the first E3 training would have crashed on any clean machine. It now uses `flash_attention_2` only if the package exists and otherwise PyTorch SDPA (which uses flash kernels on H100; same math). The chosen backend is printed at startup. |

The LD_PRELOAD guards stay removed, as you asked; the container makes them unnecessary.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing/unconfigured (§3.2). |
| `nvidia-smi` works, selfcheck says `CUDA driver version is insufficient` / GPU math FAIL | Driver older than the image's CUDA. Upgrade the driver, or rebuild with an older `CUDA_IMAGE` matching `torch.version.cuda`. |
| Build: `COPY failed … requirements.lock.txt not found` | Create the lock (§4) or build with `--build-arg REQ_FILE=requirements.txt`. |
| Build: `No matching distribution … torch==X+cuYYY` | Add `--build-arg EXTRA_INDEX=https://download.pytorch.org/whl/cuYYY`. |
| Build: pip dependency conflict with the loose list | That is the safety net working; use the lock file from the working environment. |
| selfcheck: `vLLM Qwen3_5 mimarisi kayıtlı` FAIL | Installed vLLM is too old for Qwen3.5 → use a newer `vllm` in the lock. |
| `CUDA out of memory` in vLLM | Lower `gpu_mem_util` in `gen/generate.py` call sites or free the GPU; one vLLM per GPU is the design. |
| `RuntimeError: DataLoader … shared memory` / bus error | Missing `--ipc=host` (or `--shm-size=16g`). |
| Training slow on Qwen3.5 linear-attention layers | `flash-linear-attention` failed to install (watch the build log) or build with `INSTALL_CAUSAL_CONV1D=1`. Correct either way, just slower. |
| `401/403` on a dataset | Gated repo: accept its terms on huggingface.co and pass `-e HF_TOKEN`. |
| Files in `results/` owned by root | §6.7. |
| Everything vanished after the run | You forgot the `-v "$PWD":/workspace` mount (outputs went into the container's own layer). |
