# llm-watermark-experiments

**English** · [Türkçe](README.tr.md)

Code, data pipeline, results and figures for a Turkish-language study of **LLM watermarking** and
**watermark radioactivity**: can a watermark planted in text still be detected in a model that was
trained on that text?

The full write-up (method, experiments, results and their interpretation) is in the blog series:

- 🇹🇷 Turkish: <https://kureansiklopedi.com/tr/detay/llm-watermarking-12-iz-birakmak-calisma-prensipl-2>
- 🇬🇧 English: <https://kureansiklopedi.com/en/detay/llm-watermarking-12-watermarking-working-principle>

This repository does **not** discuss the findings; it gives you the code to reproduce them, the raw
result tables, and every figure in Turkish and English.

[![CI](https://github.com/erdem-erdem/llm-watermark-experiments/actions/workflows/ci.yml/badge.svg)](https://github.com/erdem-erdem/llm-watermark-experiments/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Contents

1. [Concepts in five minutes](#concepts-in-five-minutes)
2. [Repository layout](#repository-layout)
3. [Try it yourself (no GPU)](#try-it-yourself-no-gpu)
4. [Rebuild the figures](#rebuild-the-figures)
5. [Run the full pipeline (GPU)](#run-the-full-pipeline-gpu)
6. [What is committed in `results/`](#what-is-committed-in-results)
7. [Data, models and licenses](#data-models-and-licenses)
8. [Notes and known quirks](#notes-and-known-quirks)

---

## Concepts in five minutes

**Watermarking at generation time.** A language model picks the next token from a probability
distribution `p`. We replace the random draw with a *keyed* one. A secret 64-bit `key` and the last
`k = 2` token ids are hashed (`wm/prf.py`) into a pseudo-random number `R_v ∈ (0, 1)` for every
vocabulary token `v`. The model then outputs `argmax_v R_v^(1/p_v)` (the Gumbel-max / exponential-race
trick). Without the key this is an exact sample from `p` — the text is distributed as usual — but
with the key, the chosen tokens tend to have large `R`.

**Detection.** Given a text, recompute `R` for every token with the key and score `s = −ln(1 − R)`.
For unwatermarked text `s ~ Exp(1)`, so the sum over `N` scored tokens follows `Gamma(N, 1)` and gives
a p-value. We always report `−log₁₀ p` together with `N`. Repeated windows are skipped (de-duplication
rules in `wm/detect.py`), otherwise repeated n-grams would inflate the evidence.

**Post-hoc watermarking vs. generation-time.** The same scheme can be applied while a model *writes*
an answer, or while it *rewrites* an existing answer (`gen/gen_e1.py`).

**Radioactivity and "reading mode".** If a model was fine-tuned on watermarked text, its own
predictions may drift toward watermark-favoured tokens. To test this without any generated text, we
feed text to the suspect model (one teacher-forced pass), take its predicted token at each position,
and score *that prediction* against the key instead of the text token (`wm/reading_mode.py`).

The technical specification — every parameter and rule — is in [`SPEC.md`](SPEC.md) (Turkish). For
intuition, figures and results, read the blog.

---

## Repository layout

```
wm/            watermark core: PRF, Gumbel-max sampler, scoring, detection, reading mode, vLLM adapter
  tests/       unit tests (CPU; PRF bit-exactness, sampler↔detector consistency, …)
data/          dataset preparation (prep_data.py) and key selection (prep_keys.py); DATA_CARD.md
gen/           generation scripts (vLLM + custom logits processor)
train/         supervised fine-tuning (HF Trainer) + one YAML per trained model (train/configs/)
eval/          GPQA-TR, math-olympiad accuracy, held-out loss, n-gram overlap check
analyze/       detection orchestration, tables, and all figure code (bilingual)
configs/       YAML configuration of every stage
scripts/       environment setup, GPU job runner, end-to-end run scripts
docker/        Dockerfile, self-check, fallback requirements
examples/      toy_demo.py — watermark + detect a toy model on CPU in seconds
results/       committed result tables and logs (CSV / JSON / JSONL / MD), by stage
figures/       tr/ and en/ — main/ (headline figures) and extra/ (per-stage diagnostics), PNG + SVG
SPEC.md        technical specification (Turkish)
RUN_MANUAL.md  how to run the pipeline (Turkish)      DOCKER_GUIDE.md  Docker walkthrough (English)
```

Pipeline stages, as named in code and in `results/`:

| Stage | What the code does |
|---|---|
| `E0` | sanity-checks the detector on clean models and wrong keys; picks the key |
| `E1` | rewrites existing answers with a watermark (post-hoc) and without one (control) |
| `E2` | generates answers with a watermark during generation; quality and diversity checks |
| `E3` | fine-tunes student models on data mixes of the above and measures detection in reading mode |
| `E4` | distills a further student from a student's outputs and measures detection along the chain |
| `mini` | the small demonstrations used in the blog's introductory sections |

---

## Try it yourself (no GPU)

Everything in this section runs on a laptop CPU.

```bash
git clone https://github.com/erdem-erdem/llm-watermark-experiments.git
cd llm-watermark-experiments
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu-tests.txt
```

**1. Watermark a toy "language model" and detect it** (≈10 s):

```bash
python -m examples.toy_demo
```

A random bigram model stands in for an LLM. The script samples text with the key, runs the real detector
from `wm/` on it with the right and a wrong key, and does the same for ordinary sampling. Look at how
`−log₁₀ p` behaves in the three rows. (A toy model is much peakier than a real LLM, so the absolute
numbers are not comparable to the blog's.)

**2. Use the pieces directly:**

```python
from wm.detect import detect_text

res = detect_text(prompt_ids=[...], answer_ids=[...], key=0x1234ABCD5678EF01, k=2)
print(res.N, res.neglog10p)            # scored tokens, −log10(p)
```

The `answer_ids` are token ids. Detection works on ids, not strings, because re-tokenising text can
silently change the windows (see `SPEC.md`, rule 2).

**3. Run the unit tests:**

```bash
make test        # or: python -m wm.tests.test_prf, etc.
```

---

## Rebuild the figures

All figures are generated from the committed tables in `results/`, in both languages:

```bash
pip install -r requirements-figures.txt
python -m analyze.make_figures              # tr + en, about a minute
python -m analyze.make_figures --lang en    # one language
python -m analyze.blog_figures --lang en E0 E1_1     # selected figures only
```

Output goes to `figures/<lang>/main/` (the headline figures) and `figures/<lang>/extra/`
(per-stage diagnostics), each as PNG and SVG. The Turkish model outputs shown inside some figures
(example texts, tokens) are data and stay Turkish in the English versions.
The figures need real data: if a results file is missing, the script stops and names the file and the
stage that produces it — nothing is ever filled in with placeholders.

---

## Run the full pipeline (GPU)

The study was run on **4 × H100** (80 GB). A single GPU works but is slow; the scripts split heavy jobs
over however many GPUs you give them. Expect roughly 45 GB of Hugging Face cache plus 20–25 GB for the
Docker image, and a few GB for the trained checkpoints.

**Recommended: Docker** (avoids vLLM/torch/CUDA mismatches). Full walkthrough in
[`DOCKER_GUIDE.md`](DOCKER_GUIDE.md); the short version:

```bash
docker build -f docker/Dockerfile -t llm-watermark-experiments:latest .
docker run --rm --gpus all --ipc=host llm-watermark-experiments:latest python /opt/selfcheck.py   # must pass first

export HF_TOKEN=hf_xxx                       # some datasets are gated; see "Data, models and licenses"
alias wm='docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN -v "$PWD":/workspace -v "$HOME/hf_cache":/hf llm-watermark-experiments:latest'

wm bash scripts/run_smoke_all.sh             # small end-to-end check (~1 h incl. downloads)
wm bash scripts/run_all.sh 4                 # data, keys, E1 sweep + automatic setting choice, then continues
# run_all.sh calls scripts/run_e1_full.sh itself (E2–E4, evaluations, tables, figures);
# you only run it by hand after `E1_GATE=manual` or to resume:  wm bash scripts/run_e1_full.sh 4
```

**Without Docker:** `bash scripts/setup_env.sh` (see [`RUN_MANUAL.md`](RUN_MANUAL.md) §1). The version
lock file `docker/requirements.lock.txt` is *not* shipped — generate it from a working environment
with the command in `DOCKER_GUIDE.md` §4, or build from the loose list with
`--build-arg REQ_FILE=requirements.txt`.

Every stage is idempotent: a stage whose outputs exist is skipped (`--force` redoes it). Each command is
appended to `results/LOG.md`, and library versions, seeds and durations go to `results/run_manifest.json`.
Stage-by-stage details, GPU distribution and troubleshooting are in `RUN_MANUAL.md` (Turkish) and
`DOCKER_GUIDE.md`.

---

## What is committed in `results/`

Per-stage CSV / JSON / JSONL / Markdown outputs: detection tables and curves, p-value distributions,
evaluation scores, generation samples, the key-selection record, the run log and manifest.

Not committed (too large or reproducible): model checkpoints, the generated datasets in `data/*.jsonl`
(rebuild with `python -m data.prep_data`), `.npz` prediction arrays, raw per-job logs, and the raw
post-hoc generation shards (`results/E1/full_parts/`). They are rebuilt by the pipeline; the committed
tables are what the figures are drawn from.

---

## Data, models and licenses

| Resource | Use | License / note |
|---|---|---|
| [`turkish-nlp-suite/InstrucTurca`](https://huggingface.co/datasets/turkish-nlp-suite/InstrucTurca) | instruction data (streamed, filtered subset) | CC BY-SA 4.0 — attribution required; see `data/DATA_CARD.md` |
| [`ytu-ce-cosmos/tubitak-science-olympiad-tr`](https://huggingface.co/datasets/ytu-ce-cosmos/tubitak-science-olympiad-tr) | math-olympiad quality check | see dataset card |
| [`ytu-ce-cosmos/gpqa-extended_tr`](https://huggingface.co/datasets/ytu-ce-cosmos/gpqa-extended_tr) | downstream evaluation | may be gated: accept its terms and pass `HF_TOKEN` |
| Qwen3.5 (0.8B, 2B, 2B-Base, 4B, 9B) | generators / students | see each model card on Hugging Face |
| `intfloat/multilingual-e5-large` | semantic similarity | see model card |

No model weights or dataset dumps are redistributed here. The code is released under the
[MIT License](LICENSE).

---

## Notes and known quirks

- **`aime_*` names are legacy.** The specification first planned an `aime25-tr` math set. The actual runs
  use `ytu-ce-cosmos/tubitak-science-olympiad-tr` (`ortaokul_matematik` split, rows with purely numeric
  answers). Files and functions named `aime_*` / `AIME25-TR` (e.g. `eval/aime_tr.py`,
  `results/E2/aime_summary.json`) evaluate that olympiad set, not AIME.
- **Comments and docstrings are mostly Turkish**; identifiers and the READMEs are not. `SPEC.md` and
  `RUN_MANUAL.md` are Turkish; `DOCKER_GUIDE.md` is English.
- Scripts expect UTF-8; on Windows set `PYTHONIOENCODING=utf-8` (the `Makefile` does).
- Figures use Arial when installed and fall back to DejaVu Sans; SVGs keep text as text.
- `gen/gen_opening.py` expects your own `data/opening_draft.txt` (first paragraph = the quote, then a
  blank line, then the paragraph); the file is not shipped, and its committed output is under `results/mini/`.
- `python -m analyze.summary` writes `results/SUMMARY.md` but its E3 table has not been updated for the
  final reading-mode outputs, so the file is not committed.
