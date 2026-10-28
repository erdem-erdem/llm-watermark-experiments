"""E2 ve E4 üretimleri (Bölüm 5-E2, 5-E4).

  python -m gen.gen_e2_e4 --config configs/e2.yaml --stage e2|e4 [--smoke]

E2: D_gen (watermark'lı, ana anahtar) + D_clean (watermark'sız) 5.000 prompt;
    çeşitlilik: 200 prompt × 2 üretim (wm aynı anahtar / wm'sız farklı tohum).
E4: aracı A-gen-100 ile watermark'sız greedy ve T=0.7 üretim; kontrol A₀ greedy.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from gen.generate import (DEFAULT_STUDENT, DEFAULT_TEACHER, generate_batch,
                          load_prompt_records)
from wm.util import (RESULTS, ROOT, dump_jsonl, log_cmd, load_config,
                     parse_args, read_jsonl, record_duration, update_manifest)

DATA_DIR = ROOT / "data"
E2 = RESULTS / "E2"
E4 = RESULTS / "E4"


def _main_key():
    p = RESULTS / "E0" / "selected_keys.json"
    if not p.exists():
        sys.exit("Anahtar seçimi yapılmamış (data/prep_keys.py çalıştırın).")
    return json.loads(p.read_text(encoding="utf-8"))["main_key"]


def _gen(teacher, prompts, watermark, key, key_id, T, top_p, ntok, seed, tp,
         greedy=False):
    return generate_batch(
        teacher, prompts, watermark=watermark, key=key, key_id=key_id,
        temperature=0.0 if greedy else T, top_p=1.0 if greedy else top_p,
        max_new_tokens=ntok, seed=seed, tp_size=tp,
        sampling_params_extra={"top_k": -1} if greedy else None)


def _prompts(cfg, smoke):
    n = min(50, cfg.get("n_train", 5000)) if smoke else cfg.get("n_train", 5000)
    return load_prompt_records(DATA_DIR / "train.jsonl", limit=n)


TASK_OUTPUTS = {
    "dgen": ["D_gen.jsonl"], "dclean": ["D_clean.jsonl"],
    "div": ["diversity_wm.jsonl", "diversity_nowm.jsonl"],
    "agent": ["agent_greedy.jsonl", "agent_sampled.jsonl"],
    "a0": ["a0_greedy.jsonl"],
}


def run_task(cfg, smoke, task, force=False):
    """Tek görev koşumu — GPU'ya bölünmüş paralel koşum için.

    dgen  : D_gen (wm, ana anahtar)        → data/D_gen.jsonl
    dclean: D_clean (wm'sız)               → data/D_clean.jsonl
    div   : çeşitlilik (4 üretim)          → data/diversity_{wm,nowm}.jsonl
    agent : A-gen-100 greedy + T=0.7       → data/agent_{greedy,sampled}.jsonl
    a0    : A₀ kontrol greedy              → data/a0_greedy.jsonl
    """
    outs = TASK_OUTPUTS.get(task)
    if outs and not smoke and not force and all((DATA_DIR / f).exists() for f in outs):
        print(f"[{task}] çıktılar var, atlanıyor: {outs} (--force ile yeniden üret)")
        return
    key = _main_key()
    teacher = cfg.get("teacher_model", DEFAULT_TEACHER)
    tp = cfg.get("tp_size", 1)
    ntok = cfg["max_new_tokens"]
    T, top_p, seed = cfg["T"], cfg["top_p"], cfg["seed"]
    smoke_n = 50 if smoke else cfg.get("n_train", 5000)

    if task == "dgen":
        prompts = load_prompt_records(DATA_DIR / "train.jsonl", limit=smoke_n)
        d_gen = _gen(teacher, prompts, True, key, "main", T, top_p, ntok, seed, tp)
        dump_jsonl(d_gen, DATA_DIR / "D_gen.jsonl")
        print(f"D_gen: {len(d_gen)} satır")
    elif task == "dclean":
        prompts = load_prompt_records(DATA_DIR / "train.jsonl", limit=smoke_n)
        d_clean = _gen(teacher, prompts, False, None, None, T, top_p, ntok,
                       seed + 11, tp)
        dump_jsonl(d_clean, DATA_DIR / "D_clean.jsonl")
        print(f"D_clean: {len(d_clean)} satır")
    elif task == "div":
        n_div = 10 if smoke else 200
        prompts = load_prompt_records(DATA_DIR / "train.jsonl", limit=n_div)
        wm_a = _gen(teacher, prompts, True, key, "main", T, top_p, ntok,
                    seed + 21, tp)
        wm_b = _gen(teacher, prompts, True, key, "main", T, top_p, ntok,
                    seed + 22, tp)
        nw_a = _gen(teacher, prompts, False, None, None, T, top_p, ntok,
                    seed + 23, tp)
        nw_b = _gen(teacher, prompts, False, None, None, T, top_p, ntok,
                    seed + 24, tp)
        dump_jsonl(wm_a + wm_b, DATA_DIR / "diversity_wm.jsonl")
        dump_jsonl(nw_a + nw_b, DATA_DIR / "diversity_nowm.jsonl")
        print(f"çeşitlilik: {len(wm_a + wm_b)} wm, {len(nw_a + nw_b)} wm'sız")
    elif task == "agent":
        agent_path = Path(cfg["agent_checkpoint"])
        if not (agent_path / "_TRAINING_DONE").exists():
            sys.exit(f"Aracı model yok: {agent_path} — önce E3'ü çalıştırın.")
        prompts = load_prompt_records(DATA_DIR / "train.jsonl",
                                      limit=20 if smoke else smoke_n)
        greedy = _gen(str(agent_path), prompts, False, None, None, T, top_p,
                      ntok, seed, tp, greedy=True)
        sampled = _gen(str(agent_path), prompts, False, None, None, 0.7, top_p,
                       ntok, seed + 31, tp)
        dump_jsonl(greedy, DATA_DIR / "agent_greedy.jsonl")
        dump_jsonl(sampled, DATA_DIR / "agent_sampled.jsonl")
        print(f"ajan: {len(greedy)} greedy, {len(sampled)} örnekleme")
    elif task == "a0":
        ctrl_path = Path(cfg["a0_checkpoint"])
        if not (ctrl_path / "_TRAINING_DONE").exists():
            print(f"  UYARI: A₀ kontrol modeli yok ({ctrl_path}); atlandı")
            return
        prompts = load_prompt_records(DATA_DIR / "train.jsonl",
                                      limit=20 if smoke else smoke_n)
        ctrl = _gen(str(ctrl_path), prompts, False, None, None, T, top_p,
                    ntok, seed, tp, greedy=True)
        dump_jsonl(ctrl, DATA_DIR / "a0_greedy.jsonl")
        print(f"A₀ kontrol: {len(ctrl)} satır")
    elif task == "all":
        for t in ("dgen", "dclean", "div", "agent", "a0"):
            run_task(cfg, smoke, t, force)
    else:
        sys.exit(f"bilinmeyen görev: {task}")
    update_manifest(**{"gen_task": task})


def main():
    argv = parse_args("E2/E4 üretim")
    args = sys.argv[1:]
    stage = "e2"
    task = None
    for i, a in enumerate(args):
        if a == "--stage":
            stage = args[i + 1]
        elif a == "--task":
            task = args[i + 1]
    cfg = load_config(argv.config)
    log_cmd(sys.argv)
    import time
    t0 = time.time()
    if task:
        run_task(cfg, smoke=argv.smoke, task=task, force=argv.force)
    elif stage == "e2":
        for t in ("dgen", "dclean", "div"):
            run_task(cfg, smoke=argv.smoke, task=t, force=argv.force)
    elif stage == "e4":
        for t in ("agent", "a0"):
            run_task(cfg, smoke=argv.smoke, task=t, force=argv.force)
    else:
        sys.exit(f"bilinmeyen stage: {stage}")
    record_duration(f"{stage or task}_gen", time.time() - t0)


if __name__ == "__main__":
    main()
