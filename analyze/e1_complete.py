"""E1 tamamlama (CPU): sweep.csv birleştirme + tam veri (D_post) istatistikleri.

fig_e1_sweep ve e1_report `E1/sweep.csv`'yi bekliyordu; üretim ise
E1/sweep_parts/*.csv yazıyordu (dosya hiç oluşmadığı için şekil sessizce atlandı).

  python -m analyze.e1_complete
Çıktılar: E1/sweep.csv, E1/full_summary.json, E1/example_candidates.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np

from wm.util import (RESULTS, ROOT, dump_csv, dump_json, log_cmd, parse_args,
                     read_csv, read_jsonl, record_duration)

E1 = RESULTS / "E1"


def main():
    parse_args("E1 tamamlama")
    log_cmd(sys.argv)
    t0 = time.time()
    rows = []
    for p in sorted((E1 / "sweep_parts").glob("sweep_*.csv")):
        rows += read_csv(p)
    keys = list(rows[0].keys())
    dump_csv(rows, E1 / "sweep.csv", keys)
    print(f"sweep.csv: {len(rows)} satır")

    # tam veri: D_post
    from analyze.e2_complete import _per_record, _keys
    key, _ = _keys()
    post = list(read_jsonl(ROOT / "data" / "D_post.jsonl"))
    rew = {r["id"]: r for r in read_jsonl(ROOT / "data" / "D_rew.jsonl")}
    ex, by_topic = [], {}
    from wm.scoring import log10_p
    for r in post:
        t = r["topic"]
        d = by_topic.setdefault(t, {"n": 0, "filtered": 0})
        d["n"] += 1
        d["filtered"] += bool(r["filtered"])
        lr = len(r["answer"]) / max(1, len(r["source_answer"]))
        sc = _per_record(r, key)
        ex.append({"id": r["id"], "topic": t, "filtered": r["filtered"],
                   "think_leak": r.get("think_leak"),
                   "len_ratio": lr, "N": len(sc),
                   "neglog10p": log10_p(float(sc.sum()), len(sc)) if len(sc) else "",
                   "src_chars": len(r["source_answer"]), "out_chars": len(r["answer"])})
    dump_csv(ex, E1 / "example_candidates.csv",
             ["id", "topic", "filtered", "think_leak", "len_ratio", "N",
              "neglog10p", "src_chars", "out_chars"])
    ok = [e for e in ex if not e["filtered"]]
    lr = np.asarray([e["len_ratio"] for e in ex])
    out = {"n": len(post), "filtered": sum(bool(e["filtered"]) for e in ex),
           "filter_rate": float(np.mean([bool(e["filtered"]) for e in ex])),
           "by_topic": {t: {**d, "rate": d["filtered"] / d["n"]} for t, d in by_topic.items()},
           "len_ratio_quantiles_q10_50_90": [float(np.percentile(lr, q)) for q in (10, 50, 90)],
           "median_neglog10p_kept": float(np.median([e["neglog10p"] for e in ok if e["neglog10p"] != ""])),
           "median_neglog10p_all": float(np.median([e["neglog10p"] for e in ex if e["neglog10p"] != ""])),
           "n_rew": len(rew)}
    dump_json(out, E1 / "full_summary.json")
    record_duration("e1_complete", time.time() - t0)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
