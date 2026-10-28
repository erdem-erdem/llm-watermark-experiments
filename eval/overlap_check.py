"""8-gram örtüşme kontrolü — Bölüm 4.1 (train ↔ eval sızıntı denetimi).

InstrucTurca train / eval_loss / sweep bölmlerindeki örnekler arasında
cevap metinlerinin 8-gram örtüşmesini sayar; yüksek sızıntı bulunan çiftleri
raporlar (örneklemesi değil, veri hazırlama sonrası sağlık kontrolü).

  python -m eval.overlap_check [--smoke]
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

from wm.util import ROOT, RESULTS, dump_csv, dump_json, log_cmd, parse_args, \
    read_jsonl, record_duration, update_manifest

DATA_DIR = ROOT / "data"
N = 8


def answer_ngrams(rec) -> set:
    """Cevap metninden (normalleştirilmiş) 8-gram kümesi."""
    text = rec.get("answer") or ""
    toks = "".join(c.lower() if not c.isdigit() else " " for c in text).split()
    return {" ".join(toks[i:i + N]) for i in range(max(0, len(toks) - N + 1))}


def load_sets(smoke: bool):
    files = {"train": DATA_DIR / "train.jsonl",
             "eval_loss": DATA_DIR / "eval_loss.jsonl",
             "sweep": DATA_DIR / "sweep.jsonl"}
    sets = {}
    for name, p in files.items():
        if not p.exists():
            print(f"uyarı: {p} yok — bu bölüm atlanıyor")
            continue
        recs = list(read_jsonl(p))
        if smoke:
            recs = recs[:100]
        sets[name] = {r["id"]: answer_ngrams(r) for r in recs if r.get("answer")}
    return sets


def pair_overlap(sets, a_name, b_name):
    hits = []
    for rid, ng_a in sets[a_name].items():
        for rid_b, ng_b in sets[b_name].items():
            common = ng_a & ng_b
            if common:
                hits.append({"a_id": rid, "b_id": rid_b, "n_shared_8grams": len(common),
                             "example": sorted(common)[0][:120]})
    return hits


def main():
    argv = parse_args("8-gram örtüşme denetimi")
    log_cmd(sys.argv)
    t0 = time.time()
    sets = load_sets(argv.smoke)
    names = sorted(sets)
    out_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            hits = pair_overlap(sets, names[i], names[j])
            n_hits = len(hits)
            n_a = len(sets[names[i]])
            # ÇİFT sayısı bir örneğin birçok eşle çakışmasını tekrar sayar (>%100
            # olabilir); asıl ölçü: çakışan BENZERSİZ a örneği sayısı.
            n_distinct = len({h["a_id"] for h in hits})
            print(f"{names[i]} ↔ {names[j]}: {n_hits} örnek çifti 8-gram paylaşıyor; "
                  f"çakışan benzersiz {names[i]} örneği: {n_distinct}/{n_a} "
                  f"(%{100 * n_distinct / max(1, n_a):.1f})")
            out_rows.append({"pair": f"{names[i]}↔{names[j]}",
                             "n_pairs_with_shared_8gram": n_hits,
                             "n_samples_a": n_a,
                             "n_distinct_a_overlapping": n_distinct,
                             "pct_a": 100 * n_distinct / max(1, n_a)})
    out = {"n": N, "summary": out_rows}
    dump_json(out, RESULTS / "E0" / "overlap_check.json")
    if out_rows:
        dump_csv(out_rows, RESULTS / "E0" / "overlap_check.csv",
                 fieldnames=["pair", "n_pairs_with_shared_8gram",
                             "n_samples_a", "n_distinct_a_overlapping", "pct_a"])
    update_manifest(**{"eval.overlap": out_rows})
    record_duration("overlap_check", time.time() - t0)
    print("örtüşme denetimi tamam")


if __name__ == "__main__":
    main()