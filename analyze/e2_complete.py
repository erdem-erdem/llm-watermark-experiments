"""E2 tamamlama (CPU, yalnızca data/*.jsonl'den) — eksik kalan ölçümler.

`detect_all --stage E2` yalnızca D_gen / D_clean için tespit yapıyordu. Bu betik:
  - D_post'u da aynı yöntemle puanlar (aynı sorular, post-hoc),
  - örnek düzeyi p'yi kural (a)-(c) ile (kayıt içi) hesaplar → sıfır-N kayıtlar
    -inf üretmez, kayıt sırasına bağımlılık kalmaz; kural (d) KÜRESEL toplamlar
    için ayrıca uygulanır,
  - n ∈ {25,50,100,200,300,400} eğrilerini yalnızca en az n token puanlanabilen
    kayıtlar üzerinden (kaç kayıt kaldığıyla) verir,
  - token başına ortalama skor (S/N) dağılımı: D_gen / D_post / D_clean,
  - çeşitlilik: aynı-çıktı oranı (200 prompt × 2), distinct-2, ortalama uzunluk,
  - T_E2.csv'yi satırlar: D_gen, D_post, D_clean (+ AIME) olarak yeniden yazar.

  python -m analyze.e2_complete
Çıktılar: E2/per_record.csv, E2/curves_summary.csv, E2/token_mean_scores.csv,
          E2/diversity.json, E2/T_E2.csv, E2/E2_summary.json
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from wm.detect import detect_dataset, detect_prefix_curves
from wm.scoring import log10_p
from wm.util import (RESULTS, ROOT, dump_csv, dump_json, log_cmd, parse_args,
                     read_jsonl, record_duration, update_manifest)

DATA = ROOT / "data"
E2 = RESULTS / "E2"
NS = [25, 50, 100, 200, 300, 400]
THR = 6.0
SETS = {"D_gen": "D_gen.jsonl", "D_post": "D_post.jsonl", "D_clean": "D_clean.jsonl"}


def _keys():
    sel = json.loads((RESULTS / "E0" / "selected_keys.json").read_text(encoding="utf-8"))
    return sel["main_key"], sel["wrong_keys"]


def _per_record(rec, key, k=2):
    """Kayıt içi kural (a)-(c); chunk_bounds varsa parça başı k token atlanır."""
    from wm.detect import _in_chunk_head
    from wm.prf import R
    from wm.sampler import text_detection_scores
    from wm.scoring import per_token_score
    ans, plen = rec["answer_ids"], len(rec["prompt_ids"])
    bounds = rec.get("chunk_bounds")
    seen, sc = set(), []
    for pos, window in text_detection_scores(ans, plen, key, k):
        if bounds and _in_chunk_head(pos, bounds, k):
            continue
        if window in seen:
            continue
        seen.add(window)
        sc.append(per_token_score(R(key, window, int(ans[pos]))))
    return np.asarray(sc)


def distinct2(records):
    """Korpus düzeyi distinct-2: benzersiz kelime ikilisi / toplam kelime ikilisi."""
    seen, total = set(), 0
    for r in records:
        w = (r.get("answer") or "").lower().split()
        for a, b in zip(w, w[1:]):
            seen.add((a, b))
            total += 1
    return len(seen) / total if total else float("nan")


def main():
    argv = parse_args("E2 tamamlama")
    log_cmd(sys.argv)
    t0 = time.time()
    key, wrong = _keys()
    data = {s: list(read_jsonl(DATA / f)) for s, f in SETS.items()}
    if argv.smoke:
        data = {s: v[:50] for s, v in data.items()}

    per_rows, tok_rows, summ = [], [], {}
    curve_vals = {s: {n: [] for n in NS} for s in SETS}
    curve_topic = {}
    for s, recs in data.items():
        scores_all = []
        for rec in recs:
            sc = _per_record(rec, key)
            N = len(sc)
            nl = log10_p(float(sc.sum()), N) if N else float("nan")
            per_rows.append({"set": s, "id": rec["id"], "topic": rec.get("topic"),
                             "N": N, "neglog10p": nl,
                             "mean_score": float(sc.mean()) if N else float("nan"),
                             "filtered": rec.get("filtered", False)})
            if N:
                tok_rows.append({"set": s, "id": rec["id"], "topic": rec.get("topic"),
                                 "mean_score": float(sc.mean()), "N": N})
            for n in NS:
                if N >= n:
                    v = log10_p(float(sc[:n].sum()), n)
                    curve_vals[s][n].append(v)
                    curve_topic.setdefault((s, rec.get("topic"), n), []).append(v)
            scores_all.append(N)
        # küresel (kural d) toplam
        res, _ = detect_dataset(recs, key)
        wr = [detect_dataset(recs[:500], wk)[0].neglog10p for wk in wrong]
        summ[s] = {"n": len(recs), "global_N": res.N, "global_neglog10p": res.neglog10p,
                   "wrong_key_global_neglog10p_500": wr,
                   "mean_len_tokens": float(np.mean([len(r["answer_ids"]) for r in recs])),
                   "distinct2": distinct2(recs)}
    dump_csv(per_rows, E2 / "per_record.csv",
             ["set", "id", "topic", "N", "neglog10p", "mean_score", "filtered"])
    dump_csv(tok_rows, E2 / "token_mean_scores.csv",
             ["set", "id", "topic", "mean_score", "N"])

    # eğri özeti (yalnızca N >= n olan kayıtlar) — genel ve konu bazlı
    crow = []
    for s in SETS:
        for n in NS:
            v = np.asarray(curve_vals[s][n])
            if len(v):
                crow.append({"set": s, "topic": "hepsi", "n": n, "count": len(v),
                             "median": float(np.median(v)),
                             "q25": float(np.percentile(v, 25)),
                             "q75": float(np.percentile(v, 75)),
                             "detect_rate": float(np.mean(v >= THR)),
                             "rate_p01": float(np.mean(v >= 2.0))})
        for (ss, tp, n), vv in sorted(curve_topic.items(), key=lambda x: (x[0][1] or "", x[0][2])):
            if ss == s:
                vv = np.asarray(vv)
                crow.append({"set": s, "topic": tp, "n": n, "count": len(vv),
                             "median": float(np.median(vv)),
                             "q25": float(np.percentile(vv, 25)),
                             "q75": float(np.percentile(vv, 75)),
                             "detect_rate": float(np.mean(vv >= THR)),
                             "rate_p01": float(np.mean(vv >= 2.0))})
    dump_csv(crow, E2 / "curves_summary.csv",
             ["set", "topic", "n", "count", "median", "q25", "q75",
              "detect_rate", "rate_p01"])

    # çeşitlilik: aynı prompt'a iki üretim
    div = {}
    for tag, f in (("wm", "diversity_wm.jsonl"), ("nowm", "diversity_nowm.jsonl")):
        recs = list(read_jsonl(DATA / f))
        by = {}
        for r in recs:
            by.setdefault(r["id"], []).append(tuple(r["answer_ids"]))
        pairs = [v for v in by.values() if len(v) >= 2]
        same = float(np.mean([v[0] == v[1] for v in pairs])) if pairs else float("nan")
        # distinct-2: yalnızca İLK üretim (aynı çıktılı çiftler ikinci kez
        # sayılıp oranı yapay biçimde düşürmesin)
        first, seen_ids = [], set()
        for r in recs:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                first.append(r)
        div[tag] = {"n_prompts": len(pairs), "same_output_rate": same,
                    "distinct2": distinct2(first),
                    "distinct2_all_runs": distinct2(recs),
                    "mean_len_tokens": float(np.mean([len(r["answer_ids"]) for r in recs]))}
    dump_json(div, E2 / "diversity.json")

    # T_E2
    def med(s, col="neglog10p"):
        v = [r[col] for r in per_rows if r["set"] == s and r["N"] > 0]
        return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))

    def rate(s, n):
        v = np.asarray(curve_vals[s][n])
        return (float(np.mean(v >= THR)), len(v)) if len(v) else (float("nan"), 0)

    aime = json.loads((E2 / "aime_summary.json").read_text(encoding="utf-8"))
    rows = []
    for s in SETS:
        m, q1, q3 = med(s)
        r400, c400 = rate(s, 400)
        r200, c200 = rate(s, 200)
        row = {"set": s, "n": summ[s]["n"], "median_neglog10p": m,
               "q25": q1, "q75": q3,
               "rate_n200": r200, "count_n200": c200,
               "rate_n400": r400, "count_n400": c400,
               "global_neglog10p": summ[s]["global_neglog10p"],
               "global_N": summ[s]["global_N"],
               "mean_len_tokens": summ[s]["mean_len_tokens"],
               "distinct2": summ[s]["distinct2"],
               "same_output_rate": "", "accuracy_mean": "", "accuracy_std": ""}
        if s == "D_gen":
            row["same_output_rate"] = div["wm"]["same_output_rate"]
            row["accuracy_mean"], row["accuracy_std"] = aime["wm_mean"], aime["wm_std"]
        if s == "D_clean":
            row["same_output_rate"] = div["nowm"]["same_output_rate"]
            row["accuracy_mean"], row["accuracy_std"] = aime["nowm_mean"], aime["nowm_std"]
        rows.append(row)
    dump_csv(rows, E2 / "T_E2.csv",
             ["set", "n", "median_neglog10p", "q25", "q75", "rate_n200", "count_n200",
              "rate_n400", "count_n400", "global_neglog10p", "global_N",
              "mean_len_tokens", "distinct2", "same_output_rate",
              "accuracy_mean", "accuracy_std"])
    dump_json({"sets": summ, "diversity": div}, E2 / "E2_summary.json")
    update_manifest(**{"e2.complete": {s: summ[s]["global_neglog10p"] for s in summ}})
    record_duration("e2_complete", time.time() - t0)
    print("E2 tamamlandı")
    for r in rows:
        print(r["set"], "medyan", round(r["median_neglog10p"], 2),
              "n400", r["rate_n400"], r["count_n400"])


if __name__ == "__main__":
    main()
