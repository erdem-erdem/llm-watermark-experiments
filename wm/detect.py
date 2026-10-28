"""Tespit modülü — SPEC.md Bölüm 3.3-3.4.

Metin tespiti: puanlanan token = metindeki token. Tekilleştirme kuralları
(a)-(d) uygulanır; (d) tüm veri kümesi boyunca global (pencere, token)
ikilisi kümesiyle yürütülür.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wm.prf import R
from wm.scoring import log10_p, per_token_score
from wm.sampler import text_detection_scores


@dataclass
class DetectionResult:
    N: int                     # puanlanan token sayısı
    neglog10p: float           # -log10(p)
    skipped_positions: list    # tekilleştirmede elenen pozisyonlar
    scores: np.ndarray         # puanlanan tokenların skorları


def detect_text(
    prompt_ids: list[int],
    answer_ids: list[int],
    key: int,
    k: int = 2,
    global_pairs: set | None = None,
    return_scores: bool = False,
) -> DetectionResult:
    """Tek örneği metin modunda puanla.

    global_pairs: aynı test içinde tüm veri kümesi boyunca paylaşılan
    (window, token) ikilisi kümesi (kural d). None verilirse iç küme
    oluşturulur (tek örneklik test).
    """
    if global_pairs is None:
        global_pairs = set()
    full = list(prompt_ids) + list(answer_ids)
    plen = len(prompt_ids)
    scores = []
    skipped = []
    for pos, window in text_detection_scores(answer_ids, plen, key, k):
        tok = int(answer_ids[pos])
        if (window, tok) in global_pairs:      # (d) global
            skipped.append(pos)
            continue
        global_pairs.add((window, tok))
        # (b) pencere prompt'ta geçmiyor mu? pencere prompt+cevap tam
        # dizisinden alındığı için pos>=k ile cevap içi pencereler prompt
        # ile örtüşemez; prompt sınırındaki pencere zaten (a) ile atlanır.
        scores.append(per_token_score(R(key, window, tok)))
    S = float(np.sum(scores)) if scores else 0.0
    N = len(scores)
    return DetectionResult(
        N=N, neglog10p=log10_p(S, N) if N else float("-inf"),
        skipped_positions=skipped,
        scores=np.asarray(scores, dtype=np.float64) if return_scores else np.asarray([]))


def detect_dataset(records, key: int, k: int = 2, chunk_aware: bool = True):
    """Veri kümesi (JSONL kayıt listesi) boyunca tekilleştirilmiş metin tespiti.

    Kural (d): (window, token) ikilisi TÜM veri kümesi boyunca bir kez
    puanlanır. chunk_aware=True ise kayıtlardaki chunk_bounds parçalarının
    ilk k tokenı da atlanır (E1 parçalı yeniden yazım).

    Dönüş: (DetectionResult, ornek_bazina_p_listesi)
    """
    global_pairs: set = set()
    all_scores = []
    per_sample = []
    for rec in records:
        answer_ids = rec["answer_ids"]
        bounds = rec.get("chunk_bounds") if chunk_aware else None
        local_seen: set = set()
        s_here = []
        for pos, window in text_detection_scores(
                answer_ids, len(rec["prompt_ids"]), key, k):
            # parça sınırı: pos parçanın ilk k tokenı ise atla (kural a gibi)
            if bounds and _in_chunk_head(pos, bounds, k):
                continue
            tok = int(answer_ids[pos])
            if window in local_seen or (window, tok) in global_pairs:
                continue
            local_seen.add(window)
            global_pairs.add((window, tok))
            s_here.append(per_token_score(R(key, window, tok)))
        per_sample.append((len(s_here), log10_p(sum(s_here), len(s_here)) if s_here
                           else float("-inf")))
        all_scores.extend(s_here)
    S = float(np.sum(all_scores))
    N = len(all_scores)
    result = DetectionResult(
        N=N, neglog10p=log10_p(S, N) if N else float("-inf"),
        skipped_positions=[], scores=np.asarray(all_scores))
    return result, per_sample


def _in_chunk_head(pos: int, bounds, k: int) -> bool:
    """pos, herhangi bir parçanın ilk k tokenı içinde mi? bounds: [(start,end)]."""
    for start, end in bounds:
        if start <= pos < start + k:
            return True
        if pos >= end:
            continue
    return False


def detect_prefix_curves(
    answer_ids: list[int],
    prompt_len: int,
    key: int,
    k: int = 2,
    n_points: int = None,
    ns: list[int] = None,
):
    """İlk n puanlanan token üzerinden -log10(p) eğrisi (E2 adım 2).

    ns verilmemişse n_points nokta eşit aralıklı seçilir.
    Puanlanabilir pozisyonlar sabit; eğri, ilk n puanlanan tokenın
    kümülatif skoru üzerinden hesaplanır.
    """
    scores, positions = _score_positions(answer_ids, prompt_len, key, k)
    if ns is None:
        maxn = len(scores)
        ns = sorted({max(1, int(round(n * maxn)))
                     for n in np.linspace(0.05, 1.0, n_points or 10)})
    rows = []
    for n in ns:
        s = float(np.sum(scores[:n]))
        rows.append({"n": int(n), "N": int(n), "neglog10p": log10_p(s, n)})
    return rows


def _score_positions(answer_ids, prompt_len, key, k):
    """Tek örnekte kural (a)-(c) ile skorlar (global (d) yok — eğri için)."""
    local_seen: set = set()
    scores = []
    positions = []
    for pos, window in text_detection_scores(answer_ids, prompt_len, key, k):
        if window in local_seen:
            continue
        local_seen.add(window)
        scores.append(per_token_score(R(key, window, int(answer_ids[pos]))))
        positions.append(pos)
    return scores, positions