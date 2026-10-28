"""Okuma modu (radyoaktivite testi) — SPEC.md Bölüm 3.5.

Watermark'lı her örnek için: sohbet şablonlu prompt + cevap ID'leri modele
tek ileri geçiş (teacher forcing) ile verilir; her pozisyonda modelin
tahmini ŷ_t = argmax logits_t alınır. Pencere METNDEKİ gerçek önekin son
k tokenıdır (modelin kendi tahminleriyle değil). Skor R(key, pencere, ŷ_t).

Tahminler bir kez hesaplanıp preds/*.npz olarak saklanır; farklı anahtarlar
ile yeniden puanlama bu dosya üzerinden yapılır (neredeyse bedava).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wm.prf import R
from wm.scoring import log10_p, per_token_score


def compute_predictions(model, records, k: int = 2, device="cuda",
                        max_len: int = 2048, batch_size: int = 1,
                        out_path: Path | None = None, log_fn=print):
    """Tüm örnekler için argmax tahminlerini ve pencereleri hesaplar.

    records: JSONL kayıtları (prompt_ids, answer_ids gerekli).
    out_path: .npz (pencere dizileri ve tahminler). Yoksa yalnızca döndürür.
    """
    model.eval()
    model.to(device)
    all_win = []
    all_pred = []
    all_sample = []   # örnek ve cevap-içi pozisyon
    with torch.no_grad():
        for idx, rec in enumerate(records):
            prompt_ids = list(rec["prompt_ids"])
            answer_ids = list(rec["answer_ids"])
            ids = (prompt_ids + answer_ids)[:max_len]
            if len(answer_ids) < k:
                continue
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attn = torch.ones_like(input_ids)
            logits = model(input_ids=input_ids, attention_mask=attn).logits[0]
            # logits[t] pozisyon t+1'i tahmin eder (teacher forcing)
            # cevaptaki pozisyonlar: plen-1 .. plen+len(answer_ids)-2 (tahmin
            # edilebilen son pozisyon: len(ids)-2)
            plen = len(prompt_ids)
            start = plen - 1
            end = len(ids) - 1
            for pos in range(start, end):
                t_ans = pos - (plen - 1)          # cevap içindeki indeks (0 tabanlı)
                if t_ans < k:
                    continue                      # kural (a)
                window = tuple(ids[pos - k + 1: pos + 1])  # metindeki gerçek ön ek
                pred = int(torch.argmax(logits[pos]).item())
                all_win.append(window)
                all_pred.append(pred)
                all_sample.append((idx, t_ans))
            if (idx + 1) % 200 == 0:
                log_fn(f"  preds: {idx + 1}/{len(records)} örnek")
    preds = np.asarray(all_pred, dtype=np.int64)
    samples = np.asarray(all_sample, dtype=np.int64)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # pencereler k-sabit olduğundan düz dizi olarak saklanır
        win_arr = np.asarray([t for w in all_win for t in w], dtype=np.int64)
        np.savez_compressed(
            out_path, preds=preds, samples=samples, windows=win_arr,
            k=k, V=np.asarray([0]))  # V bilgi amaçlı ayrıca saklanabilir
    return all_win, all_pred, samples


def load_predictions(npz_path: Path):
    """Saklanmış tahmin dosyasını okur. Dönüş: (windows_list, preds, samples, k)."""
    z = np.load(npz_path, allow_pickle=False)
    k = int(z["k"])
    win_arr = z["windows"].reshape(-1, k).tolist()
    return [tuple(w) for w in win_arr], z["preds"], z["samples"], k


def rescore_predictions(windows, preds, key: int, k: int = 2,
                        global_pairs: set | None = None,
                        prompt_overlap_windows: set | None = None):
    """Saklanmış tahminleri bir anahtarla puanlar (kural a-d).

    prompt_overlap_windows: kural (b) için — bu örnek grubunda promptlarda
    geçen pencereler (dışarıdan verilir; her örnek grubunun kendi
    promptlarından hazırlanır).
    """
    if global_pairs is None:
        global_pairs = set()
    if prompt_overlap_windows is None:
        prompt_overlap_windows = set()
    scores = []
    seen_here: set = set()
    for window, pred in zip(windows, preds):
        pred = int(pred)
        if window in prompt_overlap_windows:   # (b)
            continue
        if window in seen_here:                # (c)
            continue
        if (window, pred) in global_pairs:     # (d)
            continue
        seen_here.add(window)
        global_pairs.add((window, pred))
        scores.append(per_token_score(R(key, window, pred)))
    S = float(np.sum(scores)) if scores else 0.0
    N = len(scores)
    return {"N": N, "neglog10p": log10_p(S, N) if N else float("-inf")}


def collect_prompt_windows(records, k: int = 2) -> set:
    """Bir örnek grubundaki promptlarda geçen pencere kümesi (kural b)."""
    out: set = set()
    for rec in records:
        ids = list(rec["prompt_ids"])
        for i in range(k, len(ids) + 1):
            out.add(tuple(int(t) for t in ids[i - k:i]))
    return out


def sample_p_values_for_calibration(windows, preds, key: int, samples, n_parts: int = 50):
    """E0 adım 4: örnekleri eşit parçalara böl, parça başına p-değeri."""
    # örnek indeksine göre grupla
    by_sample: dict = {}
    for i, (idx, _t) in enumerate(samples):
        by_sample.setdefault(int(idx), []).append(i)
    idxs = sorted(by_sample.keys())
    part_size = max(1, len(idxs) // n_parts)
    parts = [idxs[i:i + part_size] for i in range(0, len(idxs), part_size)]
    pvals = []
    for part in parts:
        w_list = []
        p_list = []
        for idx in part:
            for i in by_sample[idx]:
                w_list.append(windows[i])
                p_list.append(int(preds[i]))
        r = rescore_predictions(w_list, p_list, key)
        if r["N"] > 0:
            pvals.append(10.0 ** (-r["neglog10p"]))
    return pvals

def _mix64_np(z):
    """uint64 numpy dizisi üzerinde splitmix64 (mod 2^64; wm.prf.mix64 ile birebir)."""
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def score_pairs(key: int, W: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(pencere, tahmin) çiftleri için puanlar −ln(1−R) — vektörleştirilmiş.

    W: (n,k) token id'leri, P: (n,) tahmin. wm.prf.R ile birebir aynı sonuç
    (test: wm/tests/test_reading_scorer.py). Python referansından ~100x hızlı.
    """
    if len(P) == 0:
        return np.zeros(0, dtype=np.float64)
    W = np.asarray(W, dtype=np.uint64).reshape(len(P), -1)
    P = np.asarray(P, dtype=np.uint64)
    h = np.full(len(P), _mix64_np(np.asarray([key & ((1 << 64) - 1)], dtype=np.uint64))[0],
                dtype=np.uint64)
    for j in range(W.shape[1]):
        h = _mix64_np(h ^ W[:, j])
    x = _mix64_np(h ^ P) >> np.uint64(11)
    r = (x.astype(np.float64) + 0.5) / float(2 ** 53)
    return -np.log1p(-r)


def select_pairs(windows, preds, samples, k: int = 2, include=None,
                 prompt_overlap_windows: set | None = None, skip_fn=None):
    """Kural (a)-(d) ile puanlanacak (pencere, tahmin) çiftlerini seç.

    Seçim anahtardan BAĞIMSIZDIR → bir kez seç, her anahtarla `score_pairs`.
    (a) ilk k token compute_predictions'ta zaten yok; (b) prompt pencereleri;
    (c) aynı cevapta pencere bir kez (örnek başına sıfırlanır); (d) (pencere,
    tahmin) bu test boyunca bir kez; skip_fn(sample_idx, t_ans): E1 chunk başları.
    include: örnek indekslerinin sırası. Dönüş: W (n,k), P (n,), owner (n,)
    """
    by: dict = {}
    for i, (idx, t) in enumerate(samples):
        by.setdefault(int(idx), []).append((int(t), i))
    order = sorted(by) if include is None else [int(x) for x in include if int(x) in by]
    overlap = prompt_overlap_windows or set()
    global_pairs: set = set()
    Ws, Ps, owner = [], [], []
    for sidx in order:
        local_seen: set = set()
        for t, i in sorted(by[sidx]):
            if skip_fn is not None and skip_fn(sidx, t):
                continue
            w, p = windows[i], int(preds[i])
            if w in overlap:
                continue
            if w in local_seen or (w, p) in global_pairs:
                continue
            local_seen.add(w)
            global_pairs.add((w, p))
            Ws.append(w)
            Ps.append(p)
            owner.append(sidx)
    W = np.asarray(Ws, dtype=np.int64).reshape(len(Ps), k)
    return W, np.asarray(Ps, dtype=np.int64), np.asarray(owner, dtype=np.int64)


def score_reading(windows, preds, samples, key: int, k: int = 2,
                  include=None, prompt_overlap_windows: set | None = None,
                  skip_fn=None):
    """Okuma modu — AGENTS 3.4'e TAM uyumlu puanlama (select_pairs + score_pairs).

    `rescore_predictions` ise (c)'yi TÜM küme boyunca uygular (seen_here hiç
    sıfırlanmaz) — spesifikasyondan daha sıkı; geriye dönük karşılaştırma için
    korunur, yeni sonuçlarda BU işlev kullanılır.
    Dönüş: (scores: np.ndarray puanlanma sırasıyla, owner: np.ndarray)
    """
    W, P, owner = select_pairs(windows, preds, samples, k, include,
                               prompt_overlap_windows, skip_fn)
    return score_pairs(key, W, P), owner


def neglog10p_of(scores) -> float:
    n = len(scores)
    return log10_p(float(np.sum(scores)), n) if n else float("nan")
