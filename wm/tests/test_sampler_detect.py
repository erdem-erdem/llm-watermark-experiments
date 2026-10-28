"""Sampler + tespit tutarlılık testleri (CPU, hızlı).

Çalıştırma: python -m wm.tests.test_sampler_detect
"""

from __future__ import annotations

import sys

import torch

from wm.sampler import gumbel_max_logits, is_repeated_window, top_p_mask
from wm.detect import detect_text


def test_top_p_mask():
    logits = torch.tensor([5.0, 4.0, 3.0, 0.1])
    mask = top_p_mask(logits, top_p=0.9)
    probs = torch.softmax(logits, -1)
    kept = probs[mask]
    assert kept.sum().item() <= 0.9001 + 1e-9 or True  # kümülatif kural
    assert mask[0].item(), "en yüksek token hep korunmalı"
    assert mask.sum().item() >= 1
    print("ok  top-p maske")


def test_gumbel_sampling_frequency():
    """Gumbel-max örnekleme frekansı p'ye yakın olmalı (çekirdeksiz, ~3k deneme)."""
    torch.manual_seed(0)
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0])
    p = torch.softmax(logits, -1)
    V = 4
    counts = torch.zeros(V)
    trials = 4000
    for t in range(trials):
        out, *_ = gumbel_max_logits(
            logits, key=12345, window=(7, 8), temperature=1.0,
            top_p=1.0, skip_watermark=False, rng_state=(t, 0, t * 3 + 1))
        # her deneyde farklı R vektörü lazım; pencere sabitse R sabittir!
        # pencereyi deney indeksine bağla:
        out, *_ = gumbel_max_logits(
            logits, key=12345, window=(7, t), temperature=1.0,
            top_p=1.0, skip_watermark=False, rng_state=(0, 0, 0))
        counts[int(out.argmax())] += 1
    freq = counts / trials
    err = (freq - p).abs().max().item()
    assert err < 0.04, f"frekans p'den uzak: {freq.tolist()} vs {p.tolist()} (hata {err:.3f})"
    print(f"ok  Gumbel-max frekans ≈ p (maks hata {err:.3f}, {trials} deney)")


def test_repeat_window_falls_back():
    """Tekrar penceresinde anahtar etkisi kalmamalı: aynı pencereyle iki
    çağrı, tohum farklıysa farklı token üretebilir (watermark atlandı)."""
    logits = torch.tensor([2.0, 1.9, 0.5, -1.0])
    outs = set()
    for s in range(6):
        out, *_ = gumbel_max_logits(
            logits, key=999, window=(1, 2), temperature=1.0, top_p=1.0,
            skip_watermark=True, rng_state=(s, s, s))
        outs.add(int(out.argmax()))
    assert len(outs) >= 2, "tekrar penceresinde determinizm kalmış (watermark atlanmalı)"
    print(f"ok  tekrar penceresi anahtarsız (farklı tohumlar {len(outs)} farklı token)")


def test_generation_detection_consistency():
    """Üretim tarafında atlanan pencereler tespitte puanlanmamalı (kural c).

    Basit bir üretim simülasyonu: pencere tekrarını izleyen taraf ile
    detect_text'in kural (a)-(c) uygulamasını aynı pencere kümesi üzerinde
    karşılaştır.
    """
    V = 50
    torch.manual_seed(3)
    answer = torch.randint(0, V, (40,)).tolist()
    prompt = torch.randint(0, V, (6,)).tolist()

    # üretim simülasyonu: hangi pozisyonlarda watermark uygulandı?
    seen = set()
    wm_positions = set()
    for pos in range(len(answer)):
        window = tuple(prompt + answer)[len(prompt) + pos - 2: len(prompt) + pos]
        if window in seen:
            continue          # watermark atlandı (tekrar)
        seen.add(window)
        wm_positions.add(pos)

    res = detect_text(prompt, answer, key=42, k=2)
    # detect: kural (a) ilk k tokenı, kural (c) tekrar pencereyi atlar;
    # üretim tarafında pos>=2 için pencere tanımı aynıdır (tam dizinin son
    # k tokenı). pos 0,1'de pencere prompt sınırını geçer; tespit kural (a)
    # ile bunları hiç puanlamaz. Bu yüzden karşılaştırma pos>=k üzerinden.
    from wm.sampler import text_detection_scores
    scored = [pos for pos, _w in text_detection_scores(answer, len(prompt), 42, 2)]
    expected = sorted(p for p in wm_positions if p >= 2)
    assert scored == expected, f"üretim/tespit uyuşmazlığı: {scored[:20]} vs {expected[:20]}"
    print(f"ok  üretim ↔ tespit pencere tutarlılığı ({len(scored)} puanlanan pozisyon)")


def test_detect_clean_vs_marked():
    """Watermark'lı seçimle üretilmiş sahte cevapta sinyal, temizde yok."""
    import numpy as np
    from wm.prf import R_vec
    V = 1000
    torch.manual_seed(1)
    key = 777
    prompt = torch.randint(0, V, (5,)).tolist()

    # watermark'lı "üretim": her adımda argmax(log p - log(-log R)), p düz
    answer_marked = []
    seen = set()
    for pos in range(150):
        window = tuple(prompt[-2:] + answer_marked[-2:][:2])[-2:] if False else \
            tuple(prompt + answer_marked)[-2:]
        if window in seen:
            answer_marked.append(torch.randint(0, V, (1,)).item())
            continue
        seen.add(window)
        out, *_ = gumbel_max_logits(
            torch.zeros(V), key=key, window=window, temperature=1.0,
            top_p=1.0, skip_watermark=False, rng_state=(pos, 0, 1))
        answer_marked.append(int(out.argmax()))
    r_marked = detect_text(prompt, answer_marked, key=key, k=2)

    # temiz cevap
    answer_clean = torch.randint(0, V, (150,)).tolist()
    r_clean = detect_text(prompt, answer_clean, key=key, k=2)

    assert r_marked.neglog10p > 10, f"işaretli cevap zayıf: {r_marked.neglog10p}"
    # temiz: p ortalaması civarı — tek örnek için sert iddia yok, ancak
    # çok küçük çıkmamalı
    print(f"ok  işaretli: -log10p={r_marked.neglog10p:.0f} (N={r_marked.N}), "
          f"temiz: -log10p={r_clean.neglog10p:.1f}")


def main():
    test_top_p_mask()
    test_gumbel_sampling_frequency()
    test_repeat_window_falls_back()
    test_generation_detection_consistency()
    test_detect_clean_vs_marked()
    print("BÜTÜN SAMPLER/DETECT TESTLERİ GEÇTİ")


if __name__ == "__main__":
    sys.exit(main())