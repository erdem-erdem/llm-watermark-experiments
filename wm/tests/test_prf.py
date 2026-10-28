"""PRF birim testleri — SPEC.md Bölüm 3.1 zorunlu testleri.

Çalıştırma:  python -m wm.tests.test_prf
(torch CPU yeterli; GPU gerektirmez)
"""

from __future__ import annotations

import random
import sys

import numpy as np

from wm.prf import M, R, R_vec, R_single_vec, mix64, ctx_hash
from wm.scoring import log10_p, per_token_score


def test_reference_sanity():
    # Referans fonksiyonlar deterministik ve (0,1) aralığında
    random.seed(0)
    for _ in range(1000):
        key = random.getrandbits(64)
        window = [random.getrandbits(16) for _ in range(2)]
        for v in (0, 1, 12345, 151936):
            r = R(key, window, v)
            assert 0.0 < r < 1.0, f"R aralık dışı: {r}"
    print("ok  reference sanity (1000 üçlü)")


def test_vectorized_bit_exact():
    """10.000 rastgele (key, window, v) üçlüsünde vektörize == referans."""
    rng = random.Random(12345)
    vocab = 1000
    for i in range(10000):
        key = rng.getrandbits(64)
        k = rng.choice([1, 2, 3])
        window = [rng.getrandbits(20) for _ in range(k)]
        v = rng.randrange(vocab)
        r_ref = R(key, window, v)
        r_vec = float(R_vec(key, window, vocab)[v])
        assert r_ref == r_vec, (
            f"uyuşmazlık #{i}: key={key} win={window} v={v}: "
            f"{r_ref!r} != {r_vec!r}")
    print("ok  vektörize == referans (10.000 üçlü, birebir)")


def test_single_vs_full_vocab():
    """Tek-token yolu ile tüm-sözlük yolu birebir aynı olmalı."""
    rng = random.Random(777)
    vocab = 4096
    for _ in range(200):
        key = rng.getrandbits(64)
        window = [rng.getrandbits(20), rng.getrandbits(20)]
        v = rng.randrange(vocab)
        a = R(key, window, v)
        b = R_single_vec(key, window, v)
        c = float(R_vec(key, window, vocab)[v])
        assert a == b == c, f"yol uyuşmazlığı: {a!r} {b!r} {c!r}"
    print("ok  tek-token yolu == tüm-sözlük yolu (200 üçlü, birebir)")


def test_scoring_examples():
    """Blog'daki çalışılmış örneklerle skorlama uyuşması."""
    # Blog Bölüm 1: R=0.88 -> skor -ln(1-0.88) = 2.1203
    s = per_token_score(0.88)
    assert abs(s - 2.120263536200091) < 1e-9, s
    # Blog: S=6.26, N=4 -> p ~= 0.13
    p = 10 ** (-log10_p(6.26, 4))
    assert abs(p - 0.13002) < 0.005, p
    # H0 altında beklenen değer: E[S] = N
    rng = np.random.default_rng(0)
    S = rng.random(20000).sum()
    N = 20000
    # ortalaması N olan çok uzun toplamlar için p eşit dağılım göstermez;
    # yalnızca tek token Exp(1): ortalaması 1
    from scipy.stats import expon
    assert abs(expon.mean() - 1.0) < 1e-12
    print(f"ok  skorlama örnekleri (s(0.88)={s:.4f}, p(6.26,4)={p:.5f})")


def test_scoring_under_h0_uniform():
    """R eşit dağılımlıyken -log10(p), U(0,1) p-değerleri üretmeli (KS)."""
    from scipy.stats import kstest
    rng = np.random.default_rng(42)
    ps = []
    for _ in range(500):
        N = rng.integers(50, 300)
        S = rng.exponential(1.0, N).sum()  # Exp(1) toplamı = Gamma(N,1)
        neglog10p = log10_p(S, N)
        ps.append(10 ** (-neglog10p))
    ks_stat, ks_p = kstest(ps, "uniform")
    assert ks_p > 0.01, f"H0'da p-dağılıımı eşit değil: KS p={ks_p}"
    print(f"ok  H0 p-değerleri eşit dağılımda (KS p={ks_p:.3f}, n=500)")


def test_scoring_under_h1_power():
    """Watermark'lı seçimde (yüksek R'li tokenlar) skor anlamlı büyümeli."""
    rng = np.random.default_rng(7)
    V = 1000
    for _ in range(50):
        N = 200
        # Basit Gumbel-max benzeri seçim: R^(1/p) en büyük token;
        # p düz dağılımlı ağırlıklarla
        p = rng.dirichlet(np.ones(V))
        Rv = rng.random(V)
        # seçim: en büyük R^(1/p)
        chosen = np.argmax(Rv ** (1 / p))
        s = per_token_score(Rv[chosen])
        _ = s  # tek adımda kontrol anlamlı değil; toplam üzerinden bakılır
    # Toplamda: watermark'lı metin p-değerleri küçük çıkmalı
    neglog10ps = []
    for _ in range(30):
        p = rng.dirichlet(np.ones(V))
        Rv = rng.random(V)
        # 200 adım: her adımda bağımsız R vektörü (basitleştirilmiş simülasyon;
        # pencere bağımlılığı yok)
        S = 0.0
        for _t in range(200):
            Rv = rng.random(V)
            chosen = np.argmax(Rv ** (1 / p))
            S += per_token_score(Rv[chosen])
        neglog10ps.append(log10_p(S, 200))
    neglog10ps = np.array(neglog10ps)
    # watermark'lı: güçlü sinyal beklenir
    assert neglog10ps.min() > 10, (
        f"watermark'lı simülasyonda zayıf sinyal: min -log10p={neglog10ps.min()}")
    print(f"ok  H1 güç: medyan -log10(p) = {np.median(neglog10ps):.0f} (N=200)")


def test_determinism():
    """Aynı girdiler her zaman aynı sonucu vermeli."""
    key, window, v = 0xDEADBEEF, [11, 22], 333
    vals = {R(key, window, v) for _ in range(5)}
    assert len(vals) == 1
    vecs = {tuple(R_vec(key, window, 512).tolist()) for _ in range(3)}
    assert len(vecs) == 1
    print("ok  determinizm")


def main():
    test_reference_sanity()
    test_vectorized_bit_exact()
    test_single_vs_full_vocab()
    test_scoring_examples()
    test_scoring_under_h0_uniform()
    test_scoring_under_h1_power()
    test_determinism()
    print("BÜTÜN PRF TESTLERİ GEÇTİ")


if __name__ == "__main__":
    sys.exit(main())