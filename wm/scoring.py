"""Skor ve p-değer modülü — SPEC.md Bölüm 3.3.

Her puanlanan token için s = -ln(1 - R). H0 altında R tekdüze => s ~ Exp(1),
N token için S ~ Gamma(N, 1). p-değeri hep logsf ile hesaplanır (taşma yok);
- log10(p) ve N birlikte raporlanır (Bölüm 7.3: kırpma yok).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import gamma as gamma_dist


def per_token_score(r: float) -> float:
    """Tek token skoru: s = -ln(1 - R). R (0,1) içinde olmalı."""
    if not (0.0 < r < 1.0):
        raise ValueError(f"R aralık dışı: {r}")
    return -math.log1p(-r)


def total_score(scores) -> float:
    """Toplam S = sum(s_t)."""
    return float(np.sum(np.asarray(scores, dtype=np.float64)))


def _gamma_logsf_analytic(S: float, N: int) -> float:
    """Gamma(N,1) üst kuyruğunun log'u için asimptotik yaklaşım.

    S çok büyükken scipy logsf -inf döndürebilir (ic underflow); bu durumda
    sf ~= S^(N-1) * exp(-S) / Gamma(N) yaklaşımı kullanılır:
    log sf ~= (N-1)*ln(S) - S - lgamma(N).
    S > N + 50*(N**0.5) bölgesinde üstel terim baskındır ve bu yaklaşım
    sf'nin log'unu doğru mertebelerde verir.
    """
    return (N - 1) * math.log(S) - S - math.lgamma(N)


def log10_p(S: float, N: int) -> float:
    """-log10(p) — S ~ Gamma(N,1) H0'ında üst kuyruk.

    logsf ile hesaplanır; çok küçük p'lerde float taşması olmaz
    (log-alanı). logsf -inf döndürürse (aşırı güçlü sinyal) asimptotik
    formül devreye girer. Gerçek değer yazılır, kırpma yapılmaz.
    """
    if N <= 0:
        raise ValueError(f"N pozitif olmalı: {N}")
    if S < 0:
        raise ValueError(f"S negatif olamaz: {S}")
    logsf = gamma_dist.logsf(S, a=N)
    if not np.isfinite(logsf):
        logsf = _gamma_logsf_analytic(S, N)
    val = -logsf / np.log(10)  # logsf negatif log verir
    return float(val)


def p_value(S: float, N: int) -> float:
    """p-değeri (0,1). log10_p'den geri çevrilir."""
    return 10.0 ** (-log10_p(S, N))


def mean_score_per_token(scores) -> float:
    """S/N — E2'de D_gen vs D_post karşılaştırmasında kullanılır."""
    a = np.asarray(scores, dtype=np.float64)
    return float(a.mean()) if a.size else 0.0