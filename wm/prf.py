"""PRF modülü — referans (saf Python) ve vektörleştirilmiş (torch) sürümler.

Referans tanım SPEC.md Bölüm 3.1'de sabitlenmiştir ve DOĞRULUK
KAYNAĞIDIR. Vektörleştirilmiş sürüm bununla birebir aynı sonucu vermek
zorundadır (bkz. wm/tests/test_prf.py).

uint64 ↔ int64: torch int64'te uint64 değerleri iki-eki (two's complement)
gösterimiyle tutulur. Toplama, XOR ve çarpım mod 2^64 iki-eki bit örüntüsüyle
birebir aynıdır; yalnızca MANTIKSAL sağa kaydırma özel işlem gerektirir
(lshr64). Python tarafında ise değerler her zaman 0..2^64-1 (işaretsiz)
tutulur; torch'a geçerken to_i64 ile çevrilir.
"""

from __future__ import annotations

M = (1 << 64) - 1

# Splitmix64 sabitleri (referans tanımdan birebir)
GOLDEN = 0x9E3779B97F4A7C15
C1 = 0xBF58476D1CE4E5B9
C2 = 0x94D049BB133111EB


def to_i64(u: int) -> int:
    """uint64 Python int -> int64 iki-eki temsil (torch'a geçiş için)."""
    u &= M
    return u - (1 << 64) if u >= (1 << 63) else u


# ---------------------------------------------------------------------------
# Referans (saf Python, işaretsiz) — DOĞRULUK KAYNAĞI
# ---------------------------------------------------------------------------

def mix64(z: int) -> int:
    """uint64 splitmix64 finalizer. z 0..2^64-1."""
    z = (z + GOLDEN) & M
    z = ((z ^ (z >> 30)) * C1) & M
    z = ((z ^ (z >> 27)) * C2) & M
    return z ^ (z >> 31)


def ctx_hash(key: int, window) -> int:
    """Pencere hash'i. window: son k token id (iterable of int)."""
    h = mix64(key & M)
    for t in window:
        h = mix64(h ^ (int(t) & M))
    return h


def R(key: int, window, v: int) -> float:
    """PRF değeri, (0,1) aralığında. Referans tanım."""
    x = mix64(ctx_hash(key, window) ^ (v & M)) >> 11
    return (x + 0.5) / 2**53


# ---------------------------------------------------------------------------
# Vektörleştirilmiş sürüm (torch) — tüm sözlük için R vektörü
# ---------------------------------------------------------------------------

import torch

_I64MIN = -(1 << 63)


def _lshr64(x: torch.Tensor, s: int) -> torch.Tensor:
    """Mantıksal sağa kaydırma (uint64), int64 iki-eki temsilde.

    x >= 0 ise u = x; x < 0 ise u = x + 2^64.
    u >> s = floor(u / 2^s); x < 0 için floor((x + 2^64)/2^s)
    = floor(x / 2^s) + 2^(64-s)  (2^64 / 2^s tam olduğu için).
    Sonuç u >> s < 2^63 olduğundan int64'te her zaman temsil edilebilir.
    s >= 1 olmalı.
    """
    shifted = x >> s                      # aritmetik kaydırma
    add = (1 << (64 - s))
    correction = torch.where(x < 0,
                             torch.tensor(add, dtype=x.dtype, device=x.device),
                             torch.tensor(0, dtype=x.dtype, device=x.device))
    return shifted + correction           # int64 wraparound sonucu negatif olmaz


def _mix64_vec(z: torch.Tensor) -> torch.Tensor:
    """mix64'ün vektörleştirilmiş hali. z: int64 tensör (uint64 iki-eki)."""
    z = z + torch.tensor(to_i64(GOLDEN), dtype=z.dtype, device=z.device)  # wraparound
    z = (z ^ _lshr64(z, 30)) * torch.tensor(to_i64(C1), dtype=z.dtype, device=z.device)  # wraparound çarpım
    z = (z ^ _lshr64(z, 27)) * torch.tensor(to_i64(C2), dtype=z.dtype, device=z.device)
    return z ^ _lshr64(z, 31)


def ctx_hash_vec(key: int, window) -> torch.Tensor:
    """Vektörleştirilmiş pencere hash'i. window: token id listesi ya da
    1-D int64 tensör. Dönüş: (1,) int64 tensör."""
    h = torch.tensor(to_i64(mix64(key & M)), dtype=torch.int64)
    # mix64(key) < 2^63 olabilir de olmayabilir de; iki-eki temsil sorun değil.
    if not torch.is_tensor(window):
        window = torch.tensor([int(t) & M for t in window], dtype=torch.int64)
    for t in window.tolist():
        t_i64 = torch.tensor(to_i64(t & M), dtype=torch.int64)
        h = _mix64_vec(h ^ t_i64)
    return h


def R_vec(key: int, window, V: int, device="cpu") -> torch.Tensor:
    """Tüm sözlük (0..V-1) için R vektörü. Dönüş: float64 tensör (V,).

    Referansla birebir: x = mix64(h ^ v) >> 11 (mantıksal), R = (x+0.5)/2^53.
    """
    h = ctx_hash_vec(key, window)
    v = torch.arange(V, dtype=torch.int64, device=device)
    x = _mix64_vec(h.to(device) ^ v)
    x_lshr = _lshr64(x, 11)               # mantıksal: uint64 >> 11, < 2^53
    num = x_lshr.to(torch.float64) + 0.5  # 2^53'e kadar float64 tam temsil
    return num / float(2**53)


def R_single_vec(key: int, window, v: int, device="cpu") -> float:
    """Tek token için vektörleştirilmiş yol (bağımsızlık kontrolü için)."""
    return float(R_vec(key, window, int(v) + 1, device=device)[int(v)])