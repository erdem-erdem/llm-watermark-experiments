"""Okuma modu puanlayıcı testi (CPU): vektörleştirilmiş score_pairs, wm.prf.R ile
birebir; select_pairs kuralları (a)-(d)'ye uygun.

  python -m wm.tests.test_reading_scorer
"""
import random

import numpy as np

from wm.prf import R
from wm.reading_mode import score_pairs, select_pairs
from wm.scoring import per_token_score


def test_vectorized_equals_reference():
    rng = random.Random(7)
    n = 3000
    W = np.asarray([[rng.randrange(0, 248000) for _ in range(2)] for _ in range(n)])
    P = np.asarray([rng.randrange(0, 248000) for _ in range(n)])
    for key in (13567654914097149964, 0, 2 ** 64 - 1, 9538258013827939100):
        got = score_pairs(key, W, P)
        ref = np.asarray([per_token_score(R(key, tuple(int(x) for x in w), int(p)))
                          for w, p in zip(W, P)])
        assert np.allclose(got, ref, rtol=0, atol=1e-12), f"anahtar {key}"
    print("ok  score_pairs == wm.prf.R (4 anahtar x 3000 çift)")


def test_selection_rules():
    # 3 örnek; pencere (1,2) örnek 0'da iki kez geçer (c), örnek 1'de aynı
    # (pencere,tahmin) tekrarlanır (d), (9,9) prompt penceresi (b)
    windows = [(1, 2), (1, 2), (3, 4), (1, 2), (3, 4), (9, 9)]
    preds = [5, 6, 7, 5, 8, 1]
    samples = np.asarray([(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)])
    W, P, owner = select_pairs(windows, preds, samples, 2,
                               prompt_overlap_windows={(9, 9)})
    got = [(tuple(w), int(p), int(o)) for w, p, o in zip(W, P, owner)]
    # (1,2)->5 [ö0]; (1,2)->6 atlanır (c); (3,4)->7 [ö0]; ö1: (1,2)->5 atlanır (d);
    # (3,4)->8 puanlanır (farklı tahmin); (9,9) atlanır (b)
    assert got == [((1, 2), 5, 0), ((3, 4), 7, 0), ((3, 4), 8, 1)], got
    # skip_fn: ö0'ın t=2 atlanır
    W, P, owner = select_pairs(windows, preds, samples, 2,
                               skip_fn=lambda s, t: s == 0 and t == 2,
                               prompt_overlap_windows={(9, 9)})
    assert (int(P[0]), int(owner[0])) == (6, 0), (P, owner)
    print("ok  (a)-(d) seçim kuralları + skip_fn")


if __name__ == "__main__":
    test_vectorized_equals_reference()
    test_selection_rules()
    print("BÜTÜN OKUMA MODU TESTLERİ GEÇTİ")
