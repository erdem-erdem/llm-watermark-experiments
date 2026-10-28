"""Toy demo (CPU, no model download): watermark a fake "language model" and detect it.

    python -m examples.toy_demo

A random bigram model stands in for an LLM. We sample text three ways — with the key,
without any watermark, and with the key but checked using a wrong key — and run the real
detector from `wm/` on each. Expect a huge -log10(p) only for (watermarked, right key).
"""

from __future__ import annotations

import math
import random

from wm.detect import detect_text
from wm.prf import R

VOCAB, LENGTH, K, KEY = 400, 400, 2, 0x1234ABCD5678EF01
WRONG_KEY = 0x0F0F0F0F0F0F0F0F


def toy_dist(prev: int, rng_seed: int = 7) -> list[float]:
    """A fixed, peaky next-token distribution that depends only on the previous token."""
    rng = random.Random(rng_seed * 1_000_003 + prev)
    w = [math.exp(rng.gauss(0, 1.6)) for _ in range(VOCAB)]
    s = sum(w)
    return [x / s for x in w]


def sample(watermark_key: int | None, seed: int) -> list[int]:
    rng = random.Random(seed)
    ids = [rng.randrange(VOCAB), rng.randrange(VOCAB)]       # arbitrary first tokens
    seen = set()
    for _ in range(LENGTH):
        p = toy_dist(ids[-1])
        window = tuple(ids[-K:])
        if watermark_key is not None and window not in seen:
            # Gumbel-max / exponential race:  argmax_v R_v ** (1 / p_v)
            nxt = max(range(VOCAB), key=lambda v: math.log(R(watermark_key, window, v)) / p[v])
        else:                                                 # ordinary sampling (also for repeated windows)
            nxt = rng.choices(range(VOCAB), weights=p)[0]
        seen.add(window)
        ids.append(nxt)
    return ids[2:]


def report(name: str, ids: list[int], key: int) -> None:
    res = detect_text(prompt_ids=[0, 0], answer_ids=ids, key=key, k=K)
    print(f"{name:<38} N={res.N:<4} -log10(p) = {res.neglog10p:8.2f}")


if __name__ == "__main__":
    marked = sample(KEY, seed=1)
    plain = sample(None, seed=2)
    report("watermarked text, right key", marked, KEY)
    report("watermarked text, WRONG key", marked, WRONG_KEY)
    report("ordinary text, right key", plain, KEY)
