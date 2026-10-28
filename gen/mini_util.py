"""Mini denemeler için ortak yardımcılar.

token_pieces : byte-düzeyli BPE token'larını OKUNUR metin parçalarına çevirir
               (convert_ids_to_tokens 'Ã¼', 'Ġ' gibi ham baytlar verir).
scored_tokens: bir cevabın puanlanan tokenları (kural a-c) ve skorları.
cum_curve    : puanlanan tokenlar üzerinde kümülatif −log10 p eğrisi.
"""

from __future__ import annotations

import numpy as np

from wm.prf import R
from wm.sampler import text_detection_scores
from wm.scoring import log10_p, per_token_score


def token_pieces(tok, ids):
    """Her token için metin parçası. Çok baytlı karakter iki tokena bölünmüşse
    parça, karakter tamamlandığında SON token'a yazılır (öncekiler boş)."""
    pieces, prev = [], ""
    for i in range(len(ids)):
        cur = tok.decode(ids[: i + 1], skip_special_tokens=False,
                         clean_up_tokenization_spaces=False)
        if cur.endswith("\ufffd"):
            pieces.append("")
            continue
        pieces.append(cur[len(prev):] if cur.startswith(prev) else cur)
        prev = cur
    return pieces


def scored_tokens(answer_ids, prompt_len, key, k=2):
    """[(pos, score)] — kural (a) ilk k token, (c) cevap içi tekrar eden pencere."""
    seen, out = set(), []
    for pos, window in text_detection_scores(answer_ids, prompt_len, key, k):
        if window in seen:
            continue
        seen.add(window)
        out.append((pos, per_token_score(R(key, window, int(answer_ids[pos])))))
    return out


def cum_curve(scores, points=None):
    """Kümülatif −log10 p: [(n, neglog10p)] — points None ise her token."""
    sc = np.asarray(scores, dtype=float)
    cs = np.cumsum(sc)
    ns = range(1, len(sc) + 1) if points is None else [n for n in points if n <= len(sc)]
    return [(int(n), log10_p(float(cs[n - 1]), int(n))) for n in ns]


def summarize(scores):
    n = len(scores)
    return {"N": n, "neglog10p": log10_p(float(np.sum(scores)), n) if n else float("nan")}
