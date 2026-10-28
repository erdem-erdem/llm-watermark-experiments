"""Skora göre renklendirilmiş token metni çizici ([GÖRSEL 1.3] stili: açık krem → turuncu).

Eksen birimi PUNTO (1/72 inç); y aşağı doğru artar. Kelime sınırından satır kaydırır.
"""

from __future__ import annotations

from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from analyze.blog_style import DPI, NAVY, ORANGE

LOW = "#FFF3E2"


def tok_color(s, vmax=4.0):
    if s is None:
        return None
    t = max(0.0, min(1.0, float(s) / vmax))
    a, b = to_rgb(LOW), to_rgb(ORANGE)
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


class Measurer:
    def __init__(self, fig):
        self.fig = fig
        self.cache = {}

    def w(self, s, fs, bold=False):
        k = (s, fs, bold)
        if k not in self.cache:
            t = self.fig.text(0, 0, s, fontsize=fs, fontweight="bold" if bold else "normal")
            r = self.fig.canvas.get_renderer()
            self.cache[k] = t.get_window_extent(r).width * 72.0 / self.fig.dpi
            t.remove()
        return self.cache[k]


def to_words(tokens):
    """[{'piece','score'}] → kelimeler: [{'sp': bool, 'pieces': [(text, score)]}] ve satır sonu işaretleri."""
    words, cur, pending = [], None, False

    def flush():
        nonlocal cur
        if cur and cur["pieces"]:
            words.append(cur)
        cur = None

    for tk in tokens:
        parts = tk["piece"].split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                flush()
                words.append("NL")
                pending = False
            if part == "":
                continue
            lead = part.startswith(" ")
            txt = part.lstrip(" ")
            if not txt:
                flush()
                pending = True
                continue
            if lead:
                flush()
            if cur is None:
                cur = {"sp": lead or pending, "pieces": []}
                pending = False
            cur["pieces"].append((txt, tk["score"]))
    flush()
    return words


def plain_tokens(text):
    out = []
    for li, line in enumerate(text.split("\n")):
        if li > 0:
            out.append({"piece": "\n", "score": None})
        for wi, w in enumerate(line.split(" ")):
            if w:
                out.append({"piece": (" " if wi else "") + w, "score": None})
    return out


def layout(m, tokens, width, fs, line_h):
    """→ (öğeler, yükseklik). öğe: (x, y, text, score)."""
    items, x, y = [], 0.0, 0.0
    space = m.w("a a", fs) - m.w("aa", fs)
    for w in to_words(tokens):
        if w == "NL":
            x, y = 0.0, y + line_h
            continue
        ww = sum(m.w(t, fs) for t, _ in w["pieces"])
        sp = space if (w["sp"] and x > 0) else 0.0
        if x > 0 and x + sp + ww > width:
            x, y, sp = 0.0, y + line_h, 0.0
        x += sp
        for t, s in w["pieces"]:
            wt = m.w(t, fs)
            items.append((x, y, t, s, wt))
            x += wt
    return items, y + line_h


def draw(ax, items, x0, y0, fs, line_h, color=NAVY, alpha_bg=1.0):
    for x, y, t, s, wt in items:
        c = tok_color(s)
        if c is not None:
            ax.add_patch(Rectangle((x0 + x - 0.6, y0 + y + 1.0), wt + 1.2, line_h - 1.5, fc=c, ec="none", alpha=alpha_bg, zorder=1))
        ax.text(x0 + x, y0 + y + 1.5, t, fontsize=fs, color=color, va="top", ha="left", zorder=2)


def text_height(m, tokens, width, fs, line_h):
    return layout(m, tokens, width, fs, line_h)[1]
