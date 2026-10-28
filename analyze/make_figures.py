"""Rebuild every figure (TR + EN) from the committed `results/` — CPU only, ~1 minute.

    python -m analyze.make_figures                # both languages
    python -m analyze.make_figures --lang en      # one language

Writes figures/<lang>/main/*.png|svg (the 9 headline figures) and figures/<lang>/extra/*.png|svg
(per-stage diagnostics). Each figure module can also be run on its own, e.g.
`python -m analyze.blog_figures --lang en E0 E1_1`.
"""

from __future__ import annotations

import subprocess
import sys

MODULES = ["analyze.blog_figures", "analyze.blog_figures_b", "analyze.figures"]


def main(argv: list[str]) -> int:
    rc = 0
    for m in MODULES:
        print(f"== {m}", flush=True)
        r = subprocess.run([sys.executable, "-m", m, *argv])
        rc = rc or r.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
