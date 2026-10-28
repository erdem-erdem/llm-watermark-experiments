"""Bilingual (TR/EN) figure text. / İki dilli (TR/EN) şekil metinleri.

Every user-visible string in a figure is written as an inline pair::

    ax.set_xlabel(L("p-değeri", "p-value"))

`L` resolves at CALL time against the active language, so never evaluate it at import time
(module-level constants); wrap such constants in a function instead.

Figures are written to `figures/<lang>/…` (see `fig_dir`). Data inputs always come from `results/`.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("tr", "en")

_lang = "tr"


def set_lang(lang: str) -> None:
    global _lang
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")
    _lang = lang


def get_lang() -> str:
    return _lang


def L(tr: str, en: str) -> str:
    """Return the Turkish or English text for the active language (both are mandatory)."""
    return en if _lang == "en" else tr


def fig_dir(sub: str = "") -> Path:
    """Output folder for the active language: figures/<lang>[/<sub>]."""
    d = ROOT / "figures" / _lang / sub if sub else ROOT / "figures" / _lang
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_lang_arg(argv: list[str]) -> tuple[list[str], list[str]]:
    """Pull `--lang tr|en|both` out of argv. Returns (languages, remaining_args). Default: both."""
    rest, lang, i = [], "both", 0
    while i < len(argv):
        if argv[i] == "--lang":
            lang = argv[i + 1]
            i += 2
            continue
        if argv[i].startswith("--lang="):
            lang = argv[i].split("=", 1)[1]
        else:
            rest.append(argv[i])
        i += 1
    if lang == "both":
        return list(LANGS), rest
    if lang not in LANGS:
        raise SystemExit(f"--lang must be tr, en or both (got {lang!r})")
    return [lang], rest
