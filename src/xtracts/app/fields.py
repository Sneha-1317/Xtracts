"""Turn per-word BIO tags into clean field values (merchant text, ISO date, numeric amounts).

No torch needed here, so the evaluation and the app tests can use it anywhere.
"""
from __future__ import annotations

import re
from datetime import date

from ..annotation.labels import ENTITIES
from ..annotation.weak_labels import _MON, parse_amount
from ..model.metrics import entity_spans

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_NUM = re.compile(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})(?!\d)")
_YMD = re.compile(r"(?<!\d)((?:19|20)\d{2})[/\-.](\d{1,2})[/\-.](\d{1,2})(?!\d)")
_NAME = re.compile(rf"(?<!\d)(\d{{1,2}})[ \-/.]?({_MON})[a-z]*[ \-/,.]*((?:19|20)?\d{{2}})", re.I)


def _valid(y: int, m: int, d: int) -> str | None:
    if y < 100:
        y += 2000
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def normalize_date(text: str) -> str | None:
    """'05/08/2026' / '2026-08-05' / '5 Aug 26' -> '2026-08-05' (day-first, as in Malaysian receipts)."""
    if not text:
        return None
    if m := _YMD.search(text):
        return _valid(int(m[1]), int(m[2]), int(m[3]))
    if m := _NUM.search(text):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        return _valid(y, b, a) or _valid(y, a, b)          # dd/mm first, then mm/dd
    if m := _NAME.search(text):
        mon = _MONTHS.get(m[2][:3].lower())
        return _valid(int(m[3]), mon, int(m[1])) if mon else None
    return None


def normalize_amount(text: str) -> float | None:
    """'RM 1,234.50' / '525.00*' -> number. Accepts a missing decimal part only as an integer ('12')."""
    if not text:
        return None
    t = text.strip()
    if (v := parse_amount(t, loose=True)) is not None:
        return v
    m = re.fullmatch(r"\D{0,3}(\d{1,6})\D{0,1}", t)
    return float(m[1]) if m else None


def tags_to_fields(words: list[str], tags: list[str], probs: list[float] | None = None) -> dict[str, str]:
    """BIO tags -> {FIELD: text}. If a field was tagged more than once, the span with the highest
    mean probability wins (or the last one for TOTAL/TAX when no probabilities are given)."""
    best: dict[str, tuple[float, int, str]] = {}
    for field, s, e in sorted(entity_spans(tags), key=lambda x: x[1]):
        conf = sum(probs[s:e + 1]) / (e - s + 1) if probs else float(s if field in ("TOTAL", "TAX") else -s)
        if field not in best or conf > best[field][0]:
            best[field] = (conf, s, " ".join(words[s:e + 1]))
    return {f: best[f][2] for f in ENTITIES if f in best}


def clean_fields(raw: dict[str, str]) -> dict:
    """Raw text per field -> {'merchant': str, 'date': 'YYYY-MM-DD'|'', 'total': float|None, 'tax': float|None}."""
    return {
        "merchant": raw.get("MERCHANT", "").strip(),
        "date": normalize_date(raw.get("DATE", "")) or "",
        "total": normalize_amount(raw.get("TOTAL", "")),
        "tax": normalize_amount(raw.get("TAX", "")),
    }
