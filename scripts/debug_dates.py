"""Diagnose why the date rule misses dates: shows what date-like text the OCR produced
on receipts flagged `no_date`. Read-only; changes nothing.

Usage (from the project root):
    python scripts/debug_dates.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xtracts.config import load_config, resolve  # noqa: E402

YEAR = re.compile(r"(19|20)\d\d")
SEP_NUMS = re.compile(r"\d{1,4}\s*[/\-.]\s*\d{1,2}")
MONTH = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.I)


def shape(text: str) -> str:
    """'Date:05/08/2026' -> 'a:99/99/9999' (digits -> 9, letter runs -> a)."""
    return re.sub(r"[A-Za-z]+", "a", re.sub(r"\d", "9", text))


def main() -> None:
    cfg = load_config()
    ocr_dir, labels_dir = resolve(cfg["paths"]["ocr_dir"]), resolve(cfg["paths"]["labels_dir"])
    missing = []
    for p in sorted(labels_dir.glob("*.json")):
        lab = json.loads(p.read_text(encoding="utf-8"))
        if "no_date" in lab["flags"]:
            missing.append(lab["image_id"])
    print(f"Receipts flagged no_date: {len(missing)}")

    shapes, example, with_year, with_any = Counter(), {}, 0, 0
    for image_id in missing:
        words = json.loads((ocr_dir / f"{image_id}.json").read_text(encoding="utf-8"))["words"]
        texts = [w["text"] for w in words]
        cands = [t for t in texts if YEAR.search(t) or SEP_NUMS.search(t) or MONTH.search(t)]
        with_any += bool(cands)
        with_year += any(YEAR.search(t) for t in texts)
        for t in cands:
            s = shape(t)
            shapes[s] += 1
            example.setdefault(s, (image_id, t))
    print(f"  ...of which contain a 19xx/20xx year somewhere: {with_year}")
    print(f"  ...of which contain any date-like word:          {with_any}\n")
    print("Most common shapes of date-like words on those receipts (9 = digit, a = letters):")
    print(f"{'count':>6}  {'shape':<28} example")
    for s, n in shapes.most_common(25):
        iid, t = example[s]
        print(f"{n:>6}  {s:<28} {t!r}  ({iid})")

    print("\nFirst 8 flagged receipts: words that look date-like (or 'date'/'time' labels):")
    for image_id in missing[:8]:
        words = json.loads((ocr_dir / f"{image_id}.json").read_text(encoding="utf-8"))["words"]
        texts = [w["text"] for w in words]
        hits = [t for t in texts if YEAR.search(t) or SEP_NUMS.search(t) or MONTH.search(t)
                or re.search(r"date|time|dt", t, re.I)]
        print(f"  {image_id}: {hits[:10]}")


if __name__ == "__main__":
    main()
