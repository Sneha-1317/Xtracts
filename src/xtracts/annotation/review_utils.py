"""Helpers for the review app (no Streamlit import, so they are easy to test)."""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .labels import ENTITIES, load_all_label_files

COLORS = {
    "MERCHANT": (30, 100, 220),
    "DATE": (230, 130, 0),
    "TOTAL": (0, 150, 60),
    "TAX": (150, 50, 190),
    "O": (170, 170, 170),
}


def load_all_labels(labels_dir: Path) -> list[dict]:
    return load_all_label_files(labels_dir)


def build_queue(labels: list[dict], mode: str = "Flagged first", include_reviewed: bool = False,
                seed: int = 42) -> list[str]:
    """Order in which receipts are shown.

    'Flagged first' puts receipts where the rules were unsure at the top (most flags first),
    'Random' gives an unbiased sample (best for estimating how good the auto labels are),
    'File order' is alphabetical by image id.
    """
    pool = [l for l in labels if include_reviewed or not l["reviewed"]]
    rnd = random.Random(seed)
    if mode == "File order":
        pool = sorted(pool, key=lambda l: l["image_id"])
    elif mode == "Random":
        rnd.shuffle(pool)
    else:
        rnd.shuffle(pool)
        pool = sorted(pool, key=lambda l: -len(l["flags"]))
    return [l["image_id"] for l in pool]


def agreement_stats(labels: list[dict]) -> dict:
    """How often the auto label for each field matched the human-verified one (reviewed receipts only)."""
    rev = [l for l in labels if l["reviewed"] and not l["exclude"]]
    stats = {"n_reviewed": len(rev)}
    for e in ENTITIES:
        if not rev:
            stats[e] = None
            continue
        same = sum(
            {i for i, x in enumerate(l["auto_entities"]) if x == e} == {i for i, x in enumerate(l["entities"]) if x == e}
            for l in rev
        )
        stats[e] = same / len(rev)
    return stats


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def draw_boxes(image: Image.Image, words: list[dict], entities: list[str], max_height: int = 1400) -> Image.Image:
    """Image with every word boxed; entity words are coloured, others light gray. Word index shown above each box."""
    img = image.convert("RGB")
    s = min(1.0, max_height / img.height)
    if s < 1.0:
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    font = _font(max(11, int(13 * (1.0 if s == 1.0 else 1.0))))
    for i, (w, e) in enumerate(zip(words, entities)):
        x0, y0, x1, y1 = [v * s for v in w["box"]]
        col = COLORS.get(e, COLORS["O"])
        d.rectangle([x0, y0, x1, y1], outline=col, width=3 if e != "O" else 1)
        d.text((x0, max(0, y0 - 14)), str(i), fill=col, font=font)
    return img
