"""Turn raw OCR output (text lines with polygons) into a clean, model-ready record.

Conventions
-----------
* Boxes are axis-aligned pixel boxes [x0, y0, x1, y1] in the coordinates of the
  image that was OCR'd. They are normalized to LayoutLMv3's 0-1000 range later,
  at training time, using the stored image width/height.
* PaddleOCR detects text LINES. Words are derived by splitting each line on
  whitespace and dividing the line's box in proportion to character count.
  This is an approximation (proportional fonts are not exact), but it is the
  usual way to get word-level boxes from line-level detections.
"""
from __future__ import annotations

from statistics import median


def poly_to_box(poly, width: int, height: int) -> list[int] | None:
    """Polygon (list of [x, y]) -> clipped axis-aligned box, or None if degenerate."""
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    x0 = max(0, int(round(min(xs))))
    y0 = max(0, int(round(min(ys))))
    x1 = min(width, int(round(max(xs))))
    y1 = min(height, int(round(max(ys))))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def sort_reading_order(lines: list[dict]) -> list[dict]:
    """Top-to-bottom, then left-to-right within a visual row.

    Lines whose vertical centres are close (within half a line height) are put
    on the same row, so 'Total' and '525.00' on one receipt row stay in order.
    """
    items = sorted(lines, key=lambda l: (l["box"][1] + l["box"][3]) / 2)
    rows: list[list[dict]] = []
    for ln in items:
        cy = (ln["box"][1] + ln["box"][3]) / 2
        h = ln["box"][3] - ln["box"][1]
        if rows:
            last = rows[-1]
            row_cy = sum((l["box"][1] + l["box"][3]) / 2 for l in last) / len(last)
            row_h = median(l["box"][3] - l["box"][1] for l in last)
            if abs(cy - row_cy) <= 0.5 * min(h, row_h):
                last.append(ln)
                continue
        rows.append([ln])
    return [ln for row in rows for ln in sorted(row, key=lambda l: l["box"][0])]


def split_line_into_words(text: str, box: list[int], conf: float, line_id: int) -> list[dict]:
    """Split one OCR line into words, sharing the line box proportionally by characters."""
    words = text.split()
    if not words:
        return []
    x0, y0, x1, y1 = box
    total = sum(len(w) for w in words) + (len(words) - 1)  # a space counts as one char
    width = x1 - x0
    out, cursor = [], float(x0)
    for i, w in enumerate(words):
        w_end = x1 if i == len(words) - 1 else cursor + width * len(w) / total
        wx0, wx1 = int(round(cursor)), int(round(w_end))
        wx1 = max(wx1, wx0 + 1)
        out.append({"text": w, "box": [wx0, y0, min(wx1, x1), y1], "conf": conf, "line_id": line_id})
        cursor += width * (len(w) + 1) / total
    return out


def build_record(image_id: str, image_name: str, width: int, height: int,
                 raw_lines: list[dict], engine_name: str, min_conf: float = 0.0) -> dict:
    """raw_lines: [{"text": str, "poly": [[x, y], ...], "conf": float}, ...]"""
    lines = []
    for r in raw_lines:
        text = (r.get("text") or "").strip()
        conf = float(r.get("conf", 0.0))
        if not text or conf < min_conf:
            continue
        box = poly_to_box(r["poly"], width, height)
        if box is None:
            continue
        lines.append({"text": text, "box": box, "conf": round(conf, 4)})

    lines = sort_reading_order(lines)
    words = []
    for i, ln in enumerate(lines):
        ln["line_id"] = i
        for w in split_line_into_words(ln["text"], ln["box"], ln["conf"], i):
            words.append(w)
    return {
        "image_id": image_id,
        "image": image_name,
        "width": width,
        "height": height,
        "ocr_engine": engine_name,
        "lines": lines,
        "words": words,
    }
