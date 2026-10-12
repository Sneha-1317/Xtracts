"""Turn reviewed labels + OCR output into model-ready rows (words, 0-1000 boxes, BIO labels)."""
from __future__ import annotations

import json
from pathlib import Path

from ..annotation.labels import entities_to_bio, entity_values, load_all_label_files


def normalize_box(box: list[int], width: int, height: int) -> list[int]:
    """Pixel box -> LayoutLMv3's 0-1000 coordinate space (clipped, x1>=x0, y1>=y0)."""
    x0, y0, x1, y1 = box
    n = [1000 * x0 / width, 1000 * y0 / height, 1000 * x1 / width, 1000 * y1 / height]
    n = [min(1000, max(0, int(round(v)))) for v in n]
    n[2], n[3] = max(n[2], n[0]), max(n[3], n[1])
    return n


def build_rows(labels_dir: Path, ocr_dir: Path, include_unreviewed: bool = False) -> tuple[list[dict], list[str]]:
    """One row per usable receipt. Returns (rows, warnings).

    Row: image_id, image (file name in the clean-images dir), words, boxes (0-1000), labels (BIO),
         merchant (text, used only to group the split)
    """
    rows, warnings = [], []
    for lab in load_all_label_files(labels_dir):
        if lab["exclude"] or not (lab["reviewed"] or include_unreviewed):
            continue
        ocr_path = Path(ocr_dir) / f"{lab['image_id']}.json"
        if not ocr_path.exists():
            warnings.append(f"{lab['image_id']}: OCR file missing, skipped")
            continue
        rec = json.loads(ocr_path.read_text(encoding="utf-8"))
        words = rec["words"]
        if len(words) != len(lab["entities"]):
            warnings.append(f"{lab['image_id']}: OCR was re-run after labeling ({len(words)} words vs "
                            f"{len(lab['entities'])} labels), skipped")
            continue
        texts = [w["text"] for w in words]
        rows.append({
            "image_id": lab["image_id"],
            "image": rec["image"],
            "words": texts,
            "boxes": [normalize_box(w["box"], rec["width"], rec["height"]) for w in words],
            "labels": entities_to_bio(lab["entities"]),
            "merchant": entity_values(texts, lab["entities"]).get("MERCHANT", ""),
        })
    return rows, warnings
