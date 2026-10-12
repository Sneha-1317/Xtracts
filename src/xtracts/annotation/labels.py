"""Label schema and label-file I/O.

Annotators work with ONE entity per word (O / MERCHANT / DATE / TOTAL / TAX).
BIO tags (B-TOTAL, I-MERCHANT, ...) are derived from them when the training set is built:
a run of consecutive words with the same entity becomes B- followed by I-.

Label file  data/annotations/labels/<image_id>.json
    image_id, n_words
    auto_entities   what the rule-based labeler proposed (never edited)
    entities        current labels, one per OCR word (edited in the review app)
    flags           reasons the receipt deserves a careful look
    reviewed        True once a human saved it
    exclude         True if the receipt is unusable (unreadable, not a receipt, ...)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ENTITIES = ["MERCHANT", "DATE", "TOTAL", "TAX"]
ENTITY_CHOICES = ["O"] + ENTITIES
LABEL_LIST = ["O"] + [f"{p}-{e}" for e in ENTITIES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def entities_to_bio(entities: list[str]) -> list[str]:
    """['O','TOTAL','TOTAL','O'] -> ['O','B-TOTAL','I-TOTAL','O']"""
    bio = []
    for i, e in enumerate(entities):
        if e == "O":
            bio.append("O")
        elif i > 0 and entities[i - 1] == e:
            bio.append(f"I-{e}")
        else:
            bio.append(f"B-{e}")
    return bio


def entity_values(words: list[str], entities: list[str]) -> dict[str, str]:
    """Readable field values, e.g. {'TOTAL': '525.00', 'MERCHANT': 'FRESH MART'}."""
    out: dict[str, str] = {}
    for w, e in zip(words, entities):
        if e != "O":
            out[e] = (out.get(e, "") + " " + w).strip()
    return out


def label_path(labels_dir: Path, image_id: str) -> Path:
    return Path(labels_dir) / f"{image_id}.json"


def load_label(labels_dir: Path, image_id: str) -> dict | None:
    p = label_path(labels_dir, image_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_label(labels_dir: Path, label: dict) -> None:
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    p = label_path(labels_dir, label["image_id"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)  # atomic: a crash never leaves a half-written label file


def new_label(image_id: str, entities: list[str], flags: list[str]) -> dict:
    return {
        "image_id": image_id,
        "n_words": len(entities),
        "auto_entities": list(entities),
        "entities": list(entities),
        "flags": flags,
        "reviewed": False,
        "exclude": False,
    }


def load_all_label_files(labels_dir: Path) -> list[dict]:
    """Every label file in the folder, sorted by image id."""
    out = []
    for p in sorted(Path(labels_dir).glob("*.json")):
        lab = load_label(labels_dir, p.stem)
        if lab:
            out.append(lab)
    return out
