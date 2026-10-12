"""Run OCR over the clean manifest. Resumable: images that already have a JSON are skipped."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .engine import OcrEngine
from .postprocess import build_record


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # an interrupted run never leaves a half-written file


def run_ocr(manifest: pd.DataFrame, images_dir: Path, out_dir: Path, engine: OcrEngine,
            engine_name: str, min_conf: float = 0.0, overwrite: bool = False,
            limit: int | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """OCR every image in the manifest -> out_dir/<image_id>.json.

    Returns (summary of images processed in THIS run, list of errors).
    """
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    done = 0
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="OCR"):
        out = out_dir / f"{r['image_id']}.json"
        if out.exists() and not overwrite:
            continue
        if limit is not None and done >= limit:
            break
        img_path = images_dir / r["clean_path"]
        try:
            with Image.open(img_path) as im:
                w, h = im.size
            raw = engine(str(img_path))
            rec = build_record(r["image_id"], r["clean_path"], w, h, raw, engine_name, min_conf)
            _write_json_atomic(out, rec)
            confs = [x["conf"] for x in rec["lines"]]
            rows.append({
                "image_id": r["image_id"], "n_lines": len(rec["lines"]), "n_words": len(rec["words"]),
                "mean_conf": round(sum(confs) / len(confs), 4) if confs else 0.0,
            })
        except Exception as e:  # keep going; failures are listed in the errors file
            errors.append({"image_id": r["image_id"], "error": f"{type(e).__name__}: {e}"[:300]})
        done += 1
    return pd.DataFrame(rows, columns=["image_id", "n_lines", "n_words", "mean_conf"]), errors
