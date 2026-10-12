"""Step 2: run PaddleOCR on every receipt in manifest_clean.csv.

Output: data/annotations/ocr/<image_id>.json  (lines + words + boxes + confidences)

Usage (from the project root):
    python scripts/02_run_ocr.py --limit 20 --preview 20   # quick trial + boxes drawn on images
    python scripts/02_run_ocr.py                           # all images (resumable: re-run to continue)
    python scripts/02_run_ocr.py --overwrite               # redo everything
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from xtracts.config import load_config, resolve  # noqa: E402
from xtracts.ocr.runner import run_ocr  # noqa: E402


def draw_preview(ocr_dir: Path, images_dir: Path, out_dir: Path, ids: list[str], clean_paths: dict) -> None:
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_id in ids:
        rec = json.loads((ocr_dir / f"{image_id}.json").read_text(encoding="utf-8"))
        img = Image.open(images_dir / clean_paths[image_id]).convert("RGB")
        d = ImageDraw.Draw(img)
        for w in rec["words"]:
            d.rectangle(w["box"], outline=(255, 0, 0) if w["conf"] < 0.8 else (0, 160, 0), width=2)
        img.save(out_dir / f"{image_id}.jpg", quality=85)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="OCR only the first N pending images")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preview", type=int, default=0, help="draw word boxes for N processed images")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    P = {k: resolve(v) for k, v in cfg["paths"].items()}
    oc = cfg["ocr"]

    manifest = pd.read_csv(P["clean_manifest_csv"])
    if "clean_path" not in manifest.columns:
        sys.exit("manifest_clean.csv has no clean_path column. Run: python scripts/01_prepare_dataset.py --export")

    from xtracts.ocr.engine import PaddleOCREngine  # imported here so --help works without paddle
    engine = PaddleOCREngine(oc["lang"], oc["use_gpu"] or a.gpu, oc["det_limit_side_len"])
    print(f"Engine: {engine.name}")

    summary, errors = run_ocr(manifest, P["clean_images_dir"], P["ocr_dir"], engine, engine.name,
                              oc["min_conf"], a.overwrite, a.limit)

    # accumulate the summary across resumed runs
    P["ocr_summary_csv"].parent.mkdir(parents=True, exist_ok=True)
    if P["ocr_summary_csv"].exists() and not a.overwrite:
        summary = pd.concat([pd.read_csv(P["ocr_summary_csv"]), summary]).drop_duplicates("image_id", keep="last")
    summary.to_csv(P["ocr_summary_csv"], index=False)
    pd.DataFrame(errors, columns=["image_id", "error"]).to_csv(P["ocr_errors_csv"], index=False)

    if a.preview and len(summary):
        ids = summary["image_id"].tail(a.preview).tolist()
        draw_preview(P["ocr_dir"], P["clean_images_dir"], P["ocr_preview_dir"], ids,
                     dict(zip(manifest["image_id"], manifest["clean_path"])))
        print(f"Preview images: {P['ocr_preview_dir']}")

    print("\n=== OCR summary ===")
    print(f"images with OCR json : {len(summary)} / {len(manifest)}")
    print(f"errors this run      : {len(errors)}  (see {P['ocr_errors_csv']})")
    if len(summary):
        print(f"empty results        : {int((summary['n_words'] == 0).sum())}")
        print(f"words per image      : mean {summary['n_words'].mean():.1f}, min {summary['n_words'].min()}, max {summary['n_words'].max()}")
        print(f"mean confidence      : {summary['mean_conf'].mean():.3f}")


if __name__ == "__main__":
    main()
