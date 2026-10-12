"""Write outputs: clean manifest, duplicates report, normalized image copies, review sheets."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

from .inventory import load_normalized


def write_reports(inv: pd.DataFrame, summary: dict, *, inventory_csv: Path, manifest_csv: Path,
                  duplicates_csv: Path, summary_json: Path) -> pd.DataFrame:
    """Save inventory, duplicates report, clean manifest and summary. Returns the manifest."""
    for p in (inventory_csv, manifest_csv, duplicates_csv, summary_json):
        p.parent.mkdir(parents=True, exist_ok=True)

    inv = inv.copy()
    for c in ("width", "height"):
        inv[c] = inv[c].astype("Int64")  # nullable int: no "500.0" in the CSVs
    inv.to_csv(inventory_csv, index=False)

    dups = inv[inv["cluster_id"] >= 0].sort_values(["cluster_id", "is_keeper"], ascending=[True, False])
    dups[["cluster_id", "cluster_size", "dup_type", "is_keeper", "path", "keeper_path", "phash_dist",
          "width", "height", "file_size"]].to_csv(duplicates_csv, index=False)

    manifest = inv[inv["is_keeper"]][["image_id", "path", "width", "height", "file_size"]]
    manifest = manifest.sort_values("path").reset_index(drop=True)
    manifest.to_csv(manifest_csv, index=False)

    summary_json.write_text(json.dumps(summary, indent=2))
    return manifest


def export_clean_images(manifest: pd.DataFrame, raw_dir: Path, out_dir: Path,
                        max_side: int | None = None, jpeg_quality: int = 95) -> pd.DataFrame:
    """Save normalized (EXIF-rotated, RGB) copies as <image_id>.jpg.
    Adds a `clean_path` column (relative to out_dir)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest.copy()
    clean_paths = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="Exporting clean images"):
        img = load_normalized(Path(raw_dir) / r["path"])
        if max_side and max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        name = f"{r['image_id']}.jpg"
        img.save(out_dir / name, "JPEG", quality=jpeg_quality)
        clean_paths.append(name)
    manifest["clean_path"] = clean_paths
    return manifest


def make_review_sheets(inv: pd.DataFrame, raw_dir: Path, out_dir: Path, max_clusters: int = 50,
                       thumb_h: int = 360) -> int:
    """Contact sheets (one JPEG per near-duplicate cluster) so you can eyeball the threshold.
    The keeper is outlined green; removed copies red. Returns number of sheets written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    near_clusters = inv[inv["dup_type"] == "near"]["cluster_id"].unique()[:max_clusters]
    for cid in near_clusters:
        members = inv[inv["cluster_id"] == cid].sort_values(
            ["is_keeper", "phash_dist"], ascending=[False, True]).head(12)  # keeper + 11 closest
        thumbs = []
        for _, r in members.iterrows():
            im = load_normalized(Path(raw_dir) / r["path"])
            im.thumbnail((thumb_h * 2, thumb_h), Image.LANCZOS)
            tile = Image.new("RGB", (im.width + 12, im.height + 36), "white")
            tile.paste(im, (6, 30))
            d = ImageDraw.Draw(tile)
            color = "green" if r["is_keeper"] else "red"
            d.rectangle([2, 26, tile.width - 3, tile.height - 3], outline=color, width=3)
            d.text((6, 6), f"{'KEEP' if r['is_keeper'] else 'DROP'} {Path(r['path']).name[:28]}", fill=color)
            thumbs.append(tile)
        sheet = Image.new("RGB", (sum(t.width for t in thumbs), max(t.height for t in thumbs)), "white")
        x = 0
        for t in thumbs:
            sheet.paste(t, (x, 0))
            x += t.width
        sheet.save(out_dir / f"cluster_{int(cid):04d}.jpg", quality=85)
    return len(near_clusters)
