"""Dataset loading: scan a folder, validate every image, and compute hashes.

Produces one row per file with:
    image_id    short stable id (first 12 chars of the SHA-256 of the file bytes)
    path        path relative to the scanned root (posix style)
    status      ok | corrupt | too_small | too_large
    sha256      exact-duplicate key
    phash/dhash perceptual hashes as 16-char hex (64-bit), near-duplicate keys
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

COLUMNS = [
    "image_id", "path", "status", "error", "sha256", "phash", "dhash",
    "width", "height", "format", "mode", "file_size",
]


def find_images(root: Path) -> list[Path]:
    """Recursively list image files under root (sorted, so runs are reproducible)."""
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_normalized(path: Path) -> Image.Image:
    """Open an image the way the rest of the pipeline should see it:
    EXIF rotation applied, converted to RGB (flattening transparency onto white)."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def inspect_image(args: tuple[str, str, int, int]) -> dict:
    """Validate one image and compute its hashes. Runs in a worker process."""
    root, rel, min_side, max_pixels = args
    path = Path(root) / rel
    row = {c: None for c in COLUMNS}
    row.update(path=rel, file_size=path.stat().st_size)
    try:
        row["sha256"] = _sha256(path)
        row["image_id"] = row["sha256"][:12]

        with Image.open(path) as probe:
            probe.verify()  # detects truncated / corrupt files without decoding everything
        with Image.open(path) as im:
            row["format"] = im.format
            w0, h0 = im.size
            if w0 * h0 > max_pixels:
                row.update(status="too_large", error=f"{w0}x{h0} exceeds max_pixels")
                return row

        img = load_normalized(path)  # full decode: catches errors verify() misses
        w, h = img.size
        row.update(width=w, height=h, mode=img.mode)
        if min(w, h) < min_side:
            row.update(status="too_small", error=f"min side {min(w, h)} < {min_side}")
            return row

        gray = img.convert("L")
        row["phash"] = str(imagehash.phash(gray, hash_size=8))
        row["dhash"] = str(imagehash.dhash(gray, hash_size=8))
        row["status"] = "ok"
    except Exception as e:  # corrupt, truncated, unsupported...
        row.update(status="corrupt", error=f"{type(e).__name__}: {e}"[:200])
    return row


def build_inventory(root: Path, min_side: int = 300, max_pixels: int = 50_000_000,
                    workers: int = 4) -> pd.DataFrame:
    """Scan `root` and return the inventory DataFrame (one row per image file)."""
    root = Path(root)
    files = find_images(root)
    if not files:
        raise FileNotFoundError(f"No images found under {root}")
    jobs = [(str(root), p.relative_to(root).as_posix(), min_side, max_pixels) for p in files]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(tqdm(ex.map(inspect_image, jobs, chunksize=16), total=len(jobs),
                             desc="Inspecting images"))
    else:
        rows = [inspect_image(j) for j in tqdm(jobs, desc="Inspecting images")]
    return pd.DataFrame(rows, columns=COLUMNS)
