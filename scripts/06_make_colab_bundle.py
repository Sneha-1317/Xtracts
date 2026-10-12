"""Step 6: pack everything needed to TRAIN into one small zip for Google Colab / Kaggle.

Training on a laptop CPU takes many hours; a free GPU takes minutes. The bundle contains the code,
config, the train/val/test .jsonl files and only the receipt images those files use. LayoutLMv3 looks
at the image at 224x224, so images are shrunk (long side 768 px) to keep the zip small.

Usage (from the project root):
    python scripts/06_make_colab_bundle.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from xtracts.config import PROJECT_ROOT, load_config, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-side", type=int, default=768)
    a = ap.parse_args()

    cfg = load_config(a.config)
    P = {k: resolve(v) for k, v in cfg["paths"].items()}
    jsonls = sorted(P["splits_dir"].glob("*.jsonl"))
    if not jsonls:
        sys.exit(f"No split files in {P['splits_dir']}. Run: python scripts/04_build_dataset.py")

    images = set()
    for j in jsonls:
        for line in j.read_text(encoding="utf-8").splitlines():
            if line.strip():
                images.add(json.loads(line)["image"])

    out = resolve(cfg["paths"]["colab_bundle"])
    rel = lambda p: Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()  # noqa: E731
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ("src", "scripts"):
            for p in sorted((PROJECT_ROOT / folder).rglob("*.py")):
                if "__pycache__" not in p.parts:
                    z.write(p, rel(p))
        for name in ("configs/config.yaml", "requirements.txt", "requirements-train.txt", "pyproject.toml"):
            z.write(PROJECT_ROOT / name, name)
        for j in jsonls:
            z.write(j, rel(j))
        missing = 0
        for name in sorted(images):
            src = P["clean_images_dir"] / name
            if not src.exists():
                missing += 1
                continue
            img = Image.open(src).convert("RGB")
            img.thumbnail((a.max_side, a.max_side), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=88)
            z.writestr(rel(P["clean_images_dir"] / name), buf.getvalue())

    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(images) - missing} images, {len(jsonls)} split files)")
    if missing:
        print(f"WARNING: {missing} images listed in the splits were not found in {P['clean_images_dir']}")


if __name__ == "__main__":
    main()
