"""Phase 1: inventory the raw receipt images, validate them, remove duplicates.

Usage (from the project root):
    python scripts/01_prepare_dataset.py                       # report only, nothing copied
    python scripts/01_prepare_dataset.py --export              # also write normalized unique images
    python scripts/01_prepare_dataset.py --raw-dir D:/receipts --phash-threshold 4 --review
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xtracts.config import load_config, resolve  # noqa: E402
from xtracts.data.dedup import find_duplicates, summarize  # noqa: E402
from xtracts.data.export import (  # noqa: E402
    export_clean_images, make_review_sheets, write_reports)
from xtracts.data.inventory import build_inventory  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw-dir", default=None, help="override paths.raw_dir")
    ap.add_argument("--phash-threshold", type=int, default=None)
    ap.add_argument("--dhash-threshold", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--export", action="store_true", help="write normalized copies of unique images")
    ap.add_argument("--review", action="store_true", help="write contact sheets for near-duplicate clusters")
    a = ap.parse_args()

    cfg = load_config(a.config)
    P = {k: resolve(v) for k, v in cfg["paths"].items()}
    raw_dir = resolve(a.raw_dir) if a.raw_dir else P["raw_dir"]
    dd = cfg["dedup"]
    ph_t = a.phash_threshold if a.phash_threshold is not None else dd["phash_threshold"]
    dh_t = a.dhash_threshold if a.dhash_threshold is not None else dd["dhash_threshold"]
    workers = a.workers or cfg["workers"]

    inv = build_inventory(raw_dir, cfg["validation"]["min_side"], cfg["validation"]["max_pixels"], workers)
    inv = find_duplicates(inv, ph_t, dh_t, dd["aspect_tolerance"])
    summary = summarize(inv)
    summary["thresholds"] = {"phash": ph_t, "dhash": dh_t, "aspect_tolerance": dd["aspect_tolerance"]}

    manifest = write_reports(inv, summary, inventory_csv=P["inventory_csv"],
                             manifest_csv=P["clean_manifest_csv"],
                             duplicates_csv=P["duplicates_report_csv"], summary_json=P["summary_json"])

    if a.export:
        ex = cfg["export"]
        manifest = export_clean_images(manifest, raw_dir, P["clean_images_dir"],
                                       ex["max_side"], ex["jpeg_quality"])
        manifest.to_csv(P["clean_manifest_csv"], index=False)
    if a.review:
        n = make_review_sheets(inv, raw_dir, P["review_dir"])
        print(f"Wrote {n} review sheets to {P['review_dir']}")

    print("\n=== Dataset summary ===")
    for k, v in summary.items():
        print(f"{k:>26}: {v}")
    print(f"\nInventory : {P['inventory_csv']}\nManifest  : {P['clean_manifest_csv']}"
          f"\nDuplicates: {P['duplicates_report_csv']}")


if __name__ == "__main__":
    main()
