"""Step 3: propose labels (merchant, date, total, tax) for every OCR'd receipt.

Output: data/annotations/labels/<image_id>.json  (reviewed=false until you check it)
These are rule-based SUGGESTIONS. Review and correct them with:
    streamlit run src/xtracts/annotation/review_app.py

Usage (from the project root):
    python scripts/03_auto_label.py                # only creates labels that do not exist yet
    python scripts/03_auto_label.py --overwrite    # regenerate auto labels of UNREVIEWED receipts
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xtracts.annotation.labels import ENTITIES, load_label, new_label, save_label  # noqa: E402
from xtracts.annotation.weak_labels import auto_label  # noqa: E402
from xtracts.config import load_config, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate labels for receipts not yet reviewed (reviewed ones are never touched)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    ocr_dir, labels_dir = resolve(cfg["paths"]["ocr_dir"]), resolve(cfg["paths"]["labels_dir"])
    files = sorted(ocr_dir.glob("*.json"))
    if not files:
        sys.exit(f"No OCR files in {ocr_dir}. Run scripts/02_run_ocr.py first.")

    found, flags_count, written, kept = Counter(), Counter(), 0, 0
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        existing = load_label(labels_dir, rec["image_id"])
        if existing and (existing["reviewed"] or not a.overwrite):
            kept += 1
            continue
        entities, flags = auto_label(rec)
        save_label(labels_dir, new_label(rec["image_id"], entities, flags))
        written += 1
        for e in ENTITIES:
            found[e] += e in entities
        flags_count.update(flags)

    print(f"\nOCR files: {len(files)} | new/regenerated labels: {written} | kept as they were: {kept}")
    if written:
        print("\nFields the rules found (suggestions, not verified):")
        for e in ENTITIES:
            print(f"  {e:<9}: {found[e]:>5} / {written}  ({100 * found[e] / written:.0f}%)")
        print("\nFlags (receipts to look at first):")
        for k, v in flags_count.most_common():
            print(f"  {k:<18}: {v}")
    print(f"\nLabels folder: {labels_dir}\nNext: streamlit run src/xtracts/annotation/review_app.py")


if __name__ == "__main__":
    main()
