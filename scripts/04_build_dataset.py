"""Step 4: build train/val/test files from the REVIEWED labels.

Output: data/processed/splits/{train,val,test}.jsonl  (words, 0-1000 boxes, BIO labels)

Usage (from the project root):
    python scripts/04_build_dataset.py
    python scripts/04_build_dataset.py --include-unreviewed   # smoke test only, uses unverified auto labels!
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xtracts.annotation.labels import ENTITIES  # noqa: E402
from xtracts.config import load_config, resolve  # noqa: E402
from xtracts.model.data_build import build_rows  # noqa: E402
from xtracts.model.splits import grouped_split, merchant_key  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--include-unreviewed", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    P = {k: resolve(v) for k, v in cfg["paths"].items()}
    rows, warnings = build_rows(P["labels_dir"], P["ocr_dir"], a.include_unreviewed)
    for w in warnings:
        print("WARNING:", w)
    if len(rows) < 10:
        sys.exit(f"Only {len(rows)} usable receipts. Review more in the app first "
                 f"(streamlit run src/xtracts/annotation/review_app.py).")

    groups = [merchant_key(r["merchant"], r["image_id"]) for r in rows]
    sp = cfg["split"]
    tr, va, te = grouped_split(groups, sp["val_size"], sp["test_size"], cfg["seed"])

    P["splits_dir"].mkdir(parents=True, exist_ok=True)
    print(f"\n{'split':<6}{'receipts':>9}" + "".join(f"{e:>10}" for e in ENTITIES) + "   (receipts containing the field)")
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        with open(P["splits_dir"] / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i in idx:
                f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        has = Counter(e for i in idx for e in ENTITIES if f"B-{e}" in rows[i]["labels"])
        print(f"{name:<6}{len(idx):>9}" + "".join(f"{has[e]:>10}" for e in ENTITIES))
    print(f"\nWritten to {P['splits_dir']}\nNext: python scripts/06_make_colab_bundle.py")


if __name__ == "__main__":
    main()
