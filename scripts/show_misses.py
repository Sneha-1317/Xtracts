"""Show the receipt rows around the value YOU labelled where the auto label missed or was wrong.
Read-only. Use it to see what words/layout the rules failed on (e.g. what the tax row looks like).

Legend:  [[word]] = what you labelled     {{word}} = what the auto label chose

Usage (from the project root):
    python scripts/show_misses.py                          # TAX and TOTAL, 12 cases each
    python scripts/show_misses.py --fields TAX --n 25
    python scripts/show_misses.py --fields MERCHANT        # shows the top rows of the receipt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xtracts.annotation.labels import ENTITIES, load_all_label_files  # noqa: E402
from xtracts.annotation.weak_labels import auto_label, group_rows  # noqa: E402
from xtracts.config import load_config, resolve  # noqa: E402


def render_row(words: list[dict], row: list[int], human: set[int], auto: set[int]) -> str:
    out = []
    for i in row:
        t = words[i]["text"]
        if i in human and i in auto:
            t = "[[{{" + t + "}}]]"
        elif i in human:
            t = "[[" + t + "]]"
        elif i in auto:
            t = "{{" + t + "}}"
        out.append(t)
    return "  ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", nargs="+", default=["TAX", "TOTAL"], choices=ENTITIES)
    ap.add_argument("--n", type=int, default=12, help="cases per field")
    a = ap.parse_args()
    cfg = load_config()
    labels_dir, ocr_dir = resolve(cfg["paths"]["labels_dir"]), resolve(cfg["paths"]["ocr_dir"])
    labs = [l for l in load_all_label_files(labels_dir) if l["reviewed"] and not l["exclude"]]

    for field in a.fields:
        print(f"\n================ {field} ================")
        shown = 0
        for l in labs:
            rec = json.loads((ocr_dir / f"{l['image_id']}.json").read_text(encoding="utf-8"))
            words = rec["words"]
            fresh, fresh_flags = auto_label(rec)               # CURRENT rules, not the stored (stale) auto labels
            l["flags"] = fresh_flags
            auto = {i for i, x in enumerate(fresh) if x == field}
            human = {i for i, x in enumerate(l["entities"]) if x == field}
            if auto == human:
                continue
            rows = group_rows(words)
            hit = [ri for ri, r in enumerate(rows) if human & set(r) or auto & set(r)]
            if field == "MERCHANT" or not hit:
                hit = list(range(min(len(rows), 4)))       # merchant: the top of the receipt
            lo, hi = max(0, min(hit) - 1), min(len(rows), max(hit) + 2)
            kind = "missed" if not auto else ("spurious" if not human else "wrong/partial")
            print(f"\n{l['image_id']}  ({kind})  flags: {','.join(l['flags']) or '-'}")
            for ri in range(lo, hi):
                print(f"   {render_row(words, rows[ri], human, auto)}")
            shown += 1
            if shown >= a.n:
                break
        if shown == 0:
            print("no disagreements")


if __name__ == "__main__":
    main()
