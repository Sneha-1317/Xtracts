"""Compare the auto labels with your corrections to see HOW the rules go wrong. Read-only.

For every reviewed receipt and field, the auto label is classified as:
    match      identical words
    missed     auto found nothing, you labelled something
    spurious   auto labelled something, you labelled nothing
    partial    overlapping but different words (e.g. merchant missing a second line)
    wrong      different words, no overlap (e.g. subtotal picked instead of total)

Usage (from the project root):
    python scripts/analyze_review.py                 # summary + 10 examples per field
    python scripts/analyze_review.py --examples 25
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xtracts.annotation.labels import ENTITIES, load_all_label_files  # noqa: E402
from xtracts.annotation.weak_labels import auto_label  # noqa: E402
from xtracts.config import load_config, resolve  # noqa: E402

CATS = ["match", "missed", "spurious", "partial", "wrong"]


def classify(auto: set[int], human: set[int]) -> str:
    if auto == human:
        return "match"
    if not auto:
        return "missed"
    if not human:
        return "spurious"
    return "partial" if auto & human else "wrong"


def table(title: str, counts: dict[str, Counter], n: int) -> None:
    print(f"\n{title} (n={n} receipts)")
    print(f"{'field':<10}" + "".join(f"{c:>10}" for c in CATS) + f"{'accuracy':>10}")
    for e in ENTITIES:
        c = counts[e]
        print(f"{e:<10}" + "".join(f"{c[k]:>10}" for k in CATS) + f"{100 * c['match'] / max(1, n):>9.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config()
    labels_dir, ocr_dir = resolve(cfg["paths"]["labels_dir"]), resolve(cfg["paths"]["ocr_dir"])

    labs = [l for l in load_all_label_files(labels_dir) if l["reviewed"] and not l["exclude"]]
    if not labs:
        sys.exit("No reviewed receipts yet.")
    all_c = {e: Counter() for e in ENTITIES}
    calm_c = {e: Counter() for e in ENTITIES}       # receipts the rules had NO doubts about
    examples = defaultdict(list)
    n_calm = 0
    for l in labs:
        rec = json.loads((ocr_dir / f"{l['image_id']}.json").read_text(encoding="utf-8"))
        words = [w["text"] for w in rec["words"]]
        fresh, l["flags"] = auto_label(rec)               # CURRENT rules (stored auto labels of reviewed receipts are old)
        calm = not (set(l["flags"]) - {"no_tax"})
        n_calm += calm
        for e in ENTITIES:
            auto = {i for i, x in enumerate(fresh) if x == e}
            human = {i for i, x in enumerate(l["entities"]) if x == e}
            cat = classify(auto, human)
            all_c[e][cat] += 1
            if calm:
                calm_c[e][cat] += 1
            if cat != "match":
                txt = lambda s: " ".join(words[i] for i in sorted(s)) or "-"  # noqa: E731
                examples[e].append((l["image_id"], cat, txt(auto), txt(human), ",".join(l["flags"])))

    table("ALL reviewed receipts", all_c, len(labs))
    if 0 < n_calm < len(labs):
        table("Receipts with NO flags (apart from no_tax)", calm_c, n_calm)

    for e in ENTITIES:
        print(f"\n--- {e}: first {a.examples} disagreements (auto vs you) ---")
        for image_id, cat, auto, human, flags in examples[e][: a.examples]:
            print(f"{image_id}  [{cat}]  auto={auto!r:<34} you={human!r}")


if __name__ == "__main__":
    main()
