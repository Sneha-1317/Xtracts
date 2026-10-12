import random
import shutil

import pandas as pd
from PIL import Image, ImageDraw

from xtracts.data.dedup import find_duplicates, summarize
from xtracts.data.inventory import build_inventory


def fake_receipt(path, seed, size=(500, 900)):
    """White page with a seeded random arrangement of text-like bars + a few text lines."""
    rnd = random.Random(seed)
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y = 30
    while y < size[1] - 40:
        x0 = rnd.randint(20, 120)
        d.rectangle([x0, y, x0 + rnd.randint(80, 330), y + rnd.randint(8, 18)], fill=(20, 20, 20))
        if rnd.random() < 0.4:
            d.text((rnd.randint(20, 300), y + 22), f"ITEM {rnd.randint(100, 999)} {rnd.random():.2f}", fill="black")
        y += rnd.randint(28, 50)
    img.save(path)
    return img


def make_dataset(root):
    root.mkdir(parents=True, exist_ok=True)
    a = fake_receipt(root / "a.png", seed=1)
    shutil.copy(root / "a.png", root / "a_copy.png")                       # exact duplicate
    a.resize((300, 540)).save(root / "a_small.jpg", quality=70)            # near duplicate (resized + jpeg)
    fake_receipt(root / "b.png", seed=2)
    fake_receipt(root / "c.png", seed=3)
    (root / "broken.jpg").write_bytes(b"not an image at all")              # corrupt
    Image.new("RGB", (100, 100), "white").save(root / "tiny.png")          # too small
    sub = root / "sub"
    sub.mkdir()
    fake_receipt(sub / "d.png", seed=4)


def run(root, **kw):
    inv = build_inventory(root, min_side=300, workers=1)
    return find_duplicates(inv, **kw)


def test_inventory_statuses(tmp_path):
    make_dataset(tmp_path)
    inv = build_inventory(tmp_path, min_side=300, workers=1)
    st = dict(zip(inv["path"], inv["status"]))
    assert st["broken.jpg"] == "corrupt"
    assert st["tiny.png"] == "too_small"
    assert st["sub/d.png"] == "ok"  # recursion works
    assert inv["status"].value_counts()["ok"] == 6


def test_exact_and_near_duplicates(tmp_path):
    make_dataset(tmp_path)
    inv = run(tmp_path)
    row = inv.set_index("path")
    assert row.loc["a_copy.png", "dup_type"] == "exact"
    assert row.loc["a_small.jpg", "dup_type"] == "near"
    # biggest resolution wins; a.png and a_copy.png tie on size -> alphabetical
    assert row.loc["a.png", "is_keeper"] and not row.loc["a_copy.png", "is_keeper"]
    assert not row.loc["a_small.jpg", "is_keeper"]
    # unrelated receipts are untouched
    for p in ("b.png", "c.png", "sub/d.png"):
        assert row.loc[p, "is_keeper"] and row.loc[p, "dup_type"] == "unique"
    # invalid files never become keepers
    assert not row.loc["broken.jpg", "is_keeper"] and not row.loc["tiny.png", "is_keeper"]


def test_summary_counts(tmp_path):
    make_dataset(tmp_path)
    s = summarize(run(tmp_path))
    assert s["valid_images"] == 6
    assert s["duplicate_clusters"] == 1
    assert s["exact_duplicates_removed"] == 1
    assert s["near_duplicates_removed"] == 1
    assert s["unique_images_kept"] == 4  # a, b, c, d


def test_greedy_assign_does_not_chain():
    import numpy as np
    from xtracts.data.dedup import greedy_assign

    # 0~1 and 1~2 match, but 0 and 2 do not (a chain 0-1-2)
    near = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    # keeper 0 takes 1; 2 is not close to the keeper, so it stays on its own
    assert list(greedy_assign(near, [0, 1, 2])) == [0, 0, 2]
    # if 1 is the best candidate it directly matches both
    assert list(greedy_assign(near, [1, 0, 2])) == [1, 1, 1]
