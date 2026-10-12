import json

import pandas as pd
from PIL import Image

from xtracts.ocr.postprocess import build_record, poly_to_box, sort_reading_order, split_line_into_words
from xtracts.ocr.runner import run_ocr


def rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_poly_to_box_clips_and_rejects_degenerate():
    assert poly_to_box(rect(-5, 10, 120, 30), 100, 50) == [0, 10, 100, 30]
    assert poly_to_box(rect(10, 10, 10, 30), 100, 50) is None


def test_split_line_into_words_covers_the_line_box():
    ws = split_line_into_words("Total: 525.00", [100, 20, 230, 40], 0.9, 3)
    assert [w["text"] for w in ws] == ["Total:", "525.00"]
    assert ws[0]["box"][0] == 100 and ws[-1]["box"][2] == 230
    assert ws[0]["box"][2] <= ws[1]["box"][0]  # no overlap, in order
    assert all(w["line_id"] == 3 and w["box"][1] == 20 and w["box"][3] == 40 for w in ws)


def test_reading_order_groups_rows():
    lines = [
        {"text": "525.00", "box": [300, 102, 380, 120]},   # same row as "Total", right side
        {"text": "FRESH MART", "box": [100, 10, 260, 30]},
        {"text": "Total", "box": [100, 100, 180, 118]},
    ]
    assert [l["text"] for l in sort_reading_order(lines)] == ["FRESH MART", "Total", "525.00"]


def test_build_record_filters_and_numbers_lines():
    raw = [
        {"text": "Total 525.00", "poly": rect(100, 100, 300, 120), "conf": 0.97},
        {"text": "  ", "poly": rect(0, 0, 10, 10), "conf": 0.9},          # empty text dropped
        {"text": "noise", "poly": rect(0, 200, 50, 220), "conf": 0.1},    # below min_conf
        {"text": "FRESH MART", "poly": rect(100, 10, 260, 30), "conf": 0.99},
    ]
    rec = build_record("abc", "abc.jpg", 500, 900, raw, "fake", min_conf=0.5)
    assert [l["text"] for l in rec["lines"]] == ["FRESH MART", "Total 525.00"]
    assert [w["text"] for w in rec["words"]] == ["FRESH", "MART", "Total", "525.00"]
    assert rec["width"] == 500 and rec["lines"][1]["line_id"] == 1


def test_run_ocr_is_resumable_and_reports_errors(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for n in ("a", "b", "c"):
        Image.new("RGB", (500, 900), "white").save(imgs / f"{n}.jpg")
    manifest = pd.DataFrame({"image_id": ["a", "b", "c"], "clean_path": ["a.jpg", "b.jpg", "missing.jpg"]})
    calls = []

    def fake_engine(path):
        calls.append(path)
        return [{"text": "Total 5.00", "poly": rect(10, 10, 200, 40), "conf": 0.9}]

    out = tmp_path / "ocr"
    summary, errors = run_ocr(manifest, imgs, out, fake_engine, "fake")
    assert sorted(summary["image_id"]) == ["a", "b"]
    assert [e["image_id"] for e in errors] == ["c"]          # missing file reported, run continues
    assert json.loads((out / "a.json").read_text())["words"][0]["text"] == "Total"

    calls.clear()
    summary2, _ = run_ocr(manifest, imgs, out, fake_engine, "fake")
    assert calls == [] and len(summary2) == 0                # already done -> skipped
