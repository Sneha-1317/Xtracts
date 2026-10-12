import json

import pytest
import yaml
from PIL import Image

from xtracts.annotation.labels import load_label, new_label, save_label
from xtracts.annotation.review_utils import agreement_stats, build_queue, draw_boxes, load_all_labels
from xtracts.annotation.weak_labels import auto_label

from test_weak_labels import make_record

ROWS = [
    (20, [(150, "FRESH"), (230, "MART")]),
    (130, [(20, "Date:"), (90, "05/08/2026")]),
    (560, [(20, "TOTAL"), (400, "7.95")]),
]


def build_project(tmp_path, n=3):
    paths = {k: str(tmp_path / v) for k, v in {
        "ocr_dir": "ocr", "clean_images_dir": "imgs", "labels_dir": "labels"}.items()}
    (tmp_path / "ocr").mkdir(); (tmp_path / "imgs").mkdir()
    for i in range(n):
        iid = f"img{i}"
        Image.new("RGB", (600, 1200), "white").save(tmp_path / "imgs" / f"{iid}.jpg")
        rec = make_record(ROWS)
        rec.update(image_id=iid, image=f"{iid}.jpg")
        (tmp_path / "ocr" / f"{iid}.json").write_text(json.dumps(rec))
        ents, flags = auto_label(rec)
        save_label(paths["labels_dir"], new_label(iid, ents, flags if i else flags + ["no_tax"]))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"paths": paths}))
    return cfg, paths


def test_queue_and_stats(tmp_path):
    _, paths = build_project(tmp_path)
    labels = load_all_labels(paths["labels_dir"])
    assert build_queue(labels, "Flagged first")[0] == "img0"             # the one with a flag
    assert sorted(build_queue(labels, "File order")) == ["img0", "img1", "img2"]

    lab = load_label(paths["labels_dir"], "img1")
    lab["entities"][-1] = "O"                                            # human removes the TOTAL label
    lab["reviewed"] = True
    save_label(paths["labels_dir"], lab)
    st = agreement_stats(load_all_labels(paths["labels_dir"]))
    assert st["n_reviewed"] == 1 and st["TOTAL"] == 0.0 and st["DATE"] == 1.0
    assert build_queue(load_all_labels(paths["labels_dir"]), "File order") == ["img0", "img2"]


def test_draw_boxes_keeps_aspect(tmp_path):
    rec = make_record(ROWS)
    ents, _ = auto_label(rec)
    out = draw_boxes(Image.new("RGB", (600, 3000), "white"), rec["words"], ents, max_height=1400)
    assert out.height == 1400 and out.width == 280


def test_app_runs_and_save_button_marks_reviewed(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    cfg, paths = build_project(tmp_path)
    monkeypatch.setenv("XTRACTS_CONFIG", str(cfg))
    from pathlib import Path
    app = Path(__file__).resolve().parents[1] / "src" / "xtracts" / "annotation" / "review_app.py"
    at = AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception, at.exception
    assert any("Receipt 1 / 3" in s.value for s in at.subheader)

    save = next(b for b in at.button if "Save as reviewed" in b.label)
    save.click().run()
    assert not at.exception, at.exception
    assert load_label(paths["labels_dir"], "img0")["reviewed"] is True      # first in queue (flagged)
    assert any("Receipt 2 / 3" in s.value for s in at.subheader)
