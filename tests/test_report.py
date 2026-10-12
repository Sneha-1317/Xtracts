import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("docx")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_report", ROOT / "scripts" / "08_make_report.py")
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)

TEMPLATE = ROOT / "docs" / "Xtracts_Mini_Project_Report_TEMPLATE.docx"


def _metrics(f1):
    fields = {f: {"precision": f1, "recall": f1, "f1": f1, "exact_match": f1, "support": 10} for f in mr.FIELDS}
    return {"entity_level": {"fields": fields, "micro": {"precision": f1, "recall": f1, "f1": f1}},
            "value_level": {f: {"value_exact": f1, "n": 10, "n_with_gold": 10} for f in mr.FIELDS}}


@pytest.mark.skipif(not TEMPLATE.exists(), reason="template not present")
def test_report_builds_with_and_without_results(tmp_path):
    from xtracts.config import load_config

    cfg = load_config()
    empty = {"splits": {}, "metrics": None, "summary": None, "reviewed": None, "per": None, "ocr_conf": None}
    left = mr.build(TEMPLATE, tmp_path / "a.docx", empty, cfg)
    assert any("07_evaluate" in x for x in left)                       # missing results stay visible, none invented

    g = dict(empty, metrics={"meta": {"split": "test", "n_receipts": 60, "generated": "now"},
                              "systems": {"Rules": _metrics(0.70), "LayoutLMv3": _metrics(0.85)}},
             splits={"train": (200, {}), "val": (40, {}), "test": (60, {})}, reviewed=300)
    left = mr.build(TEMPLATE, tmp_path / "b.docx", g, cfg)
    import docx
    text = "\n".join(p.text for p in docx.Document(str(tmp_path / "b.docx")).paragraphs)
    assert "85.0%" in text and "60 receipts" in text
    assert not any("07_evaluate" in x for x in left)
    assert TEMPLATE.exists()                                            # the original is untouched
