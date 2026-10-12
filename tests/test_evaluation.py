from xtracts.model.evaluation import flatten, markdown_summary, value_accuracy, value_rows
from xtracts.model.metrics import score

ROWS = [
    {"image_id": "a", "words": ["FRESH", "MART", "05/08/2026", "TOTAL", "5.50", "0.33"],
     "labels": ["B-MERCHANT", "I-MERCHANT", "B-DATE", "O", "B-TOTAL", "B-TAX"]},
    {"image_id": "b", "words": ["CAFE", "12/08/2026", "TOTAL", "RM9.00"],
     "labels": ["B-MERCHANT", "B-DATE", "O", "B-TOTAL"]},
]
PRED = [
    ["B-MERCHANT", "I-MERCHANT", "B-DATE", "O", "B-TOTAL", "O"],          # tax missed
    ["B-MERCHANT", "O", "O", "B-TOTAL"],                                  # date missed
]


def test_value_level_scoring_and_flatten():
    v = value_accuracy(value_rows(ROWS, PRED))
    assert v["MERCHANT"]["value_exact"] == 1.0 and v["TOTAL"]["value_exact"] == 1.0
    assert v["TAX"]["value_exact"] == 0.5 and v["DATE"]["value_exact"] == 0.5   # b has no tax -> match
    s = score([r["labels"] for r in ROWS], PRED)
    table = flatten("X", s, v)
    assert table[-1]["field"] == "ALL (micro)" and len(table) == 5
    md = markdown_summary({"split": "test", "n_receipts": 2, "generated": "now"}, table)
    assert "2 receipts" in md and "Limitations" in md
