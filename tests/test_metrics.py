from xtracts.model.metrics import entity_spans, format_report, score


def test_entity_spans_handles_runs_and_stray_I():
    tags = ["B-MERCHANT", "I-MERCHANT", "O", "B-TOTAL", "I-DATE", "I-DATE"]
    assert entity_spans(tags) == {("MERCHANT", 0, 1), ("TOTAL", 3, 3), ("DATE", 4, 5)}


def test_score_counts_only_exact_span_matches():
    gold = [["B-MERCHANT", "I-MERCHANT", "O", "B-TOTAL"], ["O", "B-DATE", "O", "O"]]
    pred = [["B-MERCHANT", "O", "O", "B-TOTAL"],           # merchant span too short -> wrong; total right
            ["O", "B-DATE", "O", "B-TAX"]]                 # date right; extra TAX -> false positive
    m = score(gold, pred)
    assert m["fields"]["TOTAL"]["f1"] == 1.0
    assert m["fields"]["DATE"]["f1"] == 1.0
    assert m["fields"]["MERCHANT"]["f1"] == 0.0 and m["fields"]["MERCHANT"]["support"] == 1
    assert m["fields"]["TAX"]["precision"] == 0.0
    assert m["fields"]["TOTAL"]["exact_match"] == 1.0
    assert m["fields"]["MERCHANT"]["exact_match"] == 0.5   # receipt 2 has no merchant on both sides
    # micro: tp=2 (total,date), fp=2 (short merchant, tax), fn=1 (merchant)
    assert round(m["micro"]["precision"], 3) == 0.5 and round(m["micro"]["recall"], 3) == round(2 / 3, 3)
    assert "TOTAL" in format_report(m)
