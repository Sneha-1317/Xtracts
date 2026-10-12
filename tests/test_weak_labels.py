from xtracts.annotation.labels import ENTITIES, entities_to_bio, entity_values
from xtracts.annotation.weak_labels import auto_label, group_rows, parse_amount


def make_record(rows, width=600, height=1200):
    """rows: list of (y, [(x, text), ...]) -> OCR-style record with word boxes."""
    words = []
    for y, items in rows:
        for x, text in items:
            words.append({"text": text, "box": [x, y, x + 12 * len(text), y + 24], "conf": 0.95, "line_id": 0})
    return {"width": width, "height": height, "words": words}


def labeled(record):
    ents, flags = auto_label(record)
    return entity_values([w["text"] for w in record["words"]], ents), flags


def test_parse_amount():
    assert parse_amount("525.00") == 525.0
    assert parse_amount("RM1,234.50") == 1234.5
    assert parse_amount("$5.99*") == 5.99
    assert parse_amount("12,50") == 12.5
    for bad in ("525", "05/08/2026", "6%", "abc", "1.234.56"):
        assert parse_amount(bad) is None


def test_group_rows_merges_label_and_value_on_same_row():
    rec = make_record([(100, [(10, "TOTAL")]), (102, [(400, "525.00")]), (200, [(10, "CASH")])])
    rows = group_rows(rec["words"])
    assert [len(r) for r in rows] == [2, 1]


def test_typical_receipt():
    rec = make_record([
        (20, [(150, "FRESH"), (230, "MART")]),
        (50, [(150, "SDN"), (200, "BHD")]),
        (90, [(20, "Tel:"), (80, "03-1234567")]),
        (130, [(20, "Date:"), (90, "05/08/2026"), (250, "14:32")]),
        (300, [(20, "Milk"), (400, "4.50")]),
        (330, [(20, "Bread"), (400, "3.00")]),
        (500, [(20, "Subtotal"), (400, "7.50")]),
        (530, [(20, "GST"), (80, "6%"), (400, "0.45")]),
        (560, [(20, "TOTAL"), (400, "7.95")]),
        (590, [(20, "CASH"), (400, "10.00")]),
        (620, [(20, "Change"), (400, "2.05")]),
    ])
    vals, flags = labeled(rec)
    assert vals == {"MERCHANT": "FRESH MART SDN BHD", "DATE": "05/08/2026", "TAX": "0.45", "TOTAL": "7.95"}
    assert flags == []


def test_total_value_on_next_row_and_month_name_date():
    rec = make_record([
        (20, [(100, "CORNER"), (200, "CAFE")]),
        (60, [(20, "12"), (50, "Aug"), (100, "2019")]),
        (400, [(20, "GRAND"), (90, "TOTAL")]),
        (430, [(400, "1,250.00")]),
    ])
    vals, _ = labeled(rec)
    assert vals["DATE"] == "12 Aug 2019"
    assert vals["TOTAL"] == "1,250.00"
    assert vals["MERCHANT"] == "CORNER CAFE"


def test_flags_when_fields_missing_or_ambiguous():
    rec = make_record([(20, [(10, "hello"), (80, "world")]), (60, [(10, "item"), (300, "3.00")])])
    _, flags = labeled(rec)
    assert {"no_total", "no_date", "no_tax"} <= set(flags)

    rec = make_record([
        (20, [(10, "STORE")]),
        (100, [(10, "TOTAL"), (300, "10.00")]),
        (140, [(10, "TOTAL"), (300, "12.00")]),
    ])
    vals, flags = labeled(rec)
    assert "multiple_totals" in flags and vals["TOTAL"] == "12.00"


def test_phone_and_invalid_dates_are_not_dates():
    rec = make_record([(20, [(10, "STORE")]), (60, [(10, "99/99/2020"), (200, "13/13/13")])])
    vals, flags = labeled(rec)
    assert "DATE" not in vals and "no_date" in flags


def test_bio_conversion():
    ents = ["O", "MERCHANT", "MERCHANT", "O", "TOTAL", "TAX", "TAX"]
    assert entities_to_bio(ents) == ["O", "B-MERCHANT", "I-MERCHANT", "O", "B-TOTAL", "B-TAX", "I-TAX"]
    assert set(ENTITIES) == {"MERCHANT", "DATE", "TOTAL", "TAX"}


# ---- dates glued to neighbours by the OCR (strings taken from real receipts) ----
import pytest  # noqa: E402

from xtracts.annotation.weak_labels import _has_embedded_date  # noqa: E402


@pytest.mark.parametrize("text", [
    "Date:05/02/2018", "Date11/09/2018",                      # glued to the label
    "19-03-201818:08:38", "08/02/20173:43:01PM", "25-01-20181810:28",   # glued to the time
    "18-03-1812:09SH01ZJ86",                                  # 2-digit year, glued to time and code
    "#19536001/06/2018",                                      # glued to an invoice number
    "2018-03-19", "19MAR2018",
])
def test_dates_glued_to_other_text_are_found(text):
    assert _has_embedded_date(text)


@pytest.mark.parametrize("text", [
    "TEL:03-40211233", "FAX03-40223233", "03-3456-7890", "1.2012-0029", "12.00SR", "23/4TOWELROD",
    "GSTD0019219747", "31802.016", "002043319-W", "3.80", "RM50.00", "1971", "99/99/9999",
])
def test_phone_numbers_prices_and_codes_are_not_dates(text):
    assert not _has_embedded_date(text)


def test_glued_date_is_labeled_as_the_whole_word():
    rec = make_record([(20, [(150, "SHOP")]), (130, [(20, "Date:05/02/2018"), (300, "Time.11:14.00")]),
                       (400, [(20, "TEL:03-40211233")])])
    vals, flags = labeled(rec)
    assert vals["DATE"] == "Date:05/02/2018" and "no_date" not in flags


# ---- second round: real strings from the first 51 reviewed receipts ----
@pytest.mark.parametrize("text", [
    "#90-REG#19-18/01/201817:09:21",        # an earlier invalid '19-18/01' must not hide '18/01/2018'
    "ORD#95-REG#19-24/07/201614:21:51",
    "#90-REG#19-18/01/2018",
    "20180428/191204",                      # yyyymmdd/hhmmss
    "05-06-201820:23:14",
    "21-03-1812:27", "19-04-1817:39", "06-04-1816:46SH01ZJ86",
])
def test_dates_missed_in_first_review_are_found(text):
    assert _has_embedded_date(text)


@pytest.mark.parametrize("text", [
    "WA10-24/72", "WA10-8/64", "GB12-21-22-36",   # product codes the first fix wrongly accepted
    "12-21-22-36", "A1-20181-3",
])
def test_product_codes_are_not_dates_round2(text):
    assert not _has_embedded_date(text)


def test_clean_date_beats_date_glued_to_time_when_no_label_row():
    rec = make_record([(20, [(150, "SHOP")]), (130, [(20, "05-06-2018")]), (700, [(20, "05-06-201820:23:14")])])
    vals, _ = labeled(rec)
    assert vals["DATE"] == "05-06-2018"


def test_gst_total_row_is_not_taken_as_the_receipt_total():
    rec = make_record([(20, [(150, "SHOP")]), (300, [(20, "GST"), (200, "TOTAL"), (400, "4.43")]),
                       (560, [(20, "TOTAL"), (400, "73.87")])])
    vals, _ = labeled(rec)
    assert vals["TOTAL"] == "73.87"


# ---- round 3: TAX / TOTAL layouts taken from the user's review (show_misses output)
def _tax_total(rows):
    vals, flags = labeled(make_record(rows))
    return vals, flags


def _rec(lines):
    """lines: list of 'word word word' strings -> one row each, words spaced left-to-right."""
    lines = ["FRESH MART SDN BHD", *lines]            # a merchant row, so the rules have a header to use
    rows = []
    for k, line in enumerate(lines):
        rows.append((20 + 40 * k, [(20 + 90 * j, w) for j, w in enumerate(line.split())]))
    return make_record(rows)


@pytest.mark.parametrize("text,val", [(".65", 0.65), ("RMO.47", 0.47), ("-RMO.02", 0.02), ("RM2.43", 2.43)])
def test_parse_amount_ocr_slips(text, val):
    assert parse_amount(text) == val


def test_parse_amount_one_decimal_only_when_loose():
    assert parse_amount("1.4") is None and parse_amount("1.4", loose=True) == 1.4


@pytest.mark.parametrize("lines,tax", [
    (["GST Summary Amount RM Tax RM)", "T@6% 28.75 1.73", "%07 0.00 0.00"], "1.73"),
    (["GSTSummary% Amount(RM) TaxRM", "SR 0.00 28.31 0.00", "i love jimart"], "0.00"),
    (["057 5uwry Amount(RM) Tax(RM)", "=6 10.85 .65", "Thank You"], ".65"),
    (["GST Summary Amaunt", "R% 27.45 1.4", "THANK YOU"], "1.4"),
    (["GST Sunmary AmountRM) TaxRM", "SR @0% 52.00 0.00"], "0.00"),
    (["TOTAL 21.00", "TOTAL INCLUDES 6%GST 2.20", "Change 11.10"], "2.20"),
    (["TOTAL 21.00", "TOTALINCLUDES6%GST 1.60", "Change 71.80"], "1.60"),
    (["GST TA AMT RM40.45", "GST 6% RM2.43", "TOTAL. RM42.88"], "RM2.43"),
    (["Total Sales(Exciuding GST) 59.43", "Total GST 3.57", "Total Sales (Inclusive of GsT): 63.00"], "3.57"),
    (["Sun-Total RM7.90", "GST 6% RMO.47", "Rounding -RMO.02", "TOTAL RM8.40"], "RMO.47"),
])
def test_tax_layouts_from_review(lines, tax):
    rec = _rec(lines)
    ents, _ = auto_label(rec)
    got = [w["text"] for w, e in zip(rec["words"], ents) if e == "TAX"]
    assert got == [tax]


def test_total_row_with_tax_column_gives_total_and_tax():
    rec = _rec(["EX/0S0%) 0.00 0.00", "TOTAL 73.87 4.43"])
    ents, _ = auto_label(rec)
    pick = lambda f: [w["text"] for w, e in zip(rec["words"], ents) if e == f]  # noqa: E731
    assert pick("TOTAL") == ["73.87"] and pick("TAX") == ["4.43"]


@pytest.mark.parametrize("lines,total", [
    (["TotalQy 1 21.20", "Total Sales Ecluding GST 20.00", "Total GST 1.20", "TotalSales 21.20", "(Inclusive of GsT)"], "21.20"),
    (["@6XRM13.80", "Jotal5ales (Inclusive G5T) RM 82.80", "CREDIT RM 82.80"], "82.80"),
    (["Tota1 RM inc.GST 4.90", "Cash 5.00", "Change 0.10"], "4.90"),
    (["Sun-Total RM7.90", "TOTAL RM8.40"], "RM8.40"),
])
def test_total_layouts_from_review(lines, total):
    rec = _rec(lines)
    ents, _ = auto_label(rec)
    got = [w["text"] for w, e in zip(rec["words"], ents) if e == "TOTAL"]
    assert got == [total]
