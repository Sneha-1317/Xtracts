import os

import pytest

from xtracts.app import analytics, db
from xtracts.app.fields import clean_fields, normalize_amount, normalize_date, tags_to_fields


# ------------------------------------------------------------------ field cleaning
@pytest.mark.parametrize("text,iso", [("05/08/2026", "2026-08-05"), ("2026-08-05", "2026-08-05"),
                                      ("5 Aug 26", "2026-08-05"), ("13/01/18", "2018-01-13"),
                                      ("31-DEC-2017", "2017-12-31"), ("12/25/2019", "2019-12-25"),
                                      ("31/02/2020", None), ("hello", None), ("", None)])
def test_normalize_date(text, iso):
    assert normalize_date(text) == iso


@pytest.mark.parametrize("text,val", [("RM12.50", 12.5), ("1,234.50", 1234.5), (".65", 0.65), ("12", 12.0),
                                      ("abc", None), ("", None)])
def test_normalize_amount(text, val):
    assert normalize_amount(text) == val


def test_tags_to_fields_and_clean():
    words = ["FRESH", "MART", "Date", "05/08/2026", "TOTAL", "5.50", "TAX", "0.33"]
    tags = ["B-MERCHANT", "I-MERCHANT", "O", "B-DATE", "O", "B-TOTAL", "O", "B-TAX"]
    raw = tags_to_fields(words, tags)
    assert raw == {"MERCHANT": "FRESH MART", "DATE": "05/08/2026", "TOTAL": "5.50", "TAX": "0.33"}
    assert clean_fields(raw) == {"merchant": "FRESH MART", "date": "2026-08-05", "total": 5.5, "tax": 0.33}


def test_two_total_spans_pick_most_confident():
    words = ["TOTAL", "4.00", "TOTAL", "5.50"]
    tags = ["O", "B-TOTAL", "O", "B-TOTAL"]
    assert tags_to_fields(words, tags, [0, 0.6, 0, 0.9]) == {"TOTAL": "5.50"}
    assert tags_to_fields(words, tags, [0, 0.9, 0, 0.6]) == {"TOTAL": "4.00"}


def test_missing_fields_give_empty_values():
    assert clean_fields({}) == {"merchant": "", "date": "", "total": None, "tax": None}


# ------------------------------------------------------------------ database
@pytest.fixture()
def dbp(tmp_path):
    p = tmp_path / "x" / "t.db"
    db.init_db(p)
    return p


def test_empty_database(dbp):
    df = db.load_expenses(dbp)
    assert df.empty and list(df.columns) == db.COLUMNS
    k = analytics.kpis(df)
    assert k["transactions"] == 0 and k["total_spent"] == 0
    assert analytics.by_merchant(df).empty and analytics.monthly(df).empty and analytics.daily(df).empty


def test_add_update_delete_roundtrip(dbp):
    i = db.add_expense(dbp, " Fresh Mart ", "2026-08-05", 12.5, 0.7, image_hash="h1")
    row = db.load_expenses(dbp).iloc[0]
    assert (row.merchant, row.date, row.total, row.tax) == ("Fresh Mart", "2026-08-05", 12.5, 0.7)
    db.update_expense(dbp, i, "Fresh Mart", "2026-08-06", 13.0, None)
    row = db.load_expenses(dbp).iloc[0]
    assert row.date == "2026-08-06" and row.total == 13.0 and row.tax is None or row.tax != row.tax
    assert db.delete_expenses(dbp, [i]) == 1 and db.load_expenses(dbp).empty


def test_duplicates_are_refused(dbp):
    db.add_expense(dbp, "A", "2026-01-01", 10, image_hash="h")
    with pytest.raises(db.DuplicateExpense):
        db.add_expense(dbp, "B", "2026-01-02", 3, image_hash="h")          # same image
    with pytest.raises(db.DuplicateExpense):
        db.add_expense(dbp, "a", "2026-01-01", 10.0)                        # same merchant/date/total
    j = db.add_expense(dbp, "A", "2026-01-02", 10)
    with pytest.raises(db.DuplicateExpense):
        db.update_expense(dbp, j, "A", "2026-01-01", 10)                    # edit into an existing receipt


@pytest.mark.parametrize("args", [("", "2026-01-01", 1), ("A", "01/02/2026", 1), ("A", "2026-13-01", 1),
                                  ("A", "2026-01-01", "x"), ("A", "2026-01-01", -1), ("A", "2026-01-01", 5, 9)])
def test_invalid_input_is_rejected(dbp, args):
    with pytest.raises(db.InvalidExpense):
        db.add_expense(dbp, *args)
    assert db.load_expenses(dbp).empty


def test_filters_and_analytics(dbp):
    for m, d, t, x in [("A Mart", "2026-08-02", 10, 0.6), ("a mart", "2026-09-10", 20, 1.2), ("B Cafe", "2026-09-15", 5.5, None)]:
        db.add_expense(dbp, m, d, t, x)
    assert len(db.load_expenses(dbp, "2026-09-01", "2026-09-30")) == 2
    assert len(db.load_expenses(dbp, merchant="cafe")) == 1
    df = db.load_expenses(dbp)
    k = analytics.kpis(df)
    assert k["transactions"] == 3 and k["total_spent"] == 35.5 and k["merchants"] == 2
    assert k["total_tax"] == pytest.approx(1.8)
    assert analytics.monthly(df).to_dict() == {"2026-08": 10.0, "2026-09": 25.5}
    m = analytics.by_merchant(df)
    assert list(m["total"]) == [30.0, 5.5] and list(m["visits"]) == [2, 1]


# ------------------------------------------------------------------ the pages render on real DB content
def _page_test(dbp, monkeypatch, page):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("XTRACTS_DB", str(dbp))
    app = os.path.join(os.path.dirname(__file__), "..", "src", "xtracts", "app", "streamlit_app.py")
    at = AppTest.from_file(app, default_timeout=60).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
    return at


def test_dashboard_page_empty_and_filled(dbp, monkeypatch):
    at = _page_test(dbp, monkeypatch, "Dashboard")
    assert any("No expenses saved" in i.value for i in at.info)
    db.add_expense(dbp, "A Mart", "2026-08-02", 10, 0.6)
    at = _page_test(dbp, monkeypatch, "Dashboard")
    assert [m.value for m in at.metric][:2] == ["10.00", "1"]


def test_history_page_empty_and_filled(dbp, monkeypatch):
    at = _page_test(dbp, monkeypatch, "History")
    assert any("No transactions" in i.value for i in at.info)
    db.add_expense(dbp, "A Mart", "2026-08-02", 10, 0.6)
    _page_test(dbp, monkeypatch, "History")
