"""Xtracts expense app.  Run from the project root:

    streamlit run src/xtracts/app/streamlit_app.py

Upload a receipt -> OCR + LayoutLMv3 -> check/correct the fields -> save to SQLite -> dashboard + history.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # makes `xtracts` importable via streamlit run

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from xtracts.app import analytics, db  # noqa: E402
from xtracts.config import load_config, resolve  # noqa: E402

MAX_UPLOAD_MB, MIN_SIDE = 15, 300

st.set_page_config(page_title="Xtracts", page_icon="🧾", layout="wide")
cfg = load_config(os.environ.get("XTRACTS_CONFIG"))
DB_PATH = Path(os.environ.get("XTRACTS_DB") or resolve(cfg["paths"]["app_db"]))
UPLOADS = Path(os.environ.get("XTRACTS_UPLOADS") or resolve(cfg["paths"]["app_uploads_dir"]))
db.init_db(DB_PATH)


@st.cache_resource(show_spinner="Loading the receipt model (first time only)...")
def get_extractor():
    from xtracts.app.inference import ReceiptExtractor

    return ReceiptExtractor(config=cfg)


def check_upload(data: bytes, name: str) -> tuple[Image.Image | None, str | None]:
    """Return (image, None) or (None, reason) for a file that cannot be used."""
    if not data:
        return None, "The file is empty."
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"The file is larger than {MAX_UPLOAD_MB} MB."
    try:
        im = Image.open(io.BytesIO(data))
        im.verify()
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None, f"'{name}' is not a readable image (use JPG or PNG)."
    if min(im.size) < MIN_SIDE:
        return None, f"The image is too small ({im.size[0]}x{im.size[1]} px); at least {MIN_SIDE}px on each side is needed for OCR."
    return im, None


# ---------------------------------------------------------------------------------- upload page
def page_upload() -> None:
    st.header("Upload a receipt")
    up = st.file_uploader("Receipt image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Choose a receipt photo or scan to extract the merchant, date, total and tax.")
        return
    data = up.getvalue()
    img, problem = check_upload(data, up.name)
    if problem:
        st.error(problem)
        return
    h = db.file_hash(data)

    left, right = st.columns([1, 1])
    left.image(img, caption=up.name)

    state = st.session_state.setdefault("extraction", {})
    if state.get("hash") != h:
        state.clear()
        state["hash"] = h
    with right:
        if "result" not in state and st.button("Extract fields", type="primary"):
            try:
                ext = get_extractor()
                with st.spinner("Reading the receipt (OCR + LayoutLMv3)..."):
                    with tempfile.TemporaryDirectory() as td:
                        p = Path(td) / f"{h[:12]}.png"
                        img.save(p)
                        state["result"] = ext.extract(p)
                state["load_error"] = ext.load_error
            except Exception as e:  # OCR engine missing, model failure, ...
                state["result"] = {"mode": "failed", "fields": {}, "raw": {}, "n_words": 0}
                state["error"] = f"{type(e).__name__}: {e}"[:300]
        if "result" not in state:
            return
        res = state["result"]
        if res["mode"] == "model":
            st.success("Fields predicted by the fine-tuned LayoutLMv3 model.")
        elif res["mode"] == "rules":
            st.warning("Rule-based fallback: no trained model was found, so these values come from the "
                       f"regex/heuristic labeler, not from LayoutLMv3. ({state.get('load_error')})")
        else:
            st.error(f"Extraction failed: {state.get('error')}. You can still type the values below.")
        if res["n_words"] == 0 and res["mode"] != "failed":
            st.warning("No text was recognised in this image. Fill the fields in by hand.")

        f = res["fields"]
        st.subheader("Check and correct")
        with st.form("confirm"):
            merchant = st.text_input("Merchant", f.get("merchant", ""))
            d = st.text_input("Date (YYYY-MM-DD)", f.get("date", ""))
            total = st.number_input("Total", min_value=0.0, value=float(f.get("total") or 0.0), step=0.01, format="%.2f")
            tax = st.number_input("Tax (0 if none)", min_value=0.0, value=float(f.get("tax") or 0.0), step=0.01, format="%.2f")
            saved = st.form_submit_button("Save expense", type="primary")
        missing = [k for k, v in (("merchant", f.get("merchant")), ("date", f.get("date")),
                                  ("total", f.get("total"))) if not v]
        if missing:
            st.caption(f"Not found automatically: {', '.join(missing)}. Please fill them in.")
        if saved:
            try:
                UPLOADS.mkdir(parents=True, exist_ok=True)
                fname = f"{h[:16]}.png"
                new_id = db.add_expense(DB_PATH, merchant, d, total, tax if tax > 0 else None,
                                        image_name=fname, image_hash=h,
                                        source="manual" if res["mode"] == "failed" else res["mode"])
                img.save(UPLOADS / fname)
                st.success(f"Saved as expense #{new_id}. See the Dashboard and History pages.")
            except db.DuplicateExpense as e:
                st.warning(str(e))
            except db.InvalidExpense as e:
                st.error(str(e))


# ---------------------------------------------------------------------------------- dashboard
def date_range_filter(key: str, df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty:
        return None, None
    lo, hi = pd.to_datetime(df["date"]).min().date(), pd.to_datetime(df["date"]).max().date()
    rng = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi, key=key)
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        return rng[0].isoformat(), rng[1].isoformat()
    return None, None


def page_dashboard() -> None:
    st.header("Expense dashboard")
    everything = db.load_expenses(DB_PATH)
    if everything.empty:
        st.info("No expenses saved yet. Upload a receipt and save it to see your spending here.")
        return
    d1, d2 = date_range_filter("dash_range", everything)
    df = db.load_expenses(DB_PATH, d1, d2)
    k = analytics.kpis(df)
    c = st.columns(4)
    c[0].metric("Total spent", f"{k['total_spent']:,.2f}")
    c[1].metric("Transactions", k["transactions"])
    c[2].metric("Average per receipt", f"{k['average']:,.2f}")
    c[3].metric("Total tax", f"{k['total_tax']:,.2f}")
    if df.empty:
        st.info("No expenses in this date range.")
        return
    a, b = st.columns(2)
    a.subheader("Monthly spending")
    a.bar_chart(analytics.monthly(df))
    b.subheader("Daily spending")
    b.line_chart(analytics.daily(df))
    st.subheader("Spending by merchant")
    m = analytics.by_merchant(df)
    st.bar_chart(m.set_index("merchant")["total"])
    st.dataframe(m.rename(columns={"merchant": "Merchant", "total": "Total", "visits": "Visits"}),
                 hide_index=True)


# ---------------------------------------------------------------------------------- history
def page_history() -> None:
    st.header("Transaction history")
    everything = db.load_expenses(DB_PATH)
    if everything.empty:
        st.info("No transactions yet.")
        return
    c1, c2 = st.columns([2, 1])
    with c1:
        d1, d2 = date_range_filter("hist_range", everything)
    q = c2.text_input("Merchant contains")
    df = db.load_expenses(DB_PATH, d1, d2, q or None)
    st.caption(f"{len(df)} transaction(s). Edit a cell, or tick Delete, then press Apply.")
    view = df[["id", "merchant", "date", "total", "tax"]].copy()
    view["delete"] = False
    edited = st.data_editor(
        view, hide_index=True, key="hist_editor", disabled=["id"],
        column_config={"total": st.column_config.NumberColumn(format="%.2f", min_value=0.0),
                       "tax": st.column_config.NumberColumn(format="%.2f", min_value=0.0)})
    if st.button("Apply changes", type="primary"):
        errors, changed, deleted = [], 0, 0
        to_delete = [int(r.id) for r in edited.itertuples() if r.delete]
        deleted = db.delete_expenses(DB_PATH, to_delete)
        orig = view.set_index("id")
        for r in edited.itertuples():
            if r.delete:
                continue
            o = orig.loc[r.id]
            tax = None if pd.isna(r.tax) else r.tax
            if (r.merchant, r.date, r.total, tax) != (o.merchant, o.date, o.total, None if pd.isna(o.tax) else o.tax):
                try:
                    db.update_expense(DB_PATH, int(r.id), r.merchant, str(r.date), r.total, tax)
                    changed += 1
                except (db.InvalidExpense, db.DuplicateExpense) as e:
                    errors.append(f"#{r.id}: {e}")
        for e in errors:
            st.error(e)
        if changed or deleted:
            st.success(f"Updated {changed}, deleted {deleted}.")
            st.rerun()
    st.download_button("Download as CSV", df.drop(columns=["image_hash"]).to_csv(index=False).encode("utf-8"),
                       "expenses.csv", "text/csv")


# ---------------------------------------------------------------------------------- main
with st.sidebar:
    st.title("🧾 Xtracts")
    page = st.radio("Go to", ["Upload & extract", "Dashboard", "History"])
    st.caption(f"Database: {DB_PATH.name}")

{"Upload & extract": page_upload, "Dashboard": page_dashboard, "History": page_history}[page]()
