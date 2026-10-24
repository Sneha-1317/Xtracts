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

ACCENT, ACCENT2, SOFT = "#4f46e5", "#06b6d4", "#e0e7ff"
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg, #4f46e5 0%, #06b6d4 100%); color: white; padding: 1.3rem 1.6rem;
       border-radius: 18px; margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(79,70,229,.25);}
.hero h1 {color: white; margin: 0; font-size: 1.9rem; padding: 0;}
.hero p {margin: .25rem 0 0 0; opacity: .92; font-size: 1rem;}
.steps {display: flex; gap: .6rem; margin: .2rem 0 1rem 0; flex-wrap: wrap;}
.step {background: white; border: 1px solid #e5e7eb; border-radius: 999px; padding: .35rem .9rem; font-size: .88rem; color: #4b5563;}
.step b {background: #4f46e5; color: white; border-radius: 50%; display: inline-block; width: 1.4rem; height: 1.4rem;
         text-align: center; line-height: 1.4rem; margin-right: .4rem; font-size: .8rem;}
.pill {display: inline-block; padding: .18rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 600; margin-right: .4rem;}
.pill.ok {background: #dcfce7; color: #166534;} .pill.warn {background: #fef3c7; color: #92400e;}
.pill.bad {background: #fee2e2; color: #991b1b;} .pill.info {background: #e0e7ff; color: #3730a3;}
[data-testid="stMetric"] {background: white; border: 1px solid #e5e7eb; border-radius: 16px; padding: 1rem 1.2rem;
                          box-shadow: 0 2px 10px rgba(17,24,39,.05);}
[data-testid="stMetricLabel"] {color: #6b7280;} [data-testid="stMetricValue"] {color: #1f2937;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}
[data-testid="stSidebar"] {background: white; border-right: 1px solid #e5e7eb;}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {border-radius: 10px; font-weight: 600;}
h2, h3 {color: #1f2937;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill {kind}">{text}</span>'


def style_chart(chart, height: int = 280):
    return chart.properties(height=height, width="container").configure_view(strokeWidth=0).configure_axis(
        grid=True, gridColor="#eef0f5", domain=False, labelColor="#6b7280", titleColor="#6b7280")


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
    hero("Scan a receipt", "Upload a photo, check what was read, and add it to your expenses in seconds.")
    st.markdown('<div class="steps"><span class="step"><b>1</b>Upload</span><span class="step"><b>2</b>Extract</span>'
                '<span class="step"><b>3</b>Check &amp; correct</span><span class="step"><b>4</b>Save</span></div>',
                unsafe_allow_html=True)
    up = st.file_uploader("Drop a receipt image here (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Choose a receipt photo or scan to extract the merchant, date, total and tax.")
        return
    data = up.getvalue()
    img, problem = check_upload(data, up.name)
    if problem:
        st.error(problem)
        return
    h = db.file_hash(data)

    left, right = st.columns([1, 1], gap="large")
    with left.container(border=True):
        st.image(img, caption=up.name)

    state = st.session_state.setdefault("extraction", {})
    if state.get("hash") != h:
        state.clear()
        state["hash"] = h
    with right:
        if "result" not in state and st.button("✨ Extract fields", type="primary"):
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
            st.markdown(pill("LayoutLMv3 model", "ok") + pill(f"{res['n_words']} words read", "info"), unsafe_allow_html=True)
        elif res["mode"] == "rules":
            st.markdown(pill("Rule-based fallback", "warn"), unsafe_allow_html=True)
            st.warning("No trained model was found, so these values come from the regex/heuristic labeler, "
                       f"not from LayoutLMv3. ({state.get('load_error')})")
        else:
            st.markdown(pill("Extraction failed", "bad"), unsafe_allow_html=True)
            st.error(f"{state.get('error')}. You can still type the values below.")
        if res["n_words"] == 0 and res["mode"] != "failed":
            st.warning("No text was recognised in this image. Fill the fields in by hand.")

        f = res["fields"]
        st.subheader("Check and correct")
        with st.form("confirm", border=True):
            merchant = st.text_input("Merchant", f.get("merchant", ""))
            d = st.text_input("Date (YYYY-MM-DD)", f.get("date", ""))
            total = st.number_input("Total", min_value=0.0, value=float(f.get("total") or 0.0), step=0.01, format="%.2f")
            tax = st.number_input("Tax (0 if none)", min_value=0.0, value=float(f.get("tax") or 0.0), step=0.01, format="%.2f")
            saved = st.form_submit_button("💾 Save expense", type="primary")
        missing = [k for k, v in (("merchant", f.get("merchant")), ("date", f.get("date")),
                                  ("total", f.get("total"))) if not v]
        if missing:
            st.markdown(pill("Please fill in: " + ", ".join(missing), "warn"), unsafe_allow_html=True)
        if saved:
            try:
                UPLOADS.mkdir(parents=True, exist_ok=True)
                fname = f"{h[:16]}.png"
                new_id = db.add_expense(DB_PATH, merchant, d, total, tax if tax > 0 else None,
                                        image_name=fname, image_hash=h,
                                        source="manual" if res["mode"] == "failed" else res["mode"])
                img.save(UPLOADS / fname)
                st.success(f"Saved as expense #{new_id}. See the Dashboard and History pages.", icon="✅")
                st.toast("Expense saved", icon="🧾")
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
    hero("Spending dashboard", "Where your money went, based on the receipts you saved.")
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
    import altair as alt

    a, b = st.columns(2, gap="large")
    with a.container(border=True):
        st.subheader("Monthly spending")
        mo = analytics.monthly(df).rename("Total").reset_index().rename(columns={"index": "Month", "date": "Month"})
        mo.columns = ["Month", "Total"]
        st.altair_chart(style_chart(alt.Chart(mo).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=ACCENT).encode(
            x=alt.X("Month:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Total:Q", title=None),
            tooltip=["Month", alt.Tooltip("Total:Q", format=",.2f")])))
    with b.container(border=True):
        st.subheader("Daily spending")
        da = analytics.daily(df).rename("Total").reset_index()
        da.columns = ["Date", "Total"]
        base = alt.Chart(da).encode(x=alt.X("Date:T", title=None), y=alt.Y("Total:Q", title=None),
                                    tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Total:Q", format=",.2f")])
        st.altair_chart(style_chart(base.mark_area(opacity=.18, color=ACCENT2) + base.mark_line(color=ACCENT2, strokeWidth=2.5)
                                    + base.mark_circle(color=ACCENT2, size=60)))
    with st.container(border=True):
        st.subheader("Spending by merchant")
        m = analytics.by_merchant(df)
        c1, c2 = st.columns([3, 2], gap="large")
        c1.altair_chart(style_chart(alt.Chart(m).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color=ACCENT).encode(
            y=alt.Y("merchant:N", sort="-x", title=None), x=alt.X("total:Q", title=None),
            tooltip=["merchant", alt.Tooltip("total:Q", format=",.2f"), "visits"]), height=max(180, 32 * len(m))))
        c2.dataframe(m.rename(columns={"merchant": "Merchant", "total": "Total", "visits": "Visits"}), hide_index=True)


# ---------------------------------------------------------------------------------- history
def page_history() -> None:
    hero("Transaction history", "Search, correct or remove saved expenses.")
    everything = db.load_expenses(DB_PATH)
    if everything.empty:
        st.info("No transactions yet.")
        return
    c1, c2 = st.columns([2, 1])
    with c1:
        d1, d2 = date_range_filter("hist_range", everything)
    q = c2.text_input("Merchant contains")
    df = db.load_expenses(DB_PATH, d1, d2, q or None)
    st.markdown(pill(f"{len(df)} transactions", "info") + pill(f"Total {df['total'].sum():,.2f}", "ok")
                + "&nbsp; Edit a cell, or tick Delete, then press Apply.", unsafe_allow_html=True)
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
MODEL_READY = (resolve(cfg["paths"]["model_dir"]) / "best" / "config.json").exists()
with st.sidebar:
    st.markdown("## 🧾 Xtracts")
    st.caption("Receipt scanner and expense tracker")
    page = st.radio("Go to", ["Upload & extract", "Dashboard", "History"], label_visibility="collapsed")
    st.divider()
    n_saved = len(db.load_expenses(DB_PATH))
    st.markdown(pill(f"{n_saved} saved", "info") + (pill("LayoutLMv3 ready", "ok") if MODEL_READY else pill("Rules fallback", "warn")),
                unsafe_allow_html=True)
    st.caption(f"Database: {DB_PATH.name}")

{"Upload & extract": page_upload, "Dashboard": page_dashboard, "History": page_history}[page]()
