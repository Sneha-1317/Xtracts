"""Label review app.  Run from the project root:

    streamlit run src/xtracts/annotation/review_app.py

For each receipt: check the coloured boxes on the image and fix the `entity` column in the table
(MERCHANT / DATE / TOTAL / TAX / O). Then press "Save as reviewed".
Only reviewed receipts are used for training.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # makes `xtracts` importable via streamlit run

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from xtracts.annotation.labels import ENTITIES, ENTITY_CHOICES, entity_values, load_label, save_label  # noqa: E402
from xtracts.annotation.review_utils import (  # noqa: E402
    COLORS, agreement_stats, build_queue, draw_boxes, load_all_labels)
from xtracts.config import load_config, resolve  # noqa: E402

st.set_page_config(page_title="Xtracts label review", layout="wide")

cfg = load_config(os.environ.get("XTRACTS_CONFIG"))
P = {k: resolve(v) for k, v in cfg["paths"].items()}
labels = load_all_labels(P["labels_dir"])
if not labels:
    st.error(f"No labels in {P['labels_dir']}. Run: python scripts/03_auto_label.py")
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Queue")
    mode = st.radio("Order", ["Flagged first", "Random", "File order"],
                    help="'Random' gives an unbiased view of how accurate the auto labels are.")
    include_reviewed = st.checkbox("Include already reviewed", value=False)
    n_rev = sum(l["reviewed"] for l in labels)
    n_exc = sum(l["exclude"] for l in labels)
    st.progress(n_rev / len(labels), text=f"Reviewed {n_rev} / {len(labels)}  (excluded: {n_exc})")

    st.subheader("Auto-label accuracy")
    stats = agreement_stats(labels)
    if stats["n_reviewed"] == 0:
        st.caption("Appears after you review some receipts.")
    else:
        st.caption(f"Share of reviewed receipts where the auto label needed no change (n={stats['n_reviewed']})")
        for e in ENTITIES:
            st.write(f"**{e}**: {100 * stats[e]:.0f}%")

    st.subheader("Legend")
    for e in ENTITIES:
        c = COLORS[e]
        st.markdown(f"<span style='color:rgb{c}'>■</span> {e}", unsafe_allow_html=True)

# queue + position live in session state so saving does not reshuffle the order
key = (mode, include_reviewed)
if st.session_state.get("queue_key") != key:
    st.session_state.queue_key = key
    st.session_state.queue = build_queue(labels, mode, include_reviewed)
    st.session_state.pos = 0
queue, pos = st.session_state.queue, st.session_state.pos

if not queue or pos >= len(queue):
    st.success("Queue finished. Change the order in the sidebar or tick 'Include already reviewed'.")
    st.stop()

# ------------------------------------------------------------------ current receipt
image_id = queue[pos]
label = load_label(P["labels_dir"], image_id)
rec = json.loads((P["ocr_dir"] / f"{image_id}.json").read_text(encoding="utf-8"))
words = rec["words"]
image = Image.open(P["clean_images_dir"] / rec["image"])

st.subheader(f"Receipt {pos + 1} / {len(queue)}   ·   {image_id}"
             + ("   ·   ✅ reviewed" if label["reviewed"] else "")
             + ("   ·   🚫 excluded" if label["exclude"] else ""))
if label["flags"]:
    st.warning("Check carefully: " + ", ".join(label["flags"]))

left, right = st.columns([1, 1])
df = pd.DataFrame({"idx": range(len(words)), "text": [w["text"] for w in words], "entity": label["entities"]})
with right:
    edited = st.data_editor(
        df, key=f"editor_{image_id}", hide_index=True, height=640, use_container_width=True,
        disabled=["idx", "text"],
        column_config={"entity": st.column_config.SelectboxColumn("entity", options=ENTITY_CHOICES, required=True)},
    )
entities = edited["entity"].tolist()
with left:
    st.image(draw_boxes(image, words, entities), use_container_width=True)

vals = entity_values([w["text"] for w in words], entities)
st.markdown("  |  ".join(f"**{e}:** {vals.get(e, '—')}" for e in ENTITIES))


def go(delta: int) -> None:
    st.session_state.pos = max(0, st.session_state.pos + delta)


c1, c2, c3, c4 = st.columns(4)
if c1.button("✅ Save as reviewed", type="primary", use_container_width=True):
    label.update(entities=entities, reviewed=True, exclude=False)
    save_label(P["labels_dir"], label)
    go(1)
    st.rerun()
if c2.button("🚫 Exclude receipt", use_container_width=True, help="unreadable / not a receipt"):
    label.update(reviewed=True, exclude=True)
    save_label(P["labels_dir"], label)
    go(1)
    st.rerun()
if c3.button("⏭ Skip", use_container_width=True):
    go(1)
    st.rerun()
if c4.button("⏮ Back", use_container_width=True):
    go(-1)
    st.rerun()
