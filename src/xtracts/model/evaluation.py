"""Stage 6 helpers: value-level scoring and report writing (pure Python, no torch)."""
from __future__ import annotations

import csv
import re
from pathlib import Path

from ..annotation.labels import ENTITIES
from ..app.fields import normalize_amount, normalize_date, tags_to_fields


def _norm(field: str, text: str | None):
    """Comparable form of a field value: merchant = letters/digits only (case-insensitive),
    date = ISO date, amounts = number rounded to 2 decimals."""
    if not text:
        return None
    if field == "MERCHANT":
        return re.sub(r"[^a-z0-9]", "", text.lower()) or None
    if field == "DATE":
        return normalize_date(text) or re.sub(r"\s", "", text)
    v = normalize_amount(text)
    return round(v, 2) if v is not None else re.sub(r"\s", "", text)


def value_rows(rows: list[dict], pred_tags: list[list[str]]) -> list[dict]:
    """Per receipt and field: gold text, predicted text and whether the normalised values agree.
    A receipt with no gold value and no prediction for a field counts as a match."""
    out = []
    for r, pt in zip(rows, pred_tags):
        gold = tags_to_fields(r["words"], r["labels"])
        pred = tags_to_fields(r["words"], pt)
        for f in ENTITIES:
            g, p = gold.get(f, ""), pred.get(f, "")
            out.append({"image_id": r["image_id"], "field": f, "gold": g, "pred": p,
                        "match": _norm(f, g) == _norm(f, p)})
    return out


def value_accuracy(vrows: list[dict]) -> dict[str, dict]:
    """{field: {"exact": share of receipts whose value matches, "n": receipts, "n_with_gold": ...}}"""
    res = {}
    for f in ENTITIES:
        sel = [v for v in vrows if v["field"] == f]
        res[f] = {"value_exact": sum(v["match"] for v in sel) / max(1, len(sel)), "n": len(sel),
                  "n_with_gold": sum(bool(v["gold"]) for v in sel)}
    return res


def flatten(system: str, scores: dict, values: dict) -> list[dict]:
    """One CSV row per field (+ an ALL row) for one system."""
    rows = []
    for f in ENTITIES:
        m, v = scores["fields"][f], values[f]
        rows.append({"system": system, "field": f, "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "span_exact_match": m["exact_match"], "value_exact_match": v["value_exact"],
                     "support": m["support"], "receipts": v["n"]})
    m = scores["micro"]
    rows.append({"system": system, "field": "ALL (micro)", "precision": m["precision"], "recall": m["recall"],
                 "f1": m["f1"], "span_exact_match": "", "value_exact_match": "", "support": m["support"],
                 "receipts": scores["n_receipts"]})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def markdown_summary(meta: dict, table_rows: list[dict]) -> str:
    def pct(x):
        return f"{100 * x:.1f}%" if isinstance(x, float) else "-"

    lines = [
        "# Stage 6 - Evaluation results", "",
        f"- Evaluation set: **{meta['split']} split, {meta['n_receipts']} receipts** whose labels were corrected by a human "
        f"(merchant-grouped split, never used for training or model selection{'' if meta['split'] == 'test' else ' - NOT true for val'}).",
        f"- Model: `{meta.get('model', 'n/a')}`; OCR words come from PaddleOCR (the same OCR the model was trained on).",
        f"- Generated: {meta['generated']}", "",
        "| System | Field | Precision | Recall | F1 | Value exact match | Receipts with this field |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in table_rows:
        lines.append(f"| {r['system']} | {r['field']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} | "
                     f"{pct(r['value_exact_match'])} | {r['support']} |")
    lines += ["", "## How to read this", "",
              "- **Precision / recall / F1** are entity-level: a prediction counts only if the field AND the exact word span match "
              "the human label (the standard CoNLL/seqeval definition).",
              "- **Value exact match** compares the extracted value after normalisation (case/punctuation-insensitive merchant, ISO date, "
              "amount to 2 decimals) - the number that matters to a user of the app.",
              "- **Rules** is the regex/heuristic labeler that produced the annotation suggestions; **LayoutLMv3** is the fine-tuned model.", "",
              "## Limitations (read before quoting these numbers)", "",
              f"- Small evaluation set ({meta['n_receipts']} receipts): differences of a few points are within noise.",
              "- Human labels were made by correcting the rules' suggestions, so annotators may be anchored to them. "
              "This favours the Rules baseline; it does not favour the model.",
              "- The reviewed receipts are not a perfectly random sample (the first ones reviewed were the most flagged), "
              "and the data is Malaysian SROIE-style receipts only.",
              "- These numbers are NOT the 'auto-label accuracy' shown in the review app, which measures the rules, not the model."]
    return "\n".join(lines) + "\n"
