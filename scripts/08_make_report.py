"""Stage 8: write the academic report from the sample report + YOUR real project outputs.

Reads  docs/Xtracts_Mini_Project_Report_TEMPLATE.docx   (the original sample, never modified)
Writes docs/Xtracts_Mini_Project_Report_UPDATED.docx    (a separate file)

Numbers come only from files your own runs produced:
    data/processed/splits/*.jsonl         split sizes
    data/annotations/labels/*.json        number of human-reviewed receipts
    models/layoutlmv3-receipts/summary.json   training summary
    reports/evaluation/metrics.json + per_receipt.csv   Stage 6 results (scripts/07_evaluate.py)
Anything that does not exist yet stays as a visible [bracketed placeholder] and is listed at the end.
Run it again after each new result - it is safe to repeat.

Usage:  python scripts/08_make_report.py
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import docx  # noqa: E402

from xtracts.annotation.labels import load_all_label_files  # noqa: E402
from xtracts.config import load_config, resolve  # noqa: E402

FIELDS = ["MERCHANT", "DATE", "TOTAL", "TAX"]
NICE = {"MERCHANT": "Merchant", "DATE": "Date", "TOTAL": "Total", "TAX": "Tax"}
TODO = "[to fill: {}]"


# ------------------------------------------------------------------ docx helpers
def set_runs(p, texts: list[str]) -> None:
    """Put texts into the paragraph's existing runs (keeps their formatting), clear the rest."""
    runs = p.runs
    if not runs:
        p.add_run("")
        runs = p.runs
    for i, r in enumerate(runs):
        r.text = texts[i] if i < len(texts) else ""
    for t in texts[len(runs):]:
        runs[-1].text += t


def set_body(p, body: str) -> None:
    """Replace the text but keep a bold lead-in run ('Objectives.')."""
    runs = p.runs
    if len(runs) > 1 and runs[0].bold:
        set_runs(p, [runs[0].text, " " + body.lstrip()])
    else:
        set_runs(p, [body])


def set_cell(cell, text: str) -> None:
    p = cell.paragraphs[0]
    set_runs(p, [str(text)])
    for extra in cell.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def delete_row(table, idx: int) -> None:
    table._tbl.remove(table.rows[idx]._tr)


def find_par(doc, startswith: str):
    for p in doc.paragraphs:
        if p.text.strip().startswith(startswith):
            return p
    raise KeyError(startswith)


# ------------------------------------------------------------------ data gathering
def jl_count(path: Path) -> tuple[int, dict]:
    if not path.exists():
        return 0, {}
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    has = {f: sum(any(t == f"B-{f}" for t in r["labels"]) for r in rows) for f in FIELDS}
    return len(rows), has


def gather(root_cfg: dict) -> dict:
    P = {k: resolve(v) for k, v in root_cfg["paths"].items()}
    g: dict = {"splits": {}, "metrics": None, "summary": None, "reviewed": None, "per": None, "ocr_conf": None}
    for s in ("train", "val", "test"):
        n, has = jl_count(P["splits_dir"] / f"{s}.jsonl")
        if n:
            g["splits"][s] = (n, has)
    if P["labels_dir"].exists():
        labs = load_all_label_files(P["labels_dir"])
        g["reviewed"] = sum(1 for l in labs if l["reviewed"] and not l["exclude"])
    if (P["model_dir"] / "summary.json").exists():
        g["summary"] = json.loads((P["model_dir"] / "summary.json").read_text())
    if (P["eval_dir"] / "metrics.json").exists():
        g["metrics"] = json.loads((P["eval_dir"] / "metrics.json").read_text())
    if (P["eval_dir"] / "per_receipt.csv").exists():
        with open(P["eval_dir"] / "per_receipt.csv", encoding="utf-8") as f:
            g["per"] = list(csv.DictReader(f))
    if P["ocr_summary_csv"].exists():
        with open(P["ocr_summary_csv"], encoding="utf-8") as f:
            vals = [float(r["mean_conf"]) for r in csv.DictReader(f) if r.get("mean_conf")]
        g["ocr_conf"] = (sum(vals) / len(vals), len(vals)) if vals else None
    return g


def pct(x) -> str:
    return f"{100 * x:.1f}%"


def system_stats(g: dict) -> dict:
    """{system: {field: {p,r,f1,exact}, 'micro': {...}, 'mean_exact': x}}"""
    out = {}
    if not g["metrics"]:
        return out
    for name, s in g["metrics"]["systems"].items():
        ent, val = s["entity_level"], s["value_level"]
        d = {f: {"p": ent["fields"][f]["precision"], "r": ent["fields"][f]["recall"], "f1": ent["fields"][f]["f1"],
                 "exact": val[f]["value_exact"]} for f in FIELDS}
        d["micro"] = ent["micro"]
        d["mean_exact"] = sum(val[f]["value_exact"] for f in FIELDS) / len(FIELDS)
        out[name] = d
    return out


# ------------------------------------------------------------------ diagram
def make_flow_png(width: int, height: int) -> bytes:
    """Two-row block diagram of the implemented pipeline, drawn at the size of the template's figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    labels = ["Receipt images\n(2,586 files)", "Cleaning and\nde-duplication\n(865 unique)", "PaddleOCR\nwords + boxes",
              "Rule-based labels\n+ human review", "Fine-tune\nLayoutLMv3", "Evaluation\n+ MLflow", "Streamlit app\n+ SQLite database"]
    pos = [(0.125, 0.74), (0.375, 0.74), (0.625, 0.74), (0.875, 0.74), (0.2, 0.26), (0.5, 0.26), (0.8, 0.26)]
    fig, ax = plt.subplots(figsize=(width / 150, height / 150), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bw, bh = 0.2, 0.3
    for i, ((x, y), t) in enumerate(zip(pos, labels), start=1):
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.008", fc="#e8f0fe", ec="#2b4c9b", lw=1.3))
        ax.text(x, y, t, ha="center", va="center", fontsize=8)
        ax.text(x - bw / 2 + 0.012, y + bh / 2 - 0.035, str(i), fontsize=7, color="#2b4c9b", weight="bold", va="center")
    arrow = dict(arrowstyle="-|>", color="#2b4c9b", lw=1.3)
    for i in (0, 1, 2, 4, 5):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        ax.annotate("", xy=(x1 - bw / 2 - 0.012, y1), xytext=(x0 + bw / 2 + 0.012, y0), arrowprops=arrow)
    ax.plot([0.875, 0.875, 0.2, 0.2][:2], [0.74 - bh / 2 - 0.01, 0.5], color="#2b4c9b", lw=1.3)
    ax.plot([0.875, 0.2], [0.5, 0.5], color="#2b4c9b", lw=1.3)
    ax.annotate("", xy=(0.2, 0.26 + bh / 2 + 0.014), xytext=(0.2, 0.5), arrowprops=arrow)
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------ the report
def build(template: Path, out: Path, g: dict, cfg: dict) -> list[str]:
    doc = docx.Document(str(template))
    hp, ocr = cfg["training"], cfg["ocr"]
    st = system_stats(g)
    n_rev = g["reviewed"]
    best = max(st, key=lambda k: st[k]["micro"]["f1"]) if "LayoutLMv3" in st else None   # discussion needs a trained model
    T = doc.tables
    todo_note = lambda what: TODO.format(what)  # noqa: E731

    # ---- 1. introduction
    set_runs(find_par(doc, "This project, "), [
        "This project, ", "Xtracts",
        ", is an OCR-based receipt information extraction and expense analytics system. It was developed on a collection "
        "of 2,586 scanned Malaysian receipt images of the kind distributed with SROIE [1], [12], from which 865 unique "
        "receipts remained after duplicate removal. Receipt text and layout are read with PaddleOCR, and a LayoutLMv3 "
        "model [11] fine-tuned for token classification labels the words belonging to four fields: merchant, date, total "
        "and tax. A Streamlit application lets the user upload a receipt, correct the extracted values, save them to a "
        "SQLite database and view spending dashboards and a transaction history. The outcome is a reproducible pipeline "
        "from unstructured receipt images to structured expense records, together with a quantitative evaluation of "
        "how accurately the fields are extracted."])
    set_body(find_par(doc, "Scope and limitations."),
             "The target fields are merchant, date, total and tax. The images come without field annotations, so the ground "
             "truth was created within this project: rule-based suggestions were corrected by a human reviewer in a review "
             "application. Line items and expense categories are not extracted; the analytics therefore cover total spending, "
             "transaction counts, daily and monthly spending and spending by merchant. Built in this project: the cleaning "
             "and de-duplication pipeline, the OCR post-processing, the rule-based labeler, the review application, the "
             "training and evaluation code, the inference pipeline, the database and the application. Reused from existing "
             "libraries: PaddleOCR (text detection and recognition) and the pre-trained LayoutLMv3-base weights.")
    set_body(find_par(doc, "Reason for selecting the approach"),
             "SROIE-style Malaysian receipts were chosen because they are a well-known benchmark domain that covers the "
             "complete chain from OCR to key-information extraction [1]. A layout-aware transformer was chosen because the "
             "total and tax cannot be found reliably from text alone: receipts contain subtotals, tax summaries, cash "
             "tendered and change, and the position of a number on the page carries much of the evidence. A rule-based "
             "extractor was kept as a transparent baseline against which the learned model is measured.")

    # ---- 4. dataset
    set_body(find_par(doc, "The project uses the SROIE"),
             "The project uses a collection of 2,586 scanned receipt images of the kind released for the ICDAR 2019 SROIE "
             "competition [1], [12]. The official competition annotation files were not used: the three SROIE tasks are "
             "replaced by a single token-classification task (merchant, date, total, tax) on words obtained from our own OCR run, "
             "with ground truth created and reviewed within this project.")
    set_body(find_par(doc, "All statistics that depend"),
             "After removing corrupt/too-small images and duplicates (exact copies by SHA-256 and near-duplicates by "
             "perceptual hashing), 865 unique receipts remained; the statistics below were measured on these files.")
    t2 = T[2]
    set_cell(t2.rows[1].cells[1], "SROIE-style Malaysian receipts, the provided image set of this project [state the exact source of your copy]; official page cited in [12]")
    set_cell(t2.rows[2].cells[1], "Scanned receipt images; OCR words with boxes (our own PaddleOCR run); field labels created by rule-based suggestion plus human review")
    set_cell(t2.rows[3].cells[0], "Number of records (images)")
    set_cell(t2.rows[3].cells[1], "2,586 image files; 865 unique receipts after cleaning and de-duplication (1,721 removed: 117 exact and 1,604 near-duplicates)")
    sp = g["splits"]
    set_cell(t2.rows[4].cells[0], "Training / validation / testing receipts")
    set_cell(t2.rows[4].cells[1], " / ".join(str(sp[s][0]) if s in sp else todo_note("run 04_build_dataset.py") for s in ("train", "val", "test"))
             + (f" (human-reviewed receipts only; {n_rev} reviewed in total)" if n_rev is not None else ""))
    set_cell(t2.rows[5].cells[1], "Per word: text, bounding box (normalised to 0-1000) and, via the image, visual content; the token-classification label")
    set_cell(t2.rows[6].cells[1], "Merchant, date, total, tax (BIO-tagged word spans)")
    set_cell(t2.rows[7].cells[1], "Unreadable or under-300-pixel images were excluded; receipts the reviewer marked unusable were excluded from training")
    set_cell(t2.rows[8].cells[1], "117 exact and 1,604 near-duplicate images were found and removed, keeping one image per cluster (largest cluster: 54 images)")
    set_cell(t2.rows[9].cells[1], "[Check and state the dataset's terms of use; the official SROIE terms are on the competition site cited in [12]]")
    t3 = T[3]
    set_cell(t3.rows[2].cells[0], "OCR word box")
    set_cell(t3.rows[2].cells[2], "Location of each OCR word (line boxes split into proportional word boxes), scaled to 0-1000 for LayoutLMv3")
    set_cell(t3.rows[3].cells[0], "OCR word text")
    set_cell(t3.rows[3].cells[1], "String")
    set_cell(t3.rows[3].cells[2], "Text recognised by PaddleOCR for each word")
    set_cell(t3.rows[4].cells[0], "Merchant")
    set_cell(t3.rows[6].cells[0], "Total")
    set_cell(t3.rows[6].cells[2], "Final amount payable on the receipt")
    set_cell(t3.rows[7].cells[0], "Tax")
    set_cell(t3.rows[7].cells[1], "String / numeric amount")
    set_cell(t3.rows[7].cells[2], "GST/SST amount as printed on the receipt")
    set_cell(t3.rows[5].cells[0], "Date")
    set_body(find_par(doc, "Preprocessing requirements."),
             "Receipt images differ in size and quality. They were validated (readable, at least 300 pixels on each side), "
             "normalised to RGB copies and de-duplicated before OCR; further image enhancement was not applied (see Table 6).")
    set_body(find_par(doc, "Training/testing split."),
             f"Only human-reviewed receipts were used. They were split into training, validation and test sets "
             f"({int(cfg['split']['val_size']*100)}% validation, {int(cfg['split']['test_size']*100)}% test, seed {cfg['seed']}) "
             "with a merchant-grouped split, so that receipts from the same merchant never appear in two different sets; "
             "this prevents the model from scoring well by memorising merchant-specific layouts. The test set was used only for "
             "the final evaluation; model selection (early stopping) used the validation set.")
    # ---- 5. flow diagram
    set_runs(find_par(doc, "Figure 1 presents"), [
        "Figure 1 presents the workflow of the implemented Xtracts system: image cleaning and de-duplication, OCR, ground-truth "
        "creation, LayoutLMv3 fine-tuning, evaluation with experiment tracking, and the Streamlit/SQLite application."])
    set_runs(find_par(doc, "Fig. 1:"), ["Fig. 1: Overall workflow of the Xtracts system."])
    pic_par = [p for p in doc.paragraphs if p._p.xpath(".//pic:pic")][0]
    blip = pic_par._p.xpath(".//a:blip")[0]
    part = doc.part.related_parts[blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")]
    from PIL import Image
    w, h = Image.open(io.BytesIO(part.blob)).size
    part._blob = make_flow_png(w, h)
    flow_steps = {
        "1. Data Collection": "1. Data Collection: 2,586 receipt images were collected in the project's raw-data folder.",
        "2. Data Preprocessing": "2. Data Preprocessing: Images are validated, converted to normalised RGB copies and de-duplicated by SHA-256 and perceptual hashing (865 unique receipts).",
        "3. Exploratory Data Analysis": "3. Exploratory Data Analysis: Duplicate clusters were inspected on contact sheets; OCR statistics (words and confidence per receipt) were summarised.",
        "4. OCR (Text": "4. OCR (Text Detection and Recognition): PaddleOCR returns text lines with boxes and confidence; lines are split into words with proportional boxes.",
        "5. Text Normalisation": "5. Ground-truth creation: A rule-based labeler proposes merchant, date, total and tax words; a human reviews and corrects them in the review application.",
        "6. Information Extraction": "6. Information Extraction: LayoutLMv3 is fine-tuned to tag each OCR word as merchant, date, total, tax or other; long receipts are read in overlapping windows.",
        "7. Model Evaluation": "7. Model Evaluation and Selection: Entity-level precision, recall and F1 and value exact match are measured on the held-out test receipts; early stopping uses the validation set.",
        "8. Structured Output": "8. Structured Output: Predicted values are normalised (ISO date, numeric amounts), shown in editable fields and saved to SQLite after user confirmation.",
        "9. Experiment Tracking": "9. Experiment Tracking: Evaluation runs are logged with MLflow (parameters, metrics and result files).",
        "10. Expense Analytics": "10. Expense Analytics and Visualisation: The dashboard summarises total spend, transaction count, daily and monthly spending and spending by merchant from the database.",
        "11. Deployment and User": "11. Deployment and User Output: The pipeline runs as a local Streamlit application (no cloud deployment).",
    }
    for key, text in flow_steps.items():
        lead, _, rest = text.partition(": ")
        set_runs(find_par(doc, key), [lead + ": ", rest])

    # ---- 6. technology
    set_runs(find_par(doc, "This section lists the technologies"), [
        "This section lists the technologies and libraries actually used in the project."])
    set_runs(find_par(doc, "Methodology."), [
        "Methodology. ", "The methodology follows a modular pipeline: (a) validate and de-duplicate the images, (b) run OCR to "
        "obtain words with positions, (c) create ground truth by rule-based suggestion followed by human correction, "
        "(d) fine-tune LayoutLMv3 for token classification, (e) evaluate against the human-verified test receipts and "
        "(f) apply the model in an application that stores confirmed expenses and computes analytics. Extraction quality is "
        "measured with entity-level precision, recall and F1-score and with value exact match, following the evaluation idea "
        "of the SROIE competition [1]. OCR character/word error rates were not measured because no character-level "
        "transcriptions exist for these images."])
    set_runs(find_par(doc, "Programming language."), ["Programming language.", " ", "Python 3.10 (conda environment ‘xtracts’)."])
    set_body(find_par(doc, "Delete any row"), "")
    t4 = T[4]
    rows4 = [
        ("Programming language: Python 3.10", "Implementation of all pipeline stages."),
        ("Image-processing library: Pillow, ImageHash [14 not used]", "Image validation, RGB conversion, resizing, perceptual hashing (pHash, dHash) for duplicate detection. Functions used: Image.open, convert, resize; imagehash.phash/dhash."),
        ("OCR engine: PaddleOCR 2.9.1 (PaddlePaddle 2.6.2)", "Text detection and recognition on receipts. Used with the English model, angle classifier on and det_limit_side_len = 1600 so that small text on tall receipts is not shrunk away."),
        ("Data-handling: pandas [15], NumPy", "Manifests and reports, grouped splitting, expense aggregation for analytics (groupby, to_datetime, to_period)."),
        ("Text-processing: Python re module", "Rule-based labeler: regular expressions for dates, amounts, GST/total keywords, plus OCR-error repair; value normalisation in the application."),
        ("Machine learning: PyTorch, Hugging Face Transformers [11], scikit-learn [16]", "LayoutLMv3ForTokenClassification fine-tuning (custom training loop, AdamW, mixed precision), LayoutLMv3 processor, merchant-grouped splitting (GroupShuffleSplit)."),
        ("Visualisation: Streamlit charts", "Dashboard charts (monthly, daily, merchant-wise spending) and the annotation review interface."),
        ("Application / UI framework: Streamlit, SQLite (sqlite3)", "Upload, review/edit, save, dashboard and history; database of confirmed expenses with duplicate protection."),
        ("Version control: Git [17]", "Source-code management (see Section 8). Experiment tracking: MLflow [18]."),
    ]
    for i, (a, b) in enumerate(rows4, start=1):
        set_cell(t4.rows[i].cells[0], a)
        set_cell(t4.rows[i].cells[1], b)
    t5 = T[5]
    set_cell(t5.rows[1].cells[1], "Windows (PowerShell)")
    set_cell(t5.rows[2].cells[1], "Python 3.10.22")
    set_cell(t5.rows[3].cells[1], "[state your editor, e.g. VS Code]")
    set_cell(t5.rows[4].cells[1], "[state your laptop CPU and RAM]")
    set_cell(t5.rows[5].cells[1], "[state the GPU used for training, e.g. Colab T4]; OCR and the app run on CPU")
    set_cell(t5.rows[6].cells[1], "[Google Colab / Kaggle if used for training, otherwise none]")

    # ---- 7. model development
    set_runs(find_par(doc, "This section describes the model-development"), [
        "This section describes the model-development process. The pipeline has two modelling stages: OCR with the pre-trained "
        "PaddleOCR models (no OCR training took place) and field extraction with a fine-tuned LayoutLMv3 model, compared "
        "with a rule-based extractor. The Status column of Table 6 records what was and was not applied."])
    set_runs(find_par(doc, "Preprocessing is carried out"), ["Preprocessing was carried out on images, OCR output and labels; Table 6 lists each step with its status."])
    t6 = T[6]
    status = ["Applied: unreadable or under-300-pixel images flagged and excluded; receipts the reviewer judged unusable excluded from training",
              "Applied: SHA-256 for exact copies and perceptual hashes (pHash, dHash, Hamming distance ≤ 6) for near-duplicates; 1,721 of 2,586 images removed",
              "Not applied (PaddleOCR and LayoutLMv3 use RGB images)",
              "Not applied",
              "Not applied",
              "Not applied separately (PaddleOCR's angle classifier is enabled)",
              "Applied: OCR with a 1,600-pixel detection limit; the LayoutLMv3 processor resizes to 224 × 224; training images were shrunk to 768 px for the Colab bundle",
              "Applied in part: size and duplicate checks; extreme-ratio images were not analysed separately",
              "Applied: amounts and dates are normalised for evaluation and in the application; OCR slips such as ‘RMO.47’ are repaired in the rule-based labeler",
              "Applied: bounding boxes scaled to 0-1000 and word labels encoded as BIO tags; no feature scaling needed",
              "Applied: dates to YYYY-MM-DD (day-first), amounts to numbers with two decimals"]
    for i, s in enumerate(status, start=1):
        set_cell(t6.rows[i].cells[2], s)
    set_runs(find_par(doc, "Exploratory analysis is used"), [
        "Exploratory analysis concentrated on the data issues that affect the method. The collection contained many duplicates: "
        "of 2,586 images only 865 were unique, and the largest duplicate cluster contained 54 images; duplicate clusters were "
        "inspected on contact sheets before images were removed. The OCR output was summarised per receipt (number of lines/words, "
        "mean confidence). During annotation, recurring layout problems were recorded: dates glued to labels, times or invoice "
        "numbers; tax shown only in a ‘GST summary’ table; totals printed together with a tax column; and cash tendered or change "
        "printed near the total. Because receipts from the same merchant can look alike, the data were split by merchant. "
        "[Insert one or two EDA figures here, e.g. words per receipt, if you produce them.]"])
    set_runs(find_par(doc, "For a rule-based extractor,"), [
        "The rule-based extractor uses regular expressions for dates (including dates glued to other text), monetary amounts "
        "(with two decimals, tolerant to OCR slips such as a letter O for zero), keyword rows for totals (total, amount due, "
        "grand total, with subtotal, change and cash rows excluded), tax rows and GST-summary tables, and position cues "
        "(merchant names near the top). The LayoutLMv3 model does not use hand-made features: its input is each word's text, its "
        "bounding box normalised to 0-1000 and the receipt image, from which it learns the visual and positional cues itself."])
    set_runs(find_par(doc, "The models, engines and methods"), [
        "The models, engines and methods used at each stage and their important parameters are listed in Table 7. The rule-based "
        "extractor serves as the baseline against which the improvement from LayoutLMv3 is measured."])
    t8 = T[8]
    set_cell(t8.rows[1].cells[0], "PaddleOCR 2.9.1 (pre-trained)")
    set_cell(t8.rows[1].cells[2], f"language {ocr['lang']}, angle classifier on, det_limit_side_len {ocr['det_limit_side_len']}, min confidence {ocr['min_conf']}")
    set_cell(t8.rows[2].cells[0], "Baseline: rule-based extractor")
    set_cell(t8.rows[2].cells[1], "Propose merchant, date, total and tax from recognised words")
    set_cell(t8.rows[2].cells[2], "Regex patterns, keyword lists, exclusion lists, row grouping by vertical position")
    set_cell(t8.rows[3].cells[0], "LayoutLMv3 (microsoft/layoutlmv3-base, fine-tuned)")
    set_cell(t8.rows[3].cells[1], "Tag every OCR word as merchant, date, total, tax or other (BIO tags, 9 classes)")
    set_cell(t8.rows[3].cells[2], f"max length {hp['max_length']} tokens with {hp['overlap_words']}-word window overlap, lr {hp['learning_rate']}, "
             f"batch {hp['batch_size']} × grad. accumulation {hp['grad_accum_steps']}, ≤ {hp['epochs']} epochs, weight decay {hp['weight_decay']}, "
             f"warm-up {hp['warmup_ratio']}, early-stopping patience {hp['early_stopping_patience']}")
    train_txt = (f"The best epoch was {g['summary']['best_epoch']} (validation micro-F1 {g['summary']['best_val_f1']:.3f}); training took "
                 f"{g['summary']['train_seconds']} seconds.") if g["summary"] else todo_note("training summary: run 05_train_layoutlmv3.py and copy models/layoutlmv3-receipts/summary.json here")
    set_runs(find_par(doc, "If a learned extractor was trained"), [
        "The reviewed receipts were divided into training, validation and test sets with a merchant-grouped split (Section 4). "
        f"LayoutLMv3-base was fine-tuned with a custom training loop: AdamW, linear warm-up and decay, gradient accumulation, "
        "mixed precision on GPU and gradient clipping. Receipts longer than 512 tokens were split into overlapping windows so that the "
        "bottom of the receipt, where the total and tax are, is never truncated; during evaluation, predictions of overlapping windows "
        "were merged per word. Because most words belong to no field, evaluation uses F1 on entities rather than plain accuracy. "
        "Training stopped early when the validation micro-F1 did not improve for several epochs, and the best checkpoint was kept. "
        "No hyper-parameter search was run; the settings of Table 7 were fixed in advance. " + train_txt])
    set_runs(find_par(doc, "The two stages are evaluated"), [
        "OCR was used with pre-trained weights and is not evaluated separately (no character-level ground truth exists), so extraction errors "
        "caused by OCR cannot be separated from model errors; the evaluation measures the whole OCR-plus-extraction chain on the OCR words."])
    set_runs(find_par(doc, "OCR quality."), ["OCR quality. ", "Character and word error rates (CER, WER) require reference transcriptions and were not measured. "
                                                "Only the engine's own mean confidence is reported in Section 12."])
    set_runs(find_par(doc, "Field-level extraction quality."), [
        "Field-level extraction quality. ", "Entity-level scores count a prediction as correct only if its field and its exact word span "
        "equal the human label (the CoNLL/seqeval convention, implemented in the project). Value exact match compares the extracted value "
        "after normalisation (merchant case- and punctuation-insensitive, date in ISO form, amount to two decimals). Precision, recall and "
        "F1-score are computed from true positives (TP), false positives (FP) and false negatives (FN):"])
    set_runs(find_par(doc, "Accuracy, the confusion matrix"), [
        "A token-level confusion matrix and ROC-AUC were not computed; entity-level precision, recall, F1-score and value exact match "
        "are reported instead. Regression metrics are not appropriate because the total is extracted, not estimated."])
    t9 = T[9]
    for i, s in enumerate(["No (no transcriptions)", "No (no transcriptions)", "Yes", "Yes (value exact match)", "No"], start=1):
        set_cell(t9.rows[i].cells[2], s)
    set_cell(t9.rows[3].cells[1], "Extraction stage (merchant, date, total, tax)")

    # ---- 8-11 versions / tracking / MLOps / deployment
    set_runs(find_par(doc, "Version control records"), [
        "Version control records the history of a project's files so that changes can be reviewed, compared and reverted [17]. "
        "The project was kept in a Git repository with a README, a .gitignore that excludes data, models and environments, and one "
        "commit per completed step (dataset preparation, OCR, annotation tools, training, evaluation, application, report)."])
    set_cell(T[10].rows[0].cells[0], "Implementation status: Implemented (Git; one commit per project step).")
    t11 = T[11]
    set_cell(t11.rows[1].cells[1], "Git")
    set_cell(t11.rows[2].cells[1], "[GitHub / local only - state which]")
    set_cell(t11.rows[3].cells[1], "[repository name / URL]")
    set_cell(t11.rows[4].cells[1], "[main]")
    set_cell(t11.rows[5].cells[1], "[number]")
    set_cell(t11.rows[6].cells[1], "Dataset cleaning -> OCR -> labeling/review app -> training code -> evaluation -> application")
    set_runs(find_par(doc, "Experiment tracking means"), [
        "Experiment tracking means recording, for every run, the data version, parameters, metrics and outputs, so that results can be "
        "reproduced and compared [18]."])
    set_cell(T[13].rows[0].cells[0], "Implementation status: Implemented for evaluation runs with MLflow (SQLite backend); training hyper-parameters are recorded in "
             "summary.json and training_log.csv. No hyper-parameter search was run.")
    set_runs(find_par(doc, "Planned approach"), [
        "Implemented approach. ", " Each evaluation run of scripts/07_evaluate.py logs the split, number of receipts, model and "
        "hyper-parameters as MLflow parameters and the per-field precision, recall, F1 and value exact match of each system as metrics, "
        "and stores metrics.json, field_metrics.csv, per_receipt.csv and summary.md as artifacts. Table 10 lists the runs."])
    t14 = T[14]
    set_cell(t14.rows[0].cells[2], "Mean value exact match (4 fields)")
    set_cell(t14.rows[0].cells[3], "Micro F1")
    names = list(st)
    for i in range(1, 4):
        if i <= len(names):
            n = names[i - 1]
            set_cell(t14.rows[i].cells[0], f"Exp-0{i}: {n}")
            set_cell(t14.rows[i].cells[1], "regex/keyword rules" if n == "Rules" else f"LayoutLMv3-base fine-tuned, lr {hp['learning_rate']}, ≤{hp['epochs']} epochs")
            set_cell(t14.rows[i].cells[2], pct(st[n]["mean_exact"]))
            set_cell(t14.rows[i].cells[3], pct(st[n]["micro"]["f1"]))
            set_cell(t14.rows[i].cells[4], f"{g['metrics']['meta']['split']} split, {g['metrics']['meta']['n_receipts']} receipts")
        elif i == 1:
            for j in range(5):
                set_cell(t14.rows[i].cells[j], todo_note("run scripts/07_evaluate.py") if j == 1 else "-")
    if len(names) < 3:
        for i in range(len(t14.rows) - 1, max(len(names), 1), -1):
            delete_row(t14, i)
    set_cell(T[15].rows[0].cells[0], "[Insert a screenshot of the MLflow UI (mlflow ui --backend-store-uri sqlite:///mlflow.db) showing the evaluation run]")
    set_cell(T[16].rows[0].cells[0], "Applicability: AutoML was not used in this project. MLOps is limited to code versioning, MLflow evaluation tracking, automated tests and a reproducible environment.")
    set_runs(find_par(doc, "AutoML automates"), [
        "AutoML automates parts of model building such as algorithm selection and hyperparameter optimisation [19]. AutoML was not used "
        "in this project: the OCR engine is pre-trained, and one LayoutLMv3 configuration was fine-tuned with fixed hyper-parameters. A "
        "hyper-parameter search (for example with Optuna) is left as future work."])
    for j in range(1, 5):
        set_cell(T[17].rows[1].cells[j], "N.A.")
    set_cell(T[17].rows[1].cells[0], "AutoML not used")
    if len(T[17].rows) > 2:
        delete_row(T[17], 2)
    set_runs(find_par(doc, "MLOps applies"), [
        "MLOps applies software-engineering discipline to the machine-learning lifecycle [20]. Implemented in this project: Git "
        "versioning, a pinned requirements file, configuration in a single YAML file, automated unit and integration tests "
        "(pytest) for the data pipeline, labeler, training loop, evaluation, database and application pages, and MLflow tracking of "
        "evaluation runs. Not implemented: a model registry, CI/CD, containerisation and monitoring after deployment."])
    set_runs(find_par(doc, "Deployment makes"), [
        "Deployment makes the pipeline usable by people other than the developers. Xtracts is run locally as a Streamlit application "
        "(streamlit run src/xtracts/app/streamlit_app.py). The user uploads a receipt image (invalid, unreadable or too-small files "
        "are rejected with a message); the application runs PaddleOCR and the fine-tuned LayoutLMv3 model, shows the extracted merchant, "
        "date, total and tax in editable fields, and saves the confirmed values to a SQLite database. Saving is refused for duplicate "
        "images or for an existing merchant-date-total combination. The dashboard shows total spending, transaction count, average "
        "receipt value, tax, monthly and daily spending and spending by merchant, computed from the database; the history page "
        "filters by date and merchant and supports editing, deleting and CSV download. If no trained model is present the application "
        "says so and falls back to the rule-based extractor, clearly labelled. No cloud deployment or container was used."])
    set_cell(T[18].rows[0].cells[0], "Implementation status: Implemented as a local Streamlit application with a SQLite database; not deployed to a server or cloud.")
    t19 = T[19]
    for i, v in enumerate(["Streamlit", "None (no separate API)", "Not used", "None", "Local execution (streamlit run)"], start=1):
        set_cell(t19.rows[i].cells[1], v)
    set_cell(T[20].rows[0].cells[0], "[Insert screenshots: Upload & extract page with editable fields; Dashboard; History]")

    # ---- 12 results
    if not st:
        set_runs(find_par(doc, "This section presents the results"), [
            todo_note("run scripts/07_evaluate.py, then run scripts/08_make_report.py again - no evaluation results exist yet, so none are shown")])
    else:
        m = g["metrics"]["meta"]
        set_runs(find_par(doc, "This section presents the results"), [
            f"This section reports the results measured by scripts/07_evaluate.py on the {m['split']} split: {m['n_receipts']} receipts whose labels "
            "were corrected by a human and that were not used for training or model selection. The scores are therefore independent of the "
            "‘auto-label accuracy’ shown in the review application, which measures the rules, not the model. All numbers come from one "
            f"evaluation run ({m['generated']}); the files are in reports/evaluation/."])
    t21 = T[21]
    conf = f"mean OCR confidence {g['ocr_conf'][0]:.3f} over {g['ocr_conf'][1]} receipts" if g["ocr_conf"] else todo_note("mean OCR confidence from ocr_summary.csv")
    set_cell(t21.rows[1].cells[0], "PaddleOCR 2.9.1, English, no extra image preprocessing")
    set_cell(t21.rows[1].cells[1], "Not measured")
    set_cell(t21.rows[1].cells[2], "Not measured")
    set_cell(t21.rows[1].cells[3], "No reference transcriptions exist; " + conf)
    delete_row(t21, 2)
    t22 = T[22]
    set_cell(t22.rows[0].cells[1], "Value exact match (mean of 4 fields)")
    set_cell(t22.rows[0].cells[2], "Precision (micro)")
    set_cell(t22.rows[0].cells[3], "Recall (micro)")
    set_cell(t22.rows[0].cells[4], "F1 Score (micro)")
    set_cell(t22.rows[0].cells[5], "AUC")
    for i in range(1, 4):
        if i <= len(names):
            n = names[i - 1]
            s = st[n]
            for j, v in enumerate([n, pct(s["mean_exact"]), pct(s["micro"]["precision"]), pct(s["micro"]["recall"]), pct(s["micro"]["f1"]), "N.A."]):
                set_cell(t22.rows[i].cells[j], v)
        else:
            for j in range(6):
                set_cell(t22.rows[i].cells[j], todo_note("run 07_evaluate.py") if (i == 1 and j == 0) else "-")
    for i in range(len(t22.rows) - 1, max(len(names), 1), -1):
        delete_row(t22, i)
    t23 = T[23]
    set_cell(t23.rows[0].cells[0], f"Field ({best or 'best method'})")
    order = ["MERCHANT", "DATE", "TOTAL", "TAX"]
    for i, f in enumerate(order, start=1):
        set_cell(t23.rows[i].cells[0], NICE[f])
        if best:
            for j, k in enumerate(["p", "r", "f1"], start=1):
                set_cell(t23.rows[i].cells[j], pct(st[best][f][k]))
        else:
            for j in range(1, 4):
                set_cell(t23.rows[i].cells[j], "[value]")
    set_cell(T[24].rows[0].cells[0], "[Insert figure: per-field comparison of the two systems from reports/evaluation/field_metrics.csv, and sample extractions from the app]")

    # discussion
    per = g["per"]
    def n_wrong(system, field):
        sel = [r for r in per if r["system"] == system and r["field"] == field] if per else []
        return sum(r["match"] != "True" for r in sel), len(sel)
    if best:
        other = [k for k in names if k != best]
        cmp = (f" The other system, {other[0]}, reached a micro F1 of {pct(st[other[0]]['micro']['f1'])} and a mean value exact match of "
               f"{pct(st[other[0]]['mean_exact'])}." if other else "")
        best_txt = (f"On the held-out test receipts, {best} obtained the highest micro F1 ({pct(st[best]['micro']['f1'])}; precision "
                    f"{pct(st[best]['micro']['precision'])}, recall {pct(st[best]['micro']['recall'])}) and a mean value exact match of "
                    f"{pct(st[best]['mean_exact'])}.{cmp} Per field (F1): " +
                    ", ".join(f"{NICE[f]} {pct(st[best][f]['f1'])}" for f in order) +
                    ". Because the human labels began as rule suggestions, annotators may be anchored to them, which favours the rule baseline; "
                    "the comparison therefore does not exaggerate the benefit of the model. The test set is small, so differences of a few points "
                    "are within the noise.")
        errs = ", ".join(f"{NICE[f]}: {n_wrong(best, f)[0]} of {n_wrong(best, f)[1]}" for f in order) if per else ""
        err_txt = ("OCR errors and extraction errors could not be separated, because the labels are defined on the OCR words and no "
                   "character-level ground truth exists. Receipts whose value did not match the human label, for the best system "
                   f"({best}): {errs}. Inspect reports/evaluation/per_receipt.csv for the individual cases.")
    else:
        best_txt = todo_note("write after running scripts/07_evaluate.py")
        err_txt = todo_note("write after running scripts/07_evaluate.py")
    set_body(find_par(doc, "Discussion. The discussion"), "The results below come from the run described above.")
    set_body(find_par(doc, "Best-performing method"), best_txt)
    set_body(find_par(doc, "Stage-wise error analysis"), err_txt)
    set_body(find_par(doc, "Expected error sources"),
             "While correcting labels, the following failure causes were observed and are the likely causes of remaining errors; they have not been "
             "individually quantified: dates glued to labels, times or invoice numbers; tax printed only in a ‘GST summary’ table or on the same "
             "row as the total; garbled keywords from OCR (for example ‘Jotal’, ‘Tota1’); cash tendered, change and rounding rows near the "
             "total; multi-line merchant names and header lines such as ‘PLEASE VISIT US AGAIN’; and faded thermal print.")
    set_body(find_par(doc, "Important features or cues"),
             "No feature-importance analysis was carried out. For the rules, the contributions of the date, total and tax rules were improved "
             "iteratively by comparing them with the human corrections (for example, handling of GST summary tables increased the share of "
             "receipts with a tax suggestion from 38% to 83% of the unreviewed receipts).")
    set_body(find_par(doc, "Comparison with existing approaches"),
             "No comparison with published SROIE results is made. The data (our own OCR words and four fields including tax), the labels and the "
             "split differ from the competition protocol, so the scores are not comparable.")
    set_body(find_par(doc, "Limitations."),
             "Only four fields are extracted; line items and categories are not. Ground truth was produced by a single reviewer correcting rule "
             "suggestions, with the first reviewed receipts selected as the most flagged ones. The data are Malaysian receipts, so other layouts "
             "and languages are untested. The number of verified receipts is a few hundred, so the confidence intervals are wide. OCR quality "
             "was not measured separately, and word boxes inside a line are estimated proportionally.")
    set_body(find_par(doc, "Suitability for the intended application"),
             "In the application every extracted value is shown in an editable field before it is saved, so occasional errors can be corrected "
             "by the user; this makes the accuracy reported above adequate for personal expense tracking if the reviewed values are checked. It "
             "is not sufficient for auditing or accounting use without a verification step.")

    # ---- 13 future scope, 14 conclusion
    set_runs(find_par(doc, "All items below"), ["The items below are proposed improvements; they are not part of the implemented system."])
    set_body(find_par(doc, "Stronger extraction models."),
             "LayoutLMv3 is already used. Further work could compare larger or OCR-free models [8], add more verified training receipts and "
             "run a hyper-parameter search (for example with Optuna).")
    set_body(find_par(doc, "Explainability and confidence."),
             "The model produces word-level probabilities, but they are not yet shown. Displaying confidence and highlighting the source of each "
             "field on the image would help users verify values faster.")
    set_body(find_par(doc, "Deployment options."),
             "The application runs locally. A cloud-hosted or containerised version, a mobile capture app and export to accounting software remain future work.")
    set_body(find_par(doc, "MLOps maturity."),
             "A model registry, CI/CD and monitoring of extraction quality in use; MLflow tracking and tests are already in place (Sections 9 and 10).")
    if best:
        concl = (f"The final system turns receipt images into structured records and expense analytics. On {g['metrics']['meta']['n_receipts']} held-out, "
                 f"human-verified test receipts, {best} reached a micro F1 of {pct(st[best]['micro']['f1'])} and a mean value exact match of "
                 f"{pct(st[best]['mean_exact'])} over merchant, date, total and tax. The main limitations are the small verified data set, the single "
                 "annotator, the lack of separate OCR evaluation and the restriction to four fields.")
    else:
        concl = todo_note("conclusion with measured scores - run scripts/07_evaluate.py and scripts/08_make_report.py again")
    set_runs(find_par(doc, "This project addressed"), [
        "This project addressed the problem of turning unstructured receipt images into structured data and simple expense analytics. Xtracts "
        "combines image cleaning and de-duplication, PaddleOCR, human-verified ground truth, a fine-tuned LayoutLMv3 model and a Streamlit "
        "application with SQLite storage. Extraction quality was measured with entity-level precision, recall and F1-score and with value exact match."])
    set_runs(find_par(doc, "[Summarise your actual outcome"), [concl + " Overall, the work demonstrates the practical value of automated receipt understanding for expense management, while the identified limitations and the future-scope items in Section 13 outline directions for improvement."])

    # ---- leftovers: literature-gap paragraph, objective, captions
    set_body(find_par(doc, "How Xtracts addresses the gap."),
             "Xtracts keeps the pipeline modular and measurable. It trains and evaluates a layout-aware transformer on human-verified receipts, "
             "compares it with a rule-based baseline, and feeds the structured output into an application with expense analytics, so that the "
             "practical value of the extracted data is demonstrated and not only its benchmark score. Because OCR is used with pre-trained "
             "weights and no transcriptions exist, OCR errors are not evaluated separately from extraction errors. It does not claim to surpass "
             "the results cited above.")
    set_runs(find_par(doc, "To evaluate OCR quality"), ["To evaluate field-level extraction accuracy on human-verified test receipts using standard, clearly defined metrics."])
    for start, new in (("Fig. 2:", "Fig. 2: Exploratory data analysis of the receipt collection."),
                       ("Fig. 3:", "Fig. 3: Project repository."),
                       ("Table 10:", "Table 10: Experiment log (evaluation runs)"),
                       ("Table 11:", "Table 11: AutoML model ranking (AutoML not used)"),
                       ("Table 13:", "Table 13: OCR quality (not measured; no reference transcriptions)"),
                       ("Table 14:", "Table 14: Comparison of methods on the human-verified test receipts"),
                       ("Table 15:", "Table 15: Field-level extraction results for the best method"),
                       ("Fig. 4:", "Fig. 4: Screenshots and sample outputs of the Xtracts application.")):
        set_runs(find_par(doc, start), [new])

    # ---- references: add the two libraries actually used
    last = find_par(doc, "[21]")
    for num, text in ((22, "Y. Du et al., “PP-OCR: A practical ultra lightweight OCR system,” arXiv:2009.09941, 2020."),
                      (23, "T. Wolf et al., “Transformers: State-of-the-art natural language processing,” in Proc. 2020 Conf. Empirical Methods in "
                           "Natural Language Processing: System Demonstrations, 2020, pp. 38–45.")):
        new = copy.deepcopy(last._p)
        last._p.addnext(new)
        from docx.text.paragraph import Paragraph
        np_ = Paragraph(new, last._parent)
        set_runs(np_, [f"[{num}]\t", text])
        last = np_
    set_cell(t4.rows[3].cells[0], "OCR engine: PaddleOCR 2.9.1 (PaddlePaddle 2.6.2) [22]")
    set_cell(t4.rows[6].cells[0], "Machine learning: PyTorch, Hugging Face Transformers [23], scikit-learn [16]")
    set_cell(t4.rows[2].cells[0], "Image processing: Pillow, ImageHash")

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    left = []
    for p in docx.Document(str(out)).paragraphs:
        left += re.findall(r"\[[^\]]{4,}\]", p.text) if not re.match(r"^\[\d+\]", p.text) else []
    for t in docx.Document(str(out)).tables:
        for r in t.rows:
            for c in r.cells:
                left += re.findall(r"\[[^\]]{4,}\]", c.text)
    return sorted(set(left))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default="docs/Xtracts_Mini_Project_Report_TEMPLATE.docx")
    ap.add_argument("--out", default="docs/Xtracts_Mini_Project_Report_UPDATED.docx")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g = gather(cfg)
    left = build(resolve(a.template), resolve(a.out), g, cfg)
    print(f"Wrote {resolve(a.out)}")
    print(f"  evaluation results used: {'yes' if g['metrics'] else 'NO (run scripts/07_evaluate.py first)'}; "
          f"splits found: {sorted(g['splits']) or 'none'}; reviewed receipts: {g['reviewed']}")
    if left:
        print("\nStill to fill by hand (visible in the report as [brackets]):")
        for x in left:
            print("  -", x[:150])


if __name__ == "__main__":
    main()
