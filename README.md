# Xtracts: OCR-based receipt extraction and expense analytics (LayoutLMv3)

Upload a receipt image -> PaddleOCR -> fine-tuned LayoutLMv3 -> editable merchant / date / total / tax ->
SQLite -> expense dashboard and transaction history.

## Setup (Windows PowerShell, conda env `xtracts`, Python 3.10)
```powershell
conda activate xtracts
pip install -r requirements.txt
pip install -e .
```
Training needs a GPU: do it on Colab/Kaggle (`python scripts/06_make_colab_bundle.py`, then `requirements-train.txt`).
The app and evaluation only need the trained model folder `models/layoutlmv3-receipts/best/` on this machine.

## Pipeline (run in this order; every script is resumable)
| Step | Command | Result |
|---|---|---|
| 1 Dataset cleaning + de-duplication | `python scripts/01_prepare_dataset.py --export` | 2586 files -> 865 unique receipts |
| 2 OCR (PaddleOCR) | `python scripts/02_run_ocr.py` | `data/annotations/ocr/*.json` (words, boxes, confidence) |
| 3 Auto-labels + human review | `python scripts/03_auto_label.py` then `streamlit run src/xtracts/annotation/review_app.py` | `data/annotations/labels/*.json` |
| 4 Build train/val/test | `python scripts/04_build_dataset.py` | `data/processed/splits/*.jsonl` (merchant-grouped, reviewed receipts only) |
| 5 Fine-tune LayoutLMv3 | `python scripts/05_train_layoutlmv3.py --eval-test` (on a GPU) | `models/layoutlmv3-receipts/best` |
| 6 Evaluate | `python scripts/07_evaluate.py` | `reports/evaluation/{metrics.json,field_metrics.csv,per_receipt.csv,summary.md}` + MLflow run |
| 7 App | `streamlit run src/xtracts/app/streamlit_app.py` | upload, edit, save, dashboard, history |

## Run the application
```powershell
streamlit run src/xtracts/app/streamlit_app.py
```
- **Upload & extract**: choose a JPG/PNG, press *Extract fields*, correct the values, press *Save expense*.
  Invalid/unreadable/small images, missing fields, extraction errors and duplicate saves (same image, or same
  merchant + date + total) are reported without crashing.
- **Dashboard**: total spent, number of receipts, average, tax, monthly and daily spending, spending by merchant,
  with a date-range filter. Everything is computed from the saved SQLite records.
- **History**: filter by date/merchant, edit cells, delete rows, download CSV.
- The database is `data/app/expenses.db` (created automatically); uploaded images are copied to `data/app/uploads/`.
- If `models/layoutlmv3-receipts/best` does not exist the app says so and uses the **rule-based fallback**
  (clearly labelled on screen). That fallback is not LayoutLMv3.

## Evaluation (Stage 6)
```powershell
python scripts/07_evaluate.py                # test split: LayoutLMv3 vs the rule-based labeler, same receipts
python scripts/07_evaluate.py --rules-only   # before a model exists
mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse the logged runs
```
Metrics: entity-level precision / recall / F1 per field, and value exact match (normalised merchant, ISO date,
amount to 2 decimals). Read the limitations in `reports/evaluation/summary.md`. The "auto-label accuracy" shown in the
review app measures the rules against your corrections, **not** the model.

## Tests
```powershell
pytest
```
(Torch-based tests are skipped automatically if torch is not installed.)

## Known limitations / future scope
- Only merchant, date, total and tax are extracted (no line items or categories).
- Training labels are human-corrected rule suggestions on SROIE-style Malaysian receipts; the number of verified
  receipts is a few hundred, so results have wide error bars.
- Hyper-parameter search (Optuna) and cloud deployment were not done.
