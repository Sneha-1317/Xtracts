"""Stage 6: evaluate on receipts whose labels were verified by a human, and save the results.

Compares the fine-tuned LayoutLMv3 with the rule-based labeler on the SAME receipts.

Output (reports/evaluation/):
    metrics.json       all numbers
    field_metrics.csv  precision / recall / F1 / exact match per field and system
    per_receipt.csv    gold vs predicted value for every receipt and field (for error analysis)
    summary.md         readable summary with sample size and limitations
MLflow: logged automatically if `mlflow` is installed (runs go to ./mlflow.db, view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).

Usage (from the project root):
    python scripts/07_evaluate.py                 # test split, needs models/layoutlmv3-receipts/best
    python scripts/07_evaluate.py --split val
    python scripts/07_evaluate.py --rules-only    # no model yet: only the rule baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xtracts.annotation.labels import entities_to_bio  # noqa: E402
from xtracts.annotation.weak_labels import auto_label  # noqa: E402
from xtracts.config import load_config, resolve  # noqa: E402
from xtracts.model.evaluation import (  # noqa: E402
    flatten, markdown_summary, value_accuracy, value_rows, write_csv)
from xtracts.model.metrics import format_report, score  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rules_predictions(rows: list[dict], ocr_dir: Path) -> list[list[str]]:
    out = []
    for r in rows:
        rec = json.loads((ocr_dir / f"{r['image_id']}.json").read_text(encoding="utf-8"))
        ents, _ = auto_label(rec)                        # CURRENT rules, run fresh on the OCR words
        out.append(entities_to_bio(ents))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    P = {k: resolve(v) for k, v in cfg["paths"].items()}
    hp = cfg["training"]
    path = P["splits_dir"] / f"{a.split}.jsonl"
    if not path.exists():
        sys.exit(f"{path} not found. Run: python scripts/04_build_dataset.py")
    rows = read_jsonl(path)
    gold = [r["labels"] for r in rows]
    print(f"Evaluating on the {a.split} split: {len(rows)} human-verified receipts")

    systems: dict[str, tuple[dict, dict]] = {}
    all_vrows: dict[str, list[dict]] = {}

    rule_tags = rules_predictions(rows, P["ocr_dir"])
    rs = score(gold, rule_tags)
    rv = value_rows(rows, rule_tags)
    systems["Rules"], all_vrows["Rules"] = (rs, value_accuracy(rv)), rv

    model_name = "none (rules only)"
    if not a.rules_only:
        best = P["model_dir"] / "best"
        if not (best / "config.json").exists():
            sys.exit(f"No trained model in {best}. Train first (scripts/05_train_layoutlmv3.py) or use --rules-only.")
        import torch
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

        from xtracts.model.train import evaluate_rows

        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        processor = LayoutLMv3Processor.from_pretrained(best, apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained(best).to(device)
        res = evaluate_rows(model, processor, rows, P["clean_images_dir"], device, hp["batch_size"],
                            hp["max_length"], hp["overlap_words"], return_predictions=True)
        mv = value_rows(rows, res["pred_tags"])
        systems["LayoutLMv3"], all_vrows["LayoutLMv3"] = (res, value_accuracy(mv)), mv
        model_name = f"{hp['model_name']} fine-tuned ({best.name})"

    table = [r for name, (s, v) in systems.items() for r in flatten(name, s, v)]
    out = P["eval_dir"]
    out.mkdir(parents=True, exist_ok=True)
    meta = {"split": a.split, "n_receipts": len(rows), "model": model_name,
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M")}
    (out / "metrics.json").write_text(json.dumps({
        "meta": meta,
        "systems": {n: {"entity_level": {k: v for k, v in s.items() if k != "pred_tags"}, "value_level": v}
                    for n, (s, v) in systems.items()}}, indent=2))
    write_csv(out / "field_metrics.csv", table)
    write_csv(out / "per_receipt.csv", [{"system": n, **v} for n, vs in all_vrows.items() for v in vs])
    (out / "summary.md").write_text(markdown_summary(meta, table), encoding="utf-8")

    for name, (s, _) in systems.items():
        print(f"\n{name}\n{format_report(s)}")
    print(f"\nSaved to {out}")

    if not a.no_mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri("sqlite:///" + resolve("mlflow.db").as_posix())
            mlflow.set_experiment("xtracts-receipt-extraction")
            with mlflow.start_run(run_name=f"eval-{a.split}"):
                mlflow.log_params({"split": a.split, "n_receipts": len(rows), "model": model_name,
                                   **{f"hp_{k}": v for k, v in hp.items()}})
                for r in table:
                    key = r["field"].split(" ")[0].lower()
                    for m in ("precision", "recall", "f1", "value_exact_match"):
                        if isinstance(r[m], float):
                            mlflow.log_metric(f"{r['system']}_{key}_{m}", r[m])
                mlflow.log_artifacts(str(out))
            print("Logged to MLflow. View: mlflow ui --backend-store-uri sqlite:///mlflow.db")
        except ImportError:
            print("MLflow not installed: skipped (pip install mlflow to track runs)")
        except Exception as e:                       # tracking must never break the evaluation
            print(f"MLflow logging failed ({type(e).__name__}: {str(e)[:150]}); results are still saved.")


if __name__ == "__main__":
    main()
