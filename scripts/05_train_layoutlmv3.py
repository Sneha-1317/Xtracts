"""Step 5: fine-tune LayoutLMv3 on the train split, validate every epoch, keep the best model.

Output (models/layoutlmv3-receipts/):
    best/              model + tokenizer + processor + label_map.json
    training_log.csv   loss and validation F1 per epoch
    val_metrics.json   per-field precision / recall / F1 of the best epoch
    summary.json       best epoch, hyperparameters, training time

Usage (from the project root):
    python scripts/05_train_layoutlmv3.py --limit 20 --epochs 1   # quick smoke test
    python scripts/05_train_layoutlmv3.py                         # full run (use a GPU: Colab / Kaggle)
    python scripts/05_train_layoutlmv3.py --eval-test             # also score the untouched test split ONCE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xtracts.config import load_config, resolve  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N train receipts (smoke test)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--num-workers", type=int, default=0, help="keep 0 on Windows")
    ap.add_argument("--eval-test", action="store_true", help="score the best model on test.jsonl (do this once, at the end)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    P = {k: resolve(v) for k, v in cfg["paths"].items()}
    hp = dict(cfg["training"])
    if a.epochs:
        hp["epochs"] = a.epochs
    if a.batch_size:
        hp["batch_size"] = a.batch_size
    if a.lr:
        hp["learning_rate"] = a.lr

    for name in ("train", "val"):
        if not (P["splits_dir"] / f"{name}.jsonl").exists():
            sys.exit(f"{P['splits_dir'] / (name + '.jsonl')} not found. Run: python scripts/04_build_dataset.py")
    train_rows = read_jsonl(P["splits_dir"] / "train.jsonl")
    val_rows = read_jsonl(P["splits_dir"] / "val.jsonl")
    if a.limit:
        train_rows, val_rows = train_rows[: a.limit], val_rows[: max(3, a.limit // 4)]

    from xtracts.model.metrics import format_report
    from xtracts.model.train import evaluate_rows, fit, load_pretrained

    model, processor = load_pretrained(hp["model_name"])
    fit(model, processor, train_rows, val_rows, P["clean_images_dir"], P["model_dir"], hp,
        device=a.device, num_workers=a.num_workers, seed=cfg["seed"])

    if a.eval_test:
        import torch
        from transformers import LayoutLMv3ForTokenClassification

        test_path = P["splits_dir"] / "test.jsonl"
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        best = LayoutLMv3ForTokenClassification.from_pretrained(P["model_dir"] / "best").to(device)
        res = evaluate_rows(best, processor, read_jsonl(test_path), P["clean_images_dir"], device,
                            hp["batch_size"], hp["max_length"], hp["overlap_words"], a.num_workers)
        (P["model_dir"] / "test_metrics.json").write_text(json.dumps(res, indent=2))
        print("\nTEST results:\n" + format_report(res))
    print(f"\nModel saved in {P['model_dir'] / 'best'}")


if __name__ == "__main__":
    main()
