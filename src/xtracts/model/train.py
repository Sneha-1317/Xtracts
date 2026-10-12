"""Fine-tune LayoutLMv3 for receipt token classification. Needs torch + transformers.

fit()            train with early stopping on validation entity-F1, save the best model
evaluate_rows()  score a model on rows (val or test): per-field precision / recall / F1 / exact match

Long receipts are split into windows (see chunking.py). For evaluation the predictions of
overlapping windows are merged back to ONE tag per word (the window where the word is farthest
from the window edge wins), so scores are per receipt, not per window.
"""
from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..annotation.labels import ID2LABEL, LABEL2ID, LABEL_LIST
from .dataset import ReceiptWindows
from .metrics import format_report, score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pretrained(model_name: str):
    """Download (first run) and return (model, processor). OCR is done by us, so apply_ocr=False."""
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

    processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID)
    return model, processor


@torch.no_grad()
def evaluate_rows(model, processor, rows: list[dict], images_dir: Path, device, batch_size: int = 4,
                  max_length: int = 512, overlap_words: int = 30, num_workers: int = 0,
                  return_predictions: bool = False) -> dict:
    """Score the model on receipts. Returns {"loss", "fields", "micro", "n_receipts"} (see metrics.score).
    With return_predictions=True the result also has "pred_tags": one BIO tag list per receipt."""
    ds = ReceiptWindows(rows, images_dir, processor, LABEL2ID, max_length, overlap_words)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()

    pred = [[0] * len(r["words"]) for r in rows]          # label id per word
    margin = [[-1] * len(r["words"]) for r in rows]       # distance of the word from its window's edge
    total_loss, n_windows, k = 0.0, 0, 0
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = {key: v.to(device) for key, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item() * batch["input_ids"].shape[0]
        guess = out.logits.argmax(-1).cpu()
        labels = batch["labels"].cpu()
        for b in range(guess.shape[0]):
            ri, s, e = ds.items[k + b]
            tags = guess[b][labels[b] != -100].tolist()   # one prediction per word (first sub-token)
            for j, t in enumerate(tags[: e - s]):
                m = min(j, (e - s) - 1 - j)
                if m > margin[ri][s + j]:
                    margin[ri][s + j], pred[ri][s + j] = m, t
        k += guess.shape[0]
        n_windows += guess.shape[0]

    gold = [r["labels"] for r in rows]
    pred_tags = [[ID2LABEL[i] for i in p] for p in pred]
    result = score(gold, pred_tags)
    result["loss"] = total_loss / max(1, n_windows)
    if return_predictions:
        result["pred_tags"] = pred_tags
    return result


def fit(model, processor, train_rows: list[dict], val_rows: list[dict], images_dir: Path, out_dir: Path,
        hp: dict, device: str | None = None, num_workers: int = 0, seed: int = 42) -> dict:
    """Train, keep the checkpoint with the best validation micro-F1, write logs. Returns a summary dict.

    hp keys: epochs, learning_rate, batch_size, grad_accum_steps, weight_decay, warmup_ratio,
             early_stopping_patience, max_length, overlap_words
    """
    from transformers import get_linear_schedule_with_warmup

    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    train_ds = ReceiptWindows(train_rows, images_dir, processor, LABEL2ID, hp["max_length"], hp["overlap_words"])
    loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, num_workers=num_workers)
    accum = hp["grad_accum_steps"]
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = steps_per_epoch * hp["epochs"]

    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         "weight_decay": hp["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=hp["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(hp["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"device={device} | train receipts={len(train_rows)} windows={len(train_ds)} | "
          f"val receipts={len(val_rows)} | optimizer steps={total_steps}")

    log_path = out_dir / "training_log.csv"
    best_f1, best_epoch, bad_epochs, t0 = -1.0, 0, 0, time.time()
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_f1", "val_precision", "val_recall", "seconds"])

    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        running, n = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}/{hp['epochs']}", leave=False)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            running += loss.item() * batch["input_ids"].shape[0]
            n += batch["input_ids"].shape[0]
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        val = evaluate_rows(model, processor, val_rows, images_dir, device, hp["batch_size"],
                            hp["max_length"], hp["overlap_words"], num_workers)
        f1 = val["micro"]["f1"]
        print(f"epoch {epoch:>2}: train_loss {running / max(1, n):.4f} | val_loss {val['loss']:.4f} | "
              f"val_F1 {f1:.3f} (P {val['micro']['precision']:.3f} R {val['micro']['recall']:.3f})")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(running / max(1, n), 5), round(val["loss"], 5), round(f1, 4),
                                    round(val["micro"]["precision"], 4), round(val["micro"]["recall"], 4),
                                    round(time.time() - t0)])

        if f1 > best_f1:
            best_f1, best_epoch, bad_epochs = f1, epoch, 0
            best_dir = out_dir / "best"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            (best_dir / "label_map.json").write_text(json.dumps(LABEL2ID, indent=2))
            (out_dir / "val_metrics.json").write_text(json.dumps(val, indent=2))
        else:
            bad_epochs += 1
            if bad_epochs >= hp["early_stopping_patience"]:
                print(f"early stopping: no improvement for {bad_epochs} epochs")
                break

    summary = {"best_epoch": best_epoch, "best_val_f1": best_f1, "train_seconds": round(time.time() - t0),
               "hyperparameters": hp, "train_receipts": len(train_rows), "val_receipts": len(val_rows)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    best_val = json.loads((out_dir / "val_metrics.json").read_text())
    print("\nBest validation results (epoch %d):\n%s" % (best_epoch, format_report(best_val)))
    return summary
