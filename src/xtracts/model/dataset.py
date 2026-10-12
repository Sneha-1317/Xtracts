"""PyTorch dataset: one sample per receipt window (see chunking.py). Needs torch + transformers."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .chunking import make_windows, word_token_counts


class ReceiptWindows(Dataset):
    def __init__(self, rows: list[dict], images_dir: Path, processor, label2id: dict[str, int],
                 max_length: int = 512, overlap_words: int = 30):
        self.rows, self.images_dir = rows, Path(images_dir)
        self.processor, self.label2id, self.max_length = processor, label2id, max_length
        budget = max_length - 2  # <s> and </s>
        self.items: list[tuple[int, int, int]] = []  # (row index, word start, word end)
        for ri, r in enumerate(rows):
            counts = word_token_counts(processor.tokenizer, r["words"])
            for s, e in make_windows(counts, budget, overlap_words):
                self.items.append((ri, s, e))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        ri, s, e = self.items[i]
        r = self.rows[ri]
        image = Image.open(self.images_dir / r["image"]).convert("RGB")
        enc = self.processor(
            image, r["words"][s:e], boxes=r["boxes"][s:e],
            word_labels=[self.label2id[l] for l in r["labels"][s:e]],
            truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}
