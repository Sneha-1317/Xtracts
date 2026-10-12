"""Leakage-aware train/val/test split.

Receipts from the same store look alike (same layout, same words). If some were in train and
others in test, the test score would be inflated. So receipts are split BY MERCHANT: every
receipt of a merchant lands in the same split.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

_SUFFIX = re.compile(r"\b(sdn|bhd|ltd|llp|llc|inc|corp|pvt|private|limited|co)\b")


def merchant_key(merchant_text: str, fallback: str) -> str:
    """Normalised merchant name used as the split group ('FRESH MART SDN. BHD.' -> 'freshmart').
    Receipts with no merchant label become their own group (key = fallback, the image id)."""
    t = _SUFFIX.sub(" ", merchant_text.lower())
    t = re.sub(r"[^a-z0-9]", "", t)
    return t or fallback


def grouped_split(groups: list[str], val_size: float = 0.15, test_size: float = 0.15,
                  seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx); no group appears in more than one split."""
    groups_arr = np.asarray(groups)
    if len(set(groups)) < 3:
        raise ValueError("Need at least 3 distinct merchants to make train/val/test splits.")
    idx = np.arange(len(groups_arr))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval, test = next(outer.split(idx, groups=groups_arr))
    inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    tr, va = next(inner.split(trainval, groups=groups_arr[trainval]))
    return trainval[tr], trainval[va], test
