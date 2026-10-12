"""Sliding windows over a receipt's words so nothing is cut off at 512 tokens.

LayoutLMv3 reads at most 512 tokens. Long receipts exceed that, and plain truncation would
drop the BOTTOM of the receipt, which is exactly where the Total and Tax are. So long
receipts are split into overlapping windows of words; each window is one training sample.
"""
from __future__ import annotations


def word_token_counts(tokenizer, words: list[str]) -> list[int]:
    """How many sub-word tokens each word becomes (uses a fast LayoutLMv3 tokenizer)."""
    enc = tokenizer(words, boxes=[[0, 0, 0, 0]] * len(words), add_special_tokens=False)
    counts = [0] * len(words)
    for wid in enc.word_ids():
        if wid is not None:
            counts[wid] += 1
    return counts


def make_windows(counts: list[int], max_tokens: int, overlap_words: int = 30) -> list[tuple[int, int]]:
    """Greedy windows [start, end) over word indices, each with <= max_tokens tokens.

    Consecutive windows overlap by `overlap_words` words so a field near a boundary
    appears whole in at least one window.
    """
    n = len(counts)
    if n == 0:
        return []
    windows, start = [], 0
    while start < n:
        end, used = start, 0
        while end < n and used + counts[end] <= max_tokens:
            used += counts[end]
            end += 1
        if end == start:          # one word alone exceeds the budget: take it, the tokenizer truncates it
            end = start + 1
        windows.append((start, end))
        if end >= n:
            break
        start = max(end - overlap_words, start + 1)
    return windows
