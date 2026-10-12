"""Entity-level evaluation for BIO-tagged receipts (no external metrics package needed).

An entity is a run of words: B-X followed by I-X. A predicted entity counts as correct only if
its field AND exact word span match a gold entity (the usual seqeval / CoNLL definition).
"""
from __future__ import annotations

from ..annotation.labels import ENTITIES


def entity_spans(tags: list[str]) -> set[tuple[str, int, int]]:
    """BIO tags -> {(field, start, end_inclusive)}. A stray I-X with no open X entity starts one."""
    spans, cur = set(), None
    for i, t in enumerate(tags):
        if t == "O":
            if cur:
                spans.add(tuple(cur))
                cur = None
            continue
        prefix, field = t.split("-", 1)
        if prefix == "B" or cur is None or cur[0] != field:
            if cur:
                spans.add(tuple(cur))
            cur = [field, i, i]
        else:
            cur[2] = i
    if cur:
        spans.add(tuple(cur))
    return spans


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": tp + fn}


def score(gold: list[list[str]], pred: list[list[str]]) -> dict:
    """gold/pred: one BIO tag list per receipt (same lengths).

    Returns {"fields": {FIELD: precision/recall/f1/support/exact_match}, "micro": {...}, "n_receipts": n}
    exact_match = share of receipts whose predicted entities for that field equal the gold ones
    (a receipt with no gold and no predicted entity of the field counts as a match).
    """
    assert len(gold) == len(pred)
    counts = {f: [0, 0, 0] for f in ENTITIES}          # tp, fp, fn
    exact = {f: 0 for f in ENTITIES}
    for g_tags, p_tags in zip(gold, pred):
        g, p = entity_spans(g_tags), entity_spans(p_tags)
        for f in ENTITIES:
            gf = {s for s in g if s[0] == f}
            pf = {s for s in p if s[0] == f}
            counts[f][0] += len(gf & pf)
            counts[f][1] += len(pf - gf)
            counts[f][2] += len(gf - pf)
            exact[f] += gf == pf
    n = max(1, len(gold))
    fields = {f: {**_prf(*counts[f]), "exact_match": exact[f] / n} for f in ENTITIES}
    micro = _prf(*(sum(counts[f][k] for f in ENTITIES) for k in range(3)))
    return {"fields": fields, "micro": micro, "n_receipts": len(gold)}


def format_report(metrics: dict) -> str:
    lines = [f"{'field':<10}{'precision':>10}{'recall':>9}{'F1':>8}{'exact':>8}{'support':>9}"]
    for f, m in metrics["fields"].items():
        lines.append(f"{f:<10}{m['precision']:>10.3f}{m['recall']:>9.3f}{m['f1']:>8.3f}"
                     f"{m['exact_match']:>8.3f}{m['support']:>9d}")
    m = metrics["micro"]
    lines.append(f"{'ALL':<10}{m['precision']:>10.3f}{m['recall']:>9.3f}{m['f1']:>8.3f}{'':>8}{m['support']:>9d}")
    return "\n".join(lines)
