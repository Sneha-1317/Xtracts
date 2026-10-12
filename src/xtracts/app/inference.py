"""Receipt image -> OCR -> LayoutLMv3 -> fields.  This is the code path the app uses.

ReceiptExtractor.extract(path) returns
    {"mode": "model" | "rules", "fields": {merchant,date,total,tax}, "raw": {FIELD: text}, "n_words": int}
mode == "rules" means NO trained model was found, so the rule-based labeler (the one that produced the
annotation suggestions) is used instead. The app shows this clearly; it is not LayoutLMv3 output.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..annotation.labels import ID2LABEL, entities_to_bio, entity_values
from ..annotation.weak_labels import auto_label
from ..config import load_config, resolve
from ..model.chunking import make_windows, word_token_counts
from ..model.data_build import normalize_box
from ..ocr.postprocess import build_record
from .fields import clean_fields, tags_to_fields


def predict_words(model, processor, image: Image.Image, words: list[str], boxes: list[list[int]],
                  device, max_length: int = 512, overlap_words: int = 30) -> tuple[list[str], list[float]]:
    """One BIO tag + probability per word. Long receipts are read in overlapping windows; for a word seen in
    several windows the one where it is farthest from the window edge wins (same as evaluation)."""
    import torch

    if not words:
        return [], []
    counts = word_token_counts(processor.tokenizer, words)
    tags, probs, margin = ["O"] * len(words), [0.0] * len(words), [-1] * len(words)
    model.eval()
    for s, e in make_windows(counts, max_length - 2, overlap_words):
        enc = processor(image, words[s:e], boxes=boxes[s:e], truncation=True, padding="max_length",
                        max_length=max_length, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0]
        word_ids = enc.word_ids(0) if hasattr(enc, "word_ids") else None
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            p = model(**enc).logits[0].softmax(-1).cpu()
        # first sub-token of each word = one prediction per word (as during training)
        first, seen = [], set()
        if word_ids is not None:
            for t, w in enumerate(word_ids):
                if w is not None and w not in seen:
                    seen.add(w)
                    first.append(t)
        else:                                           # fallback: offsets start at 0 for a word's first token
            first = [t for t in range(len(offsets)) if offsets[t][0] == 0 and offsets[t][1] != 0]
        for j, t in enumerate(first[: e - s]):
            m = min(j, (e - s) - 1 - j)
            if m > margin[s + j]:
                k = int(p[t].argmax())
                margin[s + j], tags[s + j], probs[s + j] = m, ID2LABEL[k], float(p[t][k])
    return tags, probs


class ReceiptExtractor:
    def __init__(self, model_dir: str | Path | None = None, config: dict | None = None, ocr_engine=None,
                 device: str | None = None):
        self.cfg = config or load_config()
        self.hp = self.cfg["training"]
        self.model_dir = Path(model_dir) if model_dir else resolve(self.cfg["paths"]["model_dir"]) / "best"
        self._ocr, self._ocr_given = ocr_engine, ocr_engine is not None
        self.model = self.processor = None
        self.device = device
        self.load_error: str | None = None
        self._load_model()

    # ---------------------------------------------------------------- loading
    def _load_model(self) -> None:
        if not (self.model_dir / "config.json").exists():
            self.load_error = f"No trained model in {self.model_dir}"
            return
        try:
            import torch
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

            self.device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.processor = LayoutLMv3Processor.from_pretrained(self.model_dir, apply_ocr=False)
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(self.model_dir).to(self.device)
            self.model.eval()
        except Exception as e:  # missing torch, corrupt files...
            self.model = self.processor = None
            self.load_error = f"{type(e).__name__}: {e}"[:300]

    @property
    def mode(self) -> str:
        return "model" if self.model is not None else "rules"

    def _engine(self):
        if self._ocr is None:
            from ..ocr.engine import PaddleOCREngine

            o = self.cfg["ocr"]
            self._ocr = PaddleOCREngine(o["lang"], o["use_gpu"], o["det_limit_side_len"])
        return self._ocr

    # ---------------------------------------------------------------- inference
    def ocr_record(self, image_path: str | Path) -> dict:
        path = Path(image_path)
        with Image.open(path) as im:
            w, h = im.size
        raw = self._engine()(str(path))
        name = getattr(self._ocr, "name", "ocr")
        return build_record(path.stem, path.name, w, h, raw, name, self.cfg["ocr"]["min_conf"])

    def extract_from_record(self, rec: dict, image: Image.Image) -> dict:
        words = [w["text"] for w in rec["words"]]
        if not words:
            return {"mode": self.mode, "fields": clean_fields({}), "raw": {}, "n_words": 0}
        if self.model is not None:
            boxes = [normalize_box(w["box"], rec["width"], rec["height"]) for w in rec["words"]]
            tags, probs = predict_words(self.model, self.processor, image.convert("RGB"), words, boxes,
                                        self.device, self.hp["max_length"], self.hp["overlap_words"])
            raw = tags_to_fields(words, tags, probs)
        else:
            ents, _ = auto_label(rec)
            raw = entity_values(words, ents)
        return {"mode": self.mode, "fields": clean_fields(raw), "raw": raw, "n_words": len(words)}

    def extract(self, image_path: str | Path) -> dict:
        rec = self.ocr_record(image_path)
        with Image.open(image_path) as im:
            return self.extract_from_record(rec, im.convert("RGB"))
