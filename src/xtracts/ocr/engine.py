"""PaddleOCR wrapper that works with both PaddleOCR 2.x and 3.x.

Both versions are normalized to the same output:
    [{"text": str, "poly": [[x, y], ...4 points], "conf": float}, ...]

The first run downloads the detection/recognition models (needs internet).
"""
from __future__ import annotations

from typing import Callable

OcrEngine = Callable[[str], list[dict]]  # image path -> raw lines


class PaddleOCREngine:
    def __init__(self, lang: str = "en", use_gpu: bool = False, det_limit_side_len: int = 1600):
        import paddleocr  # imported lazily so the rest of the package works without it
        from paddleocr import PaddleOCR

        self.version = getattr(paddleocr, "__version__", "unknown")
        self.major = int(self.version.split(".")[0]) if self.version[0].isdigit() else 2
        self.name = f"paddleocr {self.version}"

        if self.major >= 3:
            self._ocr = PaddleOCR(
                lang=lang,
                device="gpu" if use_gpu else "cpu",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                text_det_limit_side_len=det_limit_side_len,
                text_det_limit_type="max",
            )
        else:
            self._ocr = PaddleOCR(
                lang=lang,
                use_angle_cls=True,
                use_gpu=use_gpu,
                show_log=False,
                det_limit_side_len=det_limit_side_len,
                det_limit_type="max",
            )

    def __call__(self, image_path: str) -> list[dict]:
        if self.major >= 3:
            results = self._ocr.predict(image_path)
            if not results:
                return []
            r = results[0]
            return [
                {"text": t, "poly": [[float(x), float(y)] for x, y in p], "conf": float(s)}
                for t, s, p in zip(r["rec_texts"], r["rec_scores"], r["rec_polys"])
            ]
        res = self._ocr.ocr(image_path, cls=True)
        if not res or res[0] is None:
            return []
        return [
            {"text": text, "poly": [[float(x), float(y)] for x, y in poly], "conf": float(score)}
            for poly, (text, score) in res[0]
        ]
