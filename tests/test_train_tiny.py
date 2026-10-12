"""End-to-end check of dataset -> training -> evaluation on a TINY random LayoutLMv3 and a locally
trained tokenizer (no downloads). It verifies the plumbing, not accuracy."""
import json
import random

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from PIL import Image  # noqa: E402

from xtracts.annotation.labels import ENTITIES, ID2LABEL, LABEL2ID, LABEL_LIST  # noqa: E402

WORDS = ["FRESH", "MART", "Date:", "05/08/2026", "Item", "milk", "3.50", "bread", "2.00",
         "Tax", "0.33", "TOTAL", "5.50", "Thank", "you"]
TAGS = ["B-MERCHANT", "I-MERCHANT", "O", "B-DATE", "O", "O", "O", "O", "O",
        "O", "B-TAX", "O", "B-TOTAL", "O", "O"]


def make_rows(tmp_path, n, words_per_row=None):
    (tmp_path / "imgs").mkdir(exist_ok=True)
    rnd, rows = random.Random(0), []
    for i in range(n):
        words, tags = list(WORDS), list(TAGS)
        if words_per_row:                                   # a long receipt -> several windows
            words, tags = (words * 20)[:words_per_row], (tags * 20)[:words_per_row]
        Image.new("RGB", (400, 800), "white").save(tmp_path / "imgs" / f"r{i}.jpg")
        boxes = [[10 * (j % 50), 10 * (j // 5 % 90), 10 * (j % 50) + 40, 10 * (j // 5 % 90) + 12] for j in range(len(words))]
        rows.append({"image_id": f"r{i}", "image": f"r{i}.jpg", "words": words, "boxes": boxes,
                     "labels": tags, "merchant": "FRESH MART"})
        rnd.random()
    return rows


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import (LayoutLMv3Config, LayoutLMv3ForTokenClassification, LayoutLMv3ImageProcessor,
                              LayoutLMv3Processor, LayoutLMv3TokenizerFast)

    d = tmp_path_factory.mktemp("tok")
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator([" ".join(WORDS)] * 20, vocab_size=300, min_frequency=1,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bpe.save_model(str(d))
    tok = LayoutLMv3TokenizerFast(vocab_file=str(d / "vocab.json"), merges_file=str(d / "merges.txt"))
    processor = LayoutLMv3Processor(LayoutLMv3ImageProcessor(apply_ocr=False), tok)
    cfg = LayoutLMv3Config(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=130, coordinate_size=6, shape_size=4,
                           num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID)
    return LayoutLMv3ForTokenClassification(cfg), processor


HP = dict(epochs=2, learning_rate=1e-3, batch_size=2, grad_accum_steps=2, weight_decay=0.01,
          warmup_ratio=0.1, early_stopping_patience=3, max_length=64, overlap_words=5)


def test_dataset_windows_and_label_alignment(tiny, tmp_path):
    from xtracts.model.dataset import ReceiptWindows

    _, processor = tiny
    rows = make_rows(tmp_path, 1, words_per_row=120)          # too long for 64 tokens
    ds = ReceiptWindows(rows, tmp_path / "imgs", processor, LABEL2ID, max_length=64, overlap_words=5)
    assert len(ds) > 1                                         # split into windows, not truncated
    covered = sorted({w for _, s, e in ds.items for w in range(s, e)})
    assert covered == list(range(120))                         # every word, incl. the last, is in some window
    item = ds[0]
    assert item["input_ids"].shape == (64,) and item["bbox"].shape == (64, 4)
    assert item["pixel_values"].shape == (3, 224, 224)
    labelled = (item["labels"] != -100).sum().item()
    assert labelled == ds.items[0][2] - ds.items[0][1]         # exactly one label per word in the window
    assert item["bbox"].max() <= 1000


def test_fit_saves_best_model_and_scores(tiny, tmp_path):
    from transformers import LayoutLMv3ForTokenClassification
    from xtracts.model.train import evaluate_rows, fit

    model, processor = tiny
    rows = make_rows(tmp_path, 8)
    out = tmp_path / "out"
    summary = fit(model, processor, rows[:6], rows[6:], tmp_path / "imgs", out, HP, device="cpu")
    assert summary["best_epoch"] in (1, 2)
    for name in ("training_log.csv", "val_metrics.json", "summary.json", "best/model.safetensors",
                 "best/label_map.json", "best/config.json"):
        assert (out / name).exists(), name
    assert len((out / "training_log.csv").read_text().splitlines()) >= 2

    best = LayoutLMv3ForTokenClassification.from_pretrained(out / "best")
    res = evaluate_rows(best, processor, rows[6:], tmp_path / "imgs", "cpu", 2, 64, 5)
    assert set(res["fields"]) == set(ENTITIES)
    assert 0.0 <= res["micro"]["f1"] <= 1.0 and res["n_receipts"] == 2
    assert json.loads((out / "val_metrics.json").read_text())["n_receipts"] == 2


def test_model_can_overfit_tiny_data(tiny, tmp_path):
    """Sanity check that labels, boxes and the loss are wired correctly: on 4 identical receipts
    a tiny model must reach high F1 after enough epochs. Fails if labels are misaligned."""
    import torch

    from xtracts.model.train import evaluate_rows, fit

    model, processor = tiny
    torch.manual_seed(0)
    model = type(model)(model.config)          # fresh, seeded weights: independent of the other tests
    rows = make_rows(tmp_path, 4)
    # lr 3e-3 / 60 epochs reached F1 = 1.0 for 6 of 6 random initialisations (lr 1e-3 / 40 epochs only 2 of 6)
    hp = {**HP, "epochs": 60, "learning_rate": 3e-3, "early_stopping_patience": 60, "batch_size": 4,
          "grad_accum_steps": 1}
    fit(model, processor, rows, rows, tmp_path / "imgs", tmp_path / "o2", hp, device="cpu")
    res = evaluate_rows(model, processor, rows, tmp_path / "imgs", "cpu", 4, 64, 5)
    assert res["micro"]["f1"] > 0.9, res


def test_app_inference_with_model_and_with_fallback(tiny, tmp_path):
    """ReceiptExtractor end to end on the tiny model with a fake OCR engine (no PaddleOCR, no downloads)."""
    from xtracts.app.inference import ReceiptExtractor, predict_words
    from xtracts.config import load_config

    model, processor = tiny
    rows = make_rows(tmp_path, 1)
    img = Image.open(tmp_path / "imgs" / "r0.jpg").convert("RGB")
    tags, probs = predict_words(model, processor, img, rows[0]["words"], rows[0]["boxes"], "cpu", 64, 5)
    assert len(tags) == len(probs) == len(rows[0]["words"])
    assert all(t in ID2LABEL.values() for t in tags) and all(0 <= p <= 1 for p in probs)

    def fake_ocr(path):          # three receipt lines, in the format the OCR engine returns
        def line(t, y):
            return {"text": t, "poly": [[10, y], [300, y], [300, y + 20], [10, y + 20]], "conf": 0.99}
        return [line("FRESH MART SDN BHD", 10), line("Date: 05/08/2026", 60), line("TOTAL 5.50", 300)]

    cfg = load_config()
    cfg["training"] = {**cfg["training"], "max_length": 64, "overlap_words": 5}
    model.save_pretrained(tmp_path / "best")
    processor.save_pretrained(tmp_path / "best")
    ext = ReceiptExtractor(model_dir=tmp_path / "best", config=cfg, ocr_engine=fake_ocr, device="cpu")
    assert ext.mode == "model"
    res = ext.extract(tmp_path / "imgs" / "r0.jpg")
    assert res["mode"] == "model" and set(res["fields"]) == {"merchant", "date", "total", "tax"}

    fb = ReceiptExtractor(model_dir=tmp_path / "nothing_here", config=cfg, ocr_engine=fake_ocr)
    assert fb.mode == "rules" and fb.load_error
    res = fb.extract(tmp_path / "imgs" / "r0.jpg")
    assert res["mode"] == "rules"
    assert res["fields"]["total"] == 5.5 and res["fields"]["date"] == "2026-08-05"
