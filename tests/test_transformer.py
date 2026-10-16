"""DistilBERT path with a TINY randomly initialised model and a local tokenizer: no downloads, CPU only.

This checks that the training loop, best-epoch saving, reload and prediction
work; the numbers it produces are meaningless and never reported.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from review_intel.data.pipeline import load_split  # noqa: E402
from review_intel.models.artifacts import load_model  # noqa: E402

pytestmark = pytest.mark.transformer


def tiny_model_and_tokenizer(texts):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

    from review_intel.models.transformer import ID2LABEL, LABEL2ID

    words = sorted({w for t in texts for w in t.lower().replace(",", " ").replace(".", " ").split()})
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, **{w: i + 4 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                        cls_token="[CLS]", sep_token="[SEP]")
    config = DistilBertConfig(vocab_size=len(vocab), dim=32, n_layers=1, n_heads=2, hidden_dim=64,
                              num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID, max_position_embeddings=128)
    torch.manual_seed(0)
    return DistilBertForSequenceClassification(config), tokenizer


def test_train_save_reload_predict(fixture_data, tmp_path):
    from review_intel.models.transformer import TransformerTrainingConfig, train_transformer

    texts, _ = load_split(fixture_data, "train")
    model, tokenizer = tiny_model_and_tokenizer(texts)
    cfg = TransformerTrainingConfig(epochs=2, batch_size=8, eval_batch_size=16, max_length=64, device="cpu",
                                    learning_rate=1e-3, name="tiny_distilbert")
    report = train_transformer(fixture_data, cfg, tmp_path / "models", tmp_path / "reports", model, tokenizer)

    out = tmp_path / "models" / "tiny_distilbert"
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["model_type"] == "transformer" and meta["label2id"] == {"negative": 0, "neutral": 1, "positive": 2}
    assert meta["training_config"]["max_length"] == 64 and len(meta["history"]) == 2
    assert 1 <= meta["best_epoch"] <= 2
    assert (out / "model.safetensors").exists() and (out / "tokenizer.json").exists()
    assert report["validation"]["n"] == 13
    assert (tmp_path / "reports" / "validation" / "tiny_distilbert.json").exists()

    loaded = load_model(out)  # dispatches on metadata model_type
    prediction = loaded.predict(["The battery drains way too fast on the Orbit Laptop."])[0]
    assert loaded.score_type == "softmax_probability"
    assert sum(prediction.scores.values()) == pytest.approx(1.0, abs=1e-4)
    assert prediction.label == max(prediction.scores, key=prediction.scores.get)
